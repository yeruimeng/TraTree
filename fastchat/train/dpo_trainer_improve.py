# train_mcts_dpo_optimized.py
import os
import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    AutoConfig,
    AdamW,
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import torch.nn.functional as F
from torch import nn
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter
from transformers import default_data_collator

# 定义自定义数据集类
class MCTSImprovementDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=512):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def mask_labels(self, conversation, target, tokenizer, conv):
        sep = conv.sep
        roles = conv.roles
        sep_style = conv.sep_style

        if sep_style == SeparatorStyle.ADD_COLON_TWO:
            sep = conv.sep + roles[1] + ": "
        elif sep_style == SeparatorStyle.LLAMA2:
            sep = conv.sep + roles[1] + " "
        elif sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            sep = conv.sep + roles[1] + ":"
        elif sep_style == SeparatorStyle.NO_COLON_SINGLE:
            sep = conv.sep + roles[1]
        elif sep_style == SeparatorStyle.ADD_COLON_TWO_NO_SPACE:
            sep = conv.sep + roles[1] + ":"
        else:
            sep = conv.sep + roles[1]

        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = -100  # IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break

            parts = turn.split(sep, 1)
            if len(parts) != 2:
                cur_len += len(tokenizer(turn).input_ids) - 1
                continue

            instruction = parts[0] + sep
            response = parts[1]

            instruction_len = len(tokenizer(instruction).input_ids) - 1
            response_len = len(tokenizer(response).input_ids)

            # 忽略用户的指令部分
            target[cur_len : cur_len + instruction_len] = -100
            cur_len += instruction_len + response_len

            # 增加 turn sep 的长度
            if conv.sep2:
                cur_len += len(tokenizer(conv.sep2).input_ids)

        target[cur_len:] = -100  # IGNORE_TOKEN_ID

        return target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = sample['prompt']
        optimal_conversations = sample['optimal_conversations']
        R_diff = sample['reward_diff']

        # 获取会话模板
        conv = get_conversation_template("gpt2")  # 根据实际模型调整
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        # 构建 prompt 文本
        conv.messages = []
        for message in prompt:
            role = roles.get(message["from"], conv.roles[1])
            conv.append_message(role, message["value"])
        prompt_text = conv.get_prompt()

        # 构建 optimal_conversations 文本
        conv.messages = []
        for message in optimal_conversations:
            role = roles.get(message["from"], conv.roles[1])
            conv.append_message(role, message["value"])
        optimal_text = conv.get_prompt()

        # 编码输入和标签
        inputs = self.tokenizer(
            prompt_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

        labels = self.tokenizer(
            optimal_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )['input_ids'].squeeze()

        # 将标签中填充部分设置为 -100，以忽略它们的损失计算
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels,
            'reward_diff': torch.tensor(R_diff, dtype=torch.float)
        }

# 定义自定义损失函数
class MCTSLoss(nn.Module):
    def __init__(self, model, ref_model, tokenizer, beta=0.1):
        super(MCTSLoss, self).__init__()
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta

    def forward(self, input_ids, attention_mask, labels, reward_diff):
        # 当前模型的输出
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss_ce = outputs.loss  # 交叉熵损失

        # 参考模型的输出（注意不需要梯度）
        with torch.no_grad():
            ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            ref_logits = ref_outputs.logits

        # 当前模型的 logits
        logits = outputs.logits

        # 计算 KL 散度
        p = F.log_softmax(logits, dim=-1)
        q = F.softmax(ref_logits, dim=-1)
        kl_div = F.kl_div(p, q, reduction='batchmean')

        # 结合 R_diff 和 KL 散度
        # Lmcts = -(R_diff * loss_ce - beta * kl_div)
        # 为了最大化 R_diff - beta * KL，我们取负号转换为最小化
        loss = -(reward_diff * loss_ce - self.beta * kl_div)

        return loss

def main():
    # 配置参数
    data_path = 'contrastive_trajectoriess.json'  # 数据文件路径
    model_name = 'facebook/opt-13b'  # 替换为您使用的模型名称
    ref_model_name = 'facebook/opt-13b'  # 替换为参考模型名称
    batch_size = 2
    epochs = 3
    learning_rate = 2e-5
    beta = 0.1  # KL 散度的权重
    max_length = 4096

    # 设置训练输出目录
    output_dir = "/home/bhui/ML/ruimeng/ETO-main/dpo_output/mcts_dpo_training"

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载政策模型的配置
    policy_config = AutoConfig.from_pretrained(model_name)
    orig_ctx_len = getattr(policy_config, "max_position_embeddings", None)
    if orig_ctx_len and max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(max_length / orig_ctx_len))
        policy_config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    policy_config.use_cache = False

    # 加载政策模型（用于训练的策略模型），使用 device_map 自动分布到多个GPU
    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=policy_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 为政策模型配置 LoRA
    lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # 将 LoRA 应用到政策模型
    policy_model = get_peft_model(policy_model, lora_config)

    # 加载参考模型的配置
    ref_config = AutoConfig.from_pretrained(ref_model_name)
    orig_ctx_len = getattr(ref_config, "max_position_embeddings", None)
    if orig_ctx_len and max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(max_length / orig_ctx_len))
        ref_config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    ref_config.use_cache = False

    # 加载参考模型（SFT 后的 LLaMA 7B 模型），使用 device_map 自动分布到多个GPU
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name,
        config=ref_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # 将参考模型设置为不可训练
    for param in ref_model.parameters():
        param.requires_grad = False

    # 创建数据集和数据加载器
    dataset = MCTSImprovementDataset(data_path, tokenizer, max_length=512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=default_data_collator)

    # 定义损失函数
    criterion = MCTSLoss(model=policy_model, ref_model=ref_model, tokenizer=tokenizer, beta=beta)

    # 定义优化器
    optimizer = AdamW(policy_model.parameters(), lr=learning_rate)

    # 移动模型到设备（已经通过 device_map 分配）
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    policy_model.to(device)
    ref_model.to(device)

    # 启用梯度检查点（已经在模型加载时启用）

    # 训练循环
    policy_model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            reward_diff = batch['reward_diff'].to(device)

            optimizer.zero_grad()

            # 计算损失
            loss = criterion(input_ids, attention_mask, labels, reward_diff)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)

            # 优化器更新
            optimizer.step()

            # 记录损失
            epoch_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} - Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} finished with average loss: {avg_loss:.4f}")

        # 使用 Hugging Face 的 Trainer API 保存检查点
        # 仅保存最后一个 checkpoint 以节省空间
        checkpoint_dir = os.path.join(output_dir, f'checkpoint-epoch-{epoch + 1}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        policy_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        print(f"  Checkpoint saved at {checkpoint_dir}")

    print("Training completed.")

    # 保存最终的 LoRA 权重
    lora_save_path = os.path.join(output_dir, "final_lora")
    policy_model.save_pretrained(lora_save_path)
    print("LoRA 权重已保存。")

    # 将 LoRA 权重合并到政策模型
    print("开始合并 LoRA 权重到政策模型...")
    merged_model = policy_model.merge_and_unload()

    # 保存最终合并后的模型
    final_model_path = os.path.join(output_dir, "final_merged_model")
    merged_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"合并完成。最终模型已保存至 {final_model_path}")
