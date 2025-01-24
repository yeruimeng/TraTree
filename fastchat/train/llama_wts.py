# wts的DPO代码  是eto那篇论文的复现
import os
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from fastchat.train.dpo_trainer import DPOMultiTrainer
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

# 设置固定的beta和learning rate
beta = 0.1
learning_rate = 2e-5

# 设置最大长度
max_length = 4096

# 设置训练参数
training_args = TrainingArguments(
    output_dir="/home/bhui/ML/ruimeng/ETO-main/webshop/qwen_ceiling",
    num_train_epochs=3,
    # per_device_train_batch_size=1,
    # gradient_accumulation_steps=16,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    save_steps=80,
    save_total_limit=5,
    learning_rate=learning_rate,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="constant_with_warmup",
    logging_steps=10,
    tf32=True,
    gradient_checkpointing=True,  # 启用梯度检查点
    max_grad_norm=0.5,  # 设置梯度的最大规范化值
)

# 调整 RoPE 缩放以适应更长的上下文长度
# 政策模型的基础路径（SFT 后的 LLaMA 13B 模型）
base_model_path = "/home/bhui/ML/ruimeng/ETO-main/webshop/qwen_14B_sft/merged_model"

# 参考模型的路径（SFT 后的 LLaMA 13B 模型）
ref_model_path = "/home/bhui/ML/ruimeng/ETO-main/webshop/qwen_14B_sft/merged_model"  # 请替换为实际的模型路径

# 加载政策模型的配置
policy_config = AutoConfig.from_pretrained(base_model_path)
orig_ctx_len = getattr(policy_config, "max_position_embeddings", None)
if orig_ctx_len and max_length > orig_ctx_len:
    scaling_factor = float(math.ceil(max_length / orig_ctx_len))
    policy_config.rope_scaling = {"type": "linear", "factor": scaling_factor}
policy_config.use_cache = False

# 加载政策模型（用于训练的策略模型）
policy_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    config=policy_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 为政策模型配置 LoRA
lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    # 128和256是sciworld的参数
    # r=128,
    # lora_alpha=128,
    # 128和128是webshop的参数
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# 将 LoRA 应用到政策模型
policy_model = get_peft_model(policy_model, lora_config)

# 加载参考模型的配置
ref_config = AutoConfig.from_pretrained(ref_model_path)
orig_ctx_len = getattr(ref_config, "max_position_embeddings", None)
if orig_ctx_len and max_length > orig_ctx_len:
    scaling_factor = float(math.ceil(max_length / orig_ctx_len))
    ref_config.rope_scaling = {"type": "linear", "factor": scaling_factor}
ref_config.use_cache = False

# 加载参考模型（SFT 后的 LLaMA 7B 模型）
ref_model = AutoModelForCausalLM.from_pretrained(
    ref_model_path,
    config=ref_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 将参考模型设置为不可训练
for param in ref_model.parameters():
    param.requires_grad = False

# 加载分词器并调整填充标记（使用政策模型的分词器）
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    model_max_length=max_length,
    padding_side="right",  # 根据需要调整
    use_fast=False,
)
if tokenizer.pad_token != tokenizer.unk_token:
    tokenizer.pad_token = tokenizer.unk_token

# 加载和预处理数据集
data_path = "/home/bhui/ML/ruimeng/ETO-main/data_pm/qwen_webshop_ceiling_strong+golden_pair.json"
dataset = load_dataset("json", data_files=data_path)

IGNORE_TOKEN_ID = -100  # LabelSmoother.ignore_index

def mask_labels(conversation, target, tokenizer, conv):
    if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
        sep = conv.sep + conv.roles[1] + ": "
    elif conv.sep_style == SeparatorStyle.LLAMA2:
        sep = conv.sep + conv.roles[1] + " "
    elif conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
        sep = conv.sep + conv.roles[1] + ":"
    elif conv.sep_style == SeparatorStyle.NO_COLON_SINGLE:
        sep = conv.sep + conv.roles[1]
    elif conv.sep_style == SeparatorStyle.ADD_COLON_TWO_NO_SPACE:
        sep = conv.sep + conv.roles[1] + ":"
    else:
        # 默认行为：使用通用分隔符
        sep = conv.sep + conv.roles[1]

    total_len = int(target.ne(tokenizer.pad_token_id).sum())

    turns = conversation.split(conv.sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_TOKEN_ID
    for i, turn in enumerate(turns):
        if turn == "":
            break

        turn_len = len(tokenizer(turn).input_ids) - 1

        parts = turn.split(sep, 1)  # 只分割一次，以防文本中包含分隔符

        if len(parts) != 2:
            # 如果分割失败，跳过这个 turn
            cur_len += turn_len
            continue

        instruction = parts[0] + sep
        response = parts[1]
        
        instruction_len = len(tokenizer(instruction).input_ids) - 1
        response_len = len(tokenizer(response).input_ids)

        # 忽略用户的指令部分
        target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
        cur_len += instruction_len + response_len

        # 增加 turn sep 的长度
        if conv.sep2:
            cur_len += len(tokenizer(conv.sep2).input_ids)

    target[cur_len:] = IGNORE_TOKEN_ID

    return target

def preprocess_function(example):
    conv = get_model_adapter(base_model_path).get_default_conv_template(base_model_path)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # 应用提示模板
    conv.messages = []
    for j, sentence in enumerate(example['prompt']):
        role = roles[sentence["from"]]
        conv.append_message(role, sentence["value"])
    prompt = conv.get_prompt()

    conv.messages = []
    for j, sentence in enumerate(example['prompt'] + example['chosen']):
        role = roles[sentence["from"]]
        conv.append_message(role, sentence["value"])
    chosen = conv.get_prompt()

    conv.messages = []
    for j, sentence in enumerate(example['prompt'] + example['rejected']):
        role = roles[sentence["from"]]
        conv.append_message(role, sentence["value"])
    rejected = conv.get_prompt()

    # 对话的分词
    prompt_tokens = tokenizer(prompt, return_tensors="pt")

    chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
    chosen_labels = chosen_tokens.input_ids[0].clone()
    chosen_labels = mask_labels(chosen, chosen_labels, tokenizer, conv)
    chosen_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

    rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)
    rejected_labels = rejected_tokens.input_ids[0].clone()
    rejected_labels = mask_labels(rejected, rejected_labels, tokenizer, conv)
    rejected_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

    return {
        "prompt_input_ids": prompt_tokens['input_ids'][0],
        "prompt_attention_mask": prompt_tokens['attention_mask'][0],
        "chosen_input_ids": chosen_tokens['input_ids'][0],
        "chosen_attention_mask": chosen_tokens['attention_mask'][0],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_tokens['input_ids'][0],
        "rejected_attention_mask": rejected_tokens['attention_mask'][0],
        "rejected_labels": rejected_labels,
    }

# 预处理数据集
preprocessed_dataset = dataset['train'].map(preprocess_function, remove_columns=dataset['train'].column_names)

# 初始化 DPOMultiTrainer
trainer = DPOMultiTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=training_args,
    beta=beta,
    train_dataset=preprocessed_dataset,
    tokenizer=tokenizer,
    max_length=max_length,
    max_prompt_length=512,
    max_target_length=3072,
    generate_during_eval=False,  # 训练评估时不生成文本
)

# 开始训练，支持从检查点恢复
if os.path.exists(os.path.join(training_args.output_dir, "checkpoint-last")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# 保存 DPO LoRA 权重
dpo_lora_path = os.path.join(training_args.output_dir, "final_lora")
trainer.model.save_pretrained(dpo_lora_path)

print("DPO 训练完成，LoRA 权重已保存。")

# 将 DPO LoRA 权重合并到政策模型
print("开始合并 DPO LoRA 权重到政策模型...")
# 加载政策模型的基础模型
base_policy_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    config=policy_config,
    torch_dtype=torch.float16,
)

# 加载 LoRA 权重到政策模型
peft_model = PeftModel.from_pretrained(base_policy_model, dpo_lora_path)
merged_model = peft_model.merge_and_unload()

# 保存最终合并后的模型
final_model_path = os.path.join(training_args.output_dir, "final_merged_model")
merged_model.save_pretrained(final_model_path)
tokenizer.save_pretrained(final_model_path)

print(f"合并完成。最终模型已保存至 {final_model_path}")