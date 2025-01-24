import json
import os
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from fastchat.train.dpo_trainer import DPOMultiTrainer
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

# 根据迭代次数调整 beta 和学习率
iteration = 1  # 根据需要设置迭代次数

if iteration == 1:
    beta = 0.1
    learning_rate = 1e-6
else:
    beta = 0.5
    learning_rate = 5e-7

# 设置最大长度
max_length = 4096

# 设置训练参数
training_args = TrainingArguments(
    output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/wts_DPO_sci_lora_step",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 根据GPU内存调整
    gradient_accumulation_steps=8,  # 相应调整
    save_steps=1000,
    save_total_limit=5,
    learning_rate=learning_rate,
    weight_decay=0.0,
    warmup_ratio=0.05,
    lr_scheduler_type="constant_with_warmup",
    logging_steps=10,
    tf32=True,
    gradient_checkpointing=True,  # 启用梯度检查点
    max_grad_norm=0.5,  # 设置梯度的最大规范化值
    remove_unused_columns=False,  # 设置为 False
)

# 调整 RoPE 缩放以适应更长的上下文长度
# 政策模型的基础路径（SFT 后的 LLaMA 13B 模型）
base_model_path = "/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_strong_llama2/merged_model"

# 参考模型的路径（SFT 后的 LLaMA 7B 模型）
ref_model_path = "/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model"  # 请替换为实际的模型路径

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
    lora_alpha=256,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.1,
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
data_path = "/home/bhui/ML/ruimeng/ETO-main/dpo_samples_flattened.json"  # 使用扁平化后的JSON文件路径
dataset = load_dataset("json", data_files=data_path)["train"]

IGNORE_TOKEN_ID = -100  # LabelSmoother.ignore_index

def mask_labels(labels, mask_length):
    """
    屏蔽标签中前 mask_length 个位置。
    """
    labels[:mask_length] = IGNORE_TOKEN_ID
    return labels

def has_rejected(example):
    """
    检查 rejected_labels 是否包含任何非 IGNORE_TOKEN_ID 的标签。
    """
    return any(label != IGNORE_TOKEN_ID for label in example['rejected_labels'])

def preprocess_function(example):
    conv = get_conversation_template(base_model_path)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    
    # 构建上下文 (prompt + history)
    for message in example['prompt']:
        role = roles.get(message["from"], message["from"])
        conv.append_message(role, message["value"])
    for message in example['history']:
        role = roles.get(message["from"], message["from"])
        conv.append_message(role, message["value"])
    context = conv.get_prompt()
    
    # Tokenize context
    context_tokens = tokenizer(context, return_tensors="pt")
    context_length = len(context_tokens['input_ids'][0])
    
    # 构建 chosen
    conv_chosen = get_conversation_template(base_model_path)
    for message in example['prompt']:
        role = roles.get(message["from"], message["from"])
        conv_chosen.append_message(role, message["value"])
    for message in example['history']:
        role = roles.get(message["from"], message["from"])
        conv_chosen.append_message(role, message["value"])
    for message in example['chosen']:
        role = roles.get(message["from"], message["from"])
        conv_chosen.append_message(role, message["value"])
    chosen = conv_chosen.get_prompt()
    
    # Tokenize chosen
    chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
    chosen_labels = chosen_tokens.input_ids[0].clone()
    chosen_labels = mask_labels(chosen_labels, context_length)
    chosen_labels = chosen_labels.tolist()  # 转换为列表
    
    # 构建 rejected
    conv_rejected = get_conversation_template(base_model_path)
    for message in example['prompt']:
        role = roles.get(message["from"], message["from"])
        conv_rejected.append_message(role, message["value"])
    for message in example['history']:
        role = roles.get(message["from"], message["from"])
        conv_rejected.append_message(role, message["value"])
    for message in example['rejected']:
        role = roles.get(message["from"], message["from"])
        conv_rejected.append_message(role, message["value"])
    rejected = conv_rejected.get_prompt()
    
    # Tokenize rejected
    rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)
    rejected_labels = rejected_tokens.input_ids[0].clone()
    rejected_labels = mask_labels(rejected_labels, context_length)
    rejected_labels = rejected_labels.tolist()  # 转换为列表
    
    return {
        "prompt_input_ids": context_tokens['input_ids'][0],
        "prompt_attention_mask": context_tokens['attention_mask'][0],
        "chosen_input_ids": chosen_tokens['input_ids'][0],
        "chosen_attention_mask": chosen_tokens['attention_mask'][0],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_tokens['input_ids'][0],
        "rejected_attention_mask": rejected_tokens['attention_mask'][0],
        "rejected_labels": rejected_labels,
    }

# 预处理数据集
preprocessed_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names, batched=False)

# 过滤掉 `rejected_labels` 全部为 IGNORE_TOKEN_ID 的样本
preprocessed_dataset = preprocessed_dataset.filter(has_rejected)

# 检查过滤后的数据集是否为空
if preprocessed_dataset.num_rows == 0:
    raise ValueError("过滤后的数据集为空。请检查数据预处理和过滤逻辑。")

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
