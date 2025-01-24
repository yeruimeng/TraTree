# # mcts树得到最好和最坏的两条轨迹
# import os
# import math
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
# from datasets import load_dataset
# from peft import get_peft_model, LoraConfig, TaskType, PeftModel
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from fastchat.conversation import SeparatorStyle
# from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

# # 根据迭代次数调整 beta 和学习率
# iteration = 1

# if iteration == 1:
#     beta = 0.1
#     learning_rate = 2e-5
# else:
#     beta = 0.5
#     learning_rate = 5e-7

# # 设置最大长度
# max_length = 4096

# # 设置训练参数
# training_args = TrainingArguments(
#     output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/DPO__mcts_5_pair_sciworld_ref_weak",
#     num_train_epochs=3,
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=16,
#     save_steps=80,
#     save_total_limit=5,
#     learning_rate=learning_rate,
#     weight_decay=0.01,
#     warmup_ratio=0.03,
#     lr_scheduler_type="constant_with_warmup",
#     logging_steps=10,
#     tf32=True,
#     gradient_checkpointing=True,
#     max_grad_norm=0.5,
# )

# # 模型路径设置
# base_model_path = "/home/bhui/ML/ruimeng/ETO-main/sci_output/lora_sft_sci_strong_llama2/merged_model"
# ref_model_path = "/home/bhui/ML/ruimeng/ETO-main/sci_output/lora_sft_sci_weak_llama2/merged_model"

# # 加载政策模型的配置
# policy_config = AutoConfig.from_pretrained(base_model_path)
# orig_ctx_len = getattr(policy_config, "max_position_embeddings", None)
# if orig_ctx_len and max_length > orig_ctx_len:
#     scaling_factor = float(math.ceil(max_length / orig_ctx_len))
#     policy_config.rope_scaling = {"type": "linear", "factor": scaling_factor}
# policy_config.use_cache = False

# # 加载政策模型
# policy_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     config=policy_config,
#     device_map="auto",
#     torch_dtype=torch.float16,
# )

# # LoRA 配置
# lora_config = LoraConfig(
#     r=128,
#     lora_alpha=256,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#     lora_dropout=0.0,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )

# # 应用 LoRA
# policy_model = get_peft_model(policy_model, lora_config)

# # 加载参考模型的配置
# ref_config = AutoConfig.from_pretrained(ref_model_path)
# orig_ctx_len = getattr(ref_config, "max_position_embeddings", None)
# if orig_ctx_len and max_length > orig_ctx_len:
#     scaling_factor = float(math.ceil(max_length / orig_ctx_len))
#     ref_config.rope_scaling = {"type": "linear", "factor": scaling_factor}
# ref_config.use_cache = False

# # 加载参考模型
# ref_model = AutoModelForCausalLM.from_pretrained(
#     ref_model_path,
#     config=ref_config,
#     device_map="auto",
#     torch_dtype=torch.float16,
# )

# # 设置参考模型为不可训练
# for param in ref_model.parameters():
#     param.requires_grad = False

# # 加载分词器
# tokenizer = AutoTokenizer.from_pretrained(
#     base_model_path,
#     model_max_length=max_length,
#     padding_side="right",
#     use_fast=False,
# )
# if tokenizer.pad_token != tokenizer.unk_token:
#     tokenizer.pad_token = tokenizer.unk_token

# # 加载数据集
# data_path = "/home/bhui/ML/ruimeng/ETO-main/contrastive_trajectoriess_V5.json"
# dataset = load_dataset("json", data_files=data_path)

# IGNORE_TOKEN_ID = -100

# def mask_labels(conversation, target, tokenizer, conv):
#     if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
#         sep = conv.sep + conv.roles[1] + ": "
#     elif conv.sep_style == SeparatorStyle.LLAMA2:
#         sep = conv.sep + conv.roles[1] + " "
#     elif conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
#         sep = conv.sep + conv.roles[1] + ":"
#     elif conv.sep_style == SeparatorStyle.NO_COLON_SINGLE:
#         sep = conv.sep + conv.roles[1]
#     elif conv.sep_style == SeparatorStyle.ADD_COLON_TWO_NO_SPACE:
#         sep = conv.sep + conv.roles[1] + ":"
#     else:
#         sep = conv.sep + conv.roles[1]

#     total_len = int(target.ne(tokenizer.pad_token_id).sum())

#     turns = conversation.split(conv.sep2)
#     cur_len = 1
#     target[:cur_len] = IGNORE_TOKEN_ID
#     for i, turn in enumerate(turns):
#         if turn == "":
#             break

#         turn_len = len(tokenizer(turn).input_ids) - 1

#         parts = turn.split(sep, 1)

#         if len(parts) != 2:
#             cur_len += turn_len
#             continue

#         instruction = parts[0] + sep
#         response = parts[1]
        
#         instruction_len = len(tokenizer(instruction).input_ids) - 1
#         response_len = len(tokenizer(response).input_ids)

#         target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
#         cur_len += instruction_len + response_len

#         if conv.sep2:
#             cur_len += len(tokenizer(conv.sep2).input_ids)

#     target[cur_len:] = IGNORE_TOKEN_ID

#     return target

# def preprocess_function(example):
#     conv = get_model_adapter(base_model_path).get_default_conv_template(base_model_path)
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # 处理提示对话
#     conv.messages = []
#     for sentence in example['prompt']:
#         role = roles[sentence["from"]]
#         conv.append_message(role, sentence["value"])
#     prompt = conv.get_prompt()

#     # 处理最优对话
#     conv.messages = []
#     for sentence in example['prompt'] + example['optimal_conversations']:
#         role = roles[sentence["from"]]
#         conv.append_message(role, sentence["value"])
#     chosen = conv.get_prompt()

#     # 处理最差对话
#     conv.messages = []
#     for sentence in example['prompt'] + example['worst_conversations']:
#         role = roles[sentence["from"]]
#         conv.append_message(role, sentence["value"])
#     rejected = conv.get_prompt()

#     # 分词处理
#     prompt_tokens = tokenizer(prompt, return_tensors="pt")

#     chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
#     chosen_labels = chosen_tokens.input_ids[0].clone()
#     chosen_labels = mask_labels(chosen, chosen_labels, tokenizer, conv)
#     chosen_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)
#     rejected_labels = rejected_tokens.input_ids[0].clone()
#     rejected_labels = mask_labels(rejected, rejected_labels, tokenizer, conv)
#     rejected_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     return {
#         "prompt_input_ids": prompt_tokens['input_ids'][0],
#         "prompt_attention_mask": prompt_tokens['attention_mask'][0],
#         "chosen_input_ids": chosen_tokens['input_ids'][0],
#         "chosen_attention_mask": chosen_tokens['attention_mask'][0],
#         "chosen_labels": chosen_labels,
#         "rejected_input_ids": rejected_tokens['input_ids'][0],
#         "rejected_attention_mask": rejected_tokens['attention_mask'][0],
#         "rejected_labels": rejected_labels,
#     }

# # 预处理数据集
# preprocessed_dataset = dataset['train'].map(
#     preprocess_function, 
#     remove_columns=dataset['train'].column_names,
#     num_proc=4  # 添加多进程处理以提高速度
# )

# # 初始化 DPOMultiTrainer
# trainer = DPOMultiTrainer(
#     model=policy_model,
#     ref_model=ref_model,
#     args=training_args,
#     beta=beta,
#     train_dataset=preprocessed_dataset,
#     tokenizer=tokenizer,
#     max_length=max_length,
#     max_prompt_length=512,
#     max_target_length=3072,
#     generate_during_eval=False,
# )

# # 训练
# if os.path.exists(os.path.join(training_args.output_dir, "checkpoint-last")):
#     trainer.train(resume_from_checkpoint=True)
# else:
#     trainer.train()

# # 保存 LoRA 权重
# dpo_lora_path = os.path.join(training_args.output_dir, "final_lora")
# trainer.model.save_pretrained(dpo_lora_path)

# print("DPO 训练完成，LoRA 权重已保存。")

# # 合并权重
# print("开始合并 DPO LoRA 权重到政策模型...")
# base_policy_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     config=policy_config,
#     torch_dtype=torch.float16,
# )

# peft_model = PeftModel.from_pretrained(base_policy_model, dpo_lora_path)
# merged_model = peft_model.merge_and_unload()

# # 保存最终模型
# final_model_path = os.path.join(training_args.output_dir, "final_merged_model")
# merged_model.save_pretrained(final_model_path)
# tokenizer.save_pretrained(final_model_path)

# print(f"合并完成。最终模型已保存至 {final_model_path}")


# # wts的DPO代码  是eto那篇论文的复现
import os
import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from fastchat.train.dpo_trainer import DPOMultiTrainer
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

# 根据迭代次数调整 beta 和学习率
iteration = 1  # 根据需要设置迭代次数

if iteration == 1:
    beta = 0.1
    # learning_rate = 1e-6
    learning_rate = 2e-5
else:
    beta = 0.5
    learning_rate = 5e-7

# 设置最大长度
max_length = 4096

# 设置训练参数
training_args = TrainingArguments(
    output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/DPO__mcts_5_pair_sciworld_ref_weak",
    num_train_epochs=3,
    # per_device_train_batch_size=1,
    # gradient_accumulation_steps=16,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
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
base_model_path = "/home/bhui/ML/ruimeng/ETO-main/sci_output/lora_sft_sci_strong_llama2/merged_model"

# 参考模型的路径（SFT 后的 LLaMA 13B 模型）
ref_model_path = "/home/bhui/ML/ruimeng/ETO-main/sci_output/lora_sft_sci_weak_llama2/merged_model"  # 请替换为实际的模型路径

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
data_path = "/home/bhui/ML/ruimeng/ETO-main/contrastive_trajectoriess_V5.json"
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


# 这个结果非常好   是对于模型自身的DPO的优化
# import os
# import math
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AutoConfig
# from datasets import load_dataset
# from peft import get_peft_model, LoraConfig, TaskType, PeftModel
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from fastchat.conversation import SeparatorStyle
# from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

# # 根据迭代次数调整 beta 和学习率
# iteration = 1  # 根据需要设置迭代次数

# if iteration == 1:
#     beta = 0.1
#     learning_rate = 1e-6
# else:
#     beta = 0.5
#     learning_rate = 5e-7

# # 设置最大长度
# max_length = 4096

# # 设置训练参数
# training_args = TrainingArguments(
#     output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/wts_DPO_sci_lora",
#     num_train_epochs=3,
#     per_device_train_batch_size=1,  
#     gradient_accumulation_steps=16,  
#     save_steps=500,
#     save_total_limit=5,
#     learning_rate=learning_rate,
#     weight_decay=0.0,
#     warmup_ratio=0.05,
#     lr_scheduler_type="constant_with_warmup",
#     logging_steps=10,
#     tf32=True,
#     gradient_checkpointing=True,  # 启用梯度检查点
#     max_grad_norm=0.5,  # 设置梯度的最大规范化值
# )

# # Adjust RoPE scaling for longer context lengths
# base_model_path = "/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_strong_llama2/merged_model"  # 使用适合你的基础模型路径
# config = AutoConfig.from_pretrained(base_model_path)
# orig_ctx_len = getattr(config, "max_position_embeddings", None)
# if orig_ctx_len and max_length > orig_ctx_len:
#     scaling_factor = float(math.ceil(max_length / orig_ctx_len))
#     config.rope_scaling = {"type": "linear", "factor": scaling_factor}
# config.use_cache = False

# # 加载基础模型（用于训练的policy model）
# policy_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path, 
#     config=config,
#     device_map="auto", 
#     torch_dtype=torch.float16,
# )

# # 为policy model配置LoRA
# lora_config = LoraConfig(
#     r=128,
#     lora_alpha=256,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )

# # 将LoRA应用到policy model
# policy_model = get_peft_model(policy_model, lora_config)

# # 加载参考模型（已包含SFT LoRA权重）
# ref_model_path = "/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_strong_llama2/merged_model"
# ref_model = AutoModelForCausalLM.from_pretrained(
#     ref_model_path, 
#     config=config,
#     device_map="auto", 
#     torch_dtype=torch.float16,
# )

# # 将参考模型设置为不可训练
# for param in ref_model.parameters():
#     param.requires_grad = False

# # 加载分词器并调整填充标记
# tokenizer = AutoTokenizer.from_pretrained(
#     ref_model_path, 
#     model_max_length=max_length,
#     padding_side="right",  # 根据需要调整
#     use_fast=False,
# )
# if tokenizer.pad_token != tokenizer.unk_token:
#     tokenizer.pad_token = tokenizer.unk_token

# # 加载和预处理数据集
# data_path = "/home/bhui/ML/ruimeng/ETO-main/data_pm/wts_weak_pair.json"
# dataset = load_dataset("json", data_files=data_path)

# IGNORE_TOKEN_ID = -100  # LabelSmoother.ignore_index

# def mask_labels(conversation, target, tokenizer, conv):
#     if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
#         sep = conv.sep + conv.roles[1] + ": "
#     elif conv.sep_style == SeparatorStyle.LLAMA2:
#         sep = conv.sep + conv.roles[1] + " "
#     elif conv.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
#         sep = conv.sep + conv.roles[1] + ":"
#     elif conv.sep_style == SeparatorStyle.NO_COLON_SINGLE:
#         sep = conv.sep + conv.roles[1]
#     elif conv.sep_style == SeparatorStyle.ADD_COLON_TWO_NO_SPACE:
#         sep = conv.sep + conv.roles[1] + ":"
#     else:
#         # 默认行为：使用通用分隔符
#         sep = conv.sep + conv.roles[1]
    
#     total_len = int(target.ne(tokenizer.pad_token_id).sum())

#     turns = conversation.split(conv.sep2)
#     cur_len = 1
#     target[:cur_len] = IGNORE_TOKEN_ID
#     for i, turn in enumerate(turns):
#         if turn == "":
#             break

#         turn_len = len(tokenizer(turn).input_ids) - 1

#         parts = turn.split(sep, 1)  # 只分割一次，以防文本中包含分隔符

#         if len(parts) != 2:
#             # 如果分割失败，跳过这个 turn
#             cur_len += turn_len
#             continue

#         instruction = parts[0] + sep
#         response = parts[1]
        
#         instruction_len = len(tokenizer(instruction).input_ids) - 1
#         response_len = len(tokenizer(response).input_ids)

#         # 忽略用户的指令部分
#         target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
#         cur_len += instruction_len + response_len

#         # 增加 turn sep 的长度
#         if conv.sep2:
#             cur_len += len(tokenizer(conv.sep2).input_ids)

#     target[cur_len:] = IGNORE_TOKEN_ID

#     return target

# def preprocess_function(example):
#     conv = get_model_adapter(base_model_path).get_default_conv_template(base_model_path)
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # 应用提示模板
#     conv.messages = []
#     for j, sentence in enumerate(example['prompt']):
#         role = roles[sentence["from"]]
#         conv.append_message(role, sentence["value"])
#     prompt = conv.get_prompt()

#     conv.messages = []
#     for j, sentence in enumerate(example['prompt'] + example['chosen']):
#         role = roles[sentence["from"]]
#         conv.append_message(role, sentence["value"])
#     chosen = conv.get_prompt()

#     conv.messages = []
#     for j, sentence in enumerate(example['prompt'] + example['rejected']):
#         role = roles[sentence["from"]]
#         conv.append_message(role, sentence["value"])
#     rejected = conv.get_prompt()

#     # 对话的分词
#     prompt_tokens = tokenizer(prompt, return_tensors="pt")

#     chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
#     chosen_labels = chosen_tokens.input_ids[0].clone()
#     chosen_labels = mask_labels(chosen, chosen_labels, tokenizer, conv)
#     chosen_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)
#     rejected_labels = rejected_tokens.input_ids[0].clone()
#     rejected_labels = mask_labels(rejected, rejected_labels, tokenizer, conv)
#     rejected_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     return {
#         "prompt_input_ids": prompt_tokens['input_ids'][0],
#         "prompt_attention_mask": prompt_tokens['attention_mask'][0],
#         "chosen_input_ids": chosen_tokens['input_ids'][0],
#         "chosen_attention_mask": chosen_tokens['attention_mask'][0],
#         "chosen_labels": chosen_labels,
#         "rejected_input_ids": rejected_tokens['input_ids'][0],
#         "rejected_attention_mask": rejected_tokens['attention_mask'][0],
#         "rejected_labels": rejected_labels,
#     }

# # 预处理数据集
# preprocessed_dataset = dataset['train'].map(preprocess_function, remove_columns=dataset['train'].column_names)

# # 初始化DPOMultiTrainer
# trainer = DPOMultiTrainer(
#     model=policy_model,
#     ref_model=ref_model,
#     args=training_args,
#     beta=beta,
#     train_dataset=preprocessed_dataset,
#     tokenizer=tokenizer,
#     max_length=max_length,
#     max_prompt_length=512,
#     max_target_length=3072,
#     generate_during_eval=False,  # 训练评估时不生成文本
# )

# # 开始训练，支持从检查点恢复
# if os.path.exists(os.path.join(training_args.output_dir, "checkpoint-last")):
#     trainer.train(resume_from_checkpoint=True)
# else:
#     trainer.train()

# # 保存DPO LoRA权重
# dpo_lora_path = os.path.join(training_args.output_dir, "final_lora")
# trainer.model.save_pretrained(dpo_lora_path)

# print("DPO训练完成，LoRA权重已保存。")

# # 将DPO LoRA权重合并到参考模型
# print("开始合并DPO LoRA权重到参考模型...")
# ref_model = PeftModel.from_pretrained(ref_model, dpo_lora_path)
# merged_model = ref_model.merge_and_unload()

# # 保存最终合并后的模型
# final_model_path = os.path.join(training_args.output_dir, "final_merged_model")
# merged_model.save_pretrained(final_model_path)

# print(f"合并完成。最终模型已保存至 {final_model_path}")



# 这个卡空下来的时候试一下
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from datasets import load_dataset
# import torch
# import numpy as np
# from sklearn.metrics import accuracy_score


# # 加载预训练模型和tokenizer
# model_path = "/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model"
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
# ref_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)

# # 将参考模型设置为不可训练
# for param in ref_model.parameters():
#     param.requires_grad = False

# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # 加载数据集
# data_path = "/home/bhui/ML/ruimeng/ETO-main/data_pm/7B_weak+golden_pm.json"
# dataset = load_dataset("json", data_files=data_path)

# # 划分训练集和验证集
# train_val_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
# train_dataset = train_val_dataset['train']
# val_dataset = train_val_dataset['test']

# def preprocess_function(example):
#     def format_conversation(messages):
#         conversation = []
#         for msg in messages:
#             if msg['from'] == 'human':
#                 conversation.append({"role": "user", "content": msg['value']})
#             elif msg['from'] == 'gpt':
#                 conversation.append({"role": "assistant", "content": msg['value']})
#         return conversation

#     prompt = format_conversation(example['prompt'])
#     chosen = format_conversation(example['chosen'])
#     rejected = format_conversation(example['rejected'])

#     prompt_str = format_chat(prompt)
#     chosen_str = format_chat(prompt + chosen)
#     rejected_str = format_chat(prompt + rejected)

#     # 对 prompt、chosen 和 rejected 进行编码
#     prompt_tokens = tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True)
#     chosen_tokens = tokenizer(chosen_str, return_tensors="pt", padding=True, truncation=True)
#     rejected_tokens = tokenizer(rejected_str, return_tensors="pt", padding=True, truncation=True)

#     # 创建 labels
#     chosen_labels = chosen_tokens.input_ids.clone()
#     chosen_labels[chosen_labels == tokenizer.pad_token_id] = -100
#     chosen_labels[:, :len(prompt_tokens.input_ids[0])] = -100

#     rejected_labels = rejected_tokens.input_ids.clone()
#     rejected_labels[rejected_labels == tokenizer.pad_token_id] = -100
#     rejected_labels[:, :len(prompt_tokens.input_ids[0])] = -100

#     return {
#         "prompt_input_ids": prompt_tokens.input_ids[0],
#         "prompt_attention_mask": prompt_tokens.attention_mask[0],
#         "chosen_input_ids": chosen_tokens.input_ids[0],
#         "chosen_attention_mask": chosen_tokens.attention_mask[0],
#         "chosen_labels": chosen_labels[0],
#         "rejected_input_ids": rejected_tokens.input_ids[0],
#         "rejected_attention_mask": rejected_tokens.attention_mask[0],
#         "rejected_labels": rejected_labels[0],
#     }

# def format_chat(messages):
#     formatted = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
#     for i, message in enumerate(messages):
#         if message["role"] == "user":
#             formatted += f"{message['content']} [/INST] "
#         elif message["role"] == "assistant":
#             formatted += f"{message['content']} </s><s>[INST] "
#     formatted = formatted.rstrip(" </s><s>[INST] ")
#     return formatted

# # 预处理数据集
# preprocessed_train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
# preprocessed_val_dataset = val_dataset.map(preprocess_function, remove_columns=val_dataset.column_names)

# # 定义动态参数调整函数
# def get_dynamic_params(iteration):
#     if iteration == 1:
#         return {"beta": 0.1, "lr": 1e-6}
#     else:
#         return {"beta": 0.5, "lr": 5e-7}

# # 设置训练参数
# def get_training_args(iteration, output_dir):
#     dynamic_params = get_dynamic_params(iteration)
#     return TrainingArguments(
#         output_dir=f"{output_dir}/iteration_{iteration}",
#         num_train_epochs=3,
#         per_device_train_batch_size=1,
#         per_device_eval_batch_size=1,
#         gradient_accumulation_steps=64,  # 梯度累积
#         save_strategy="steps",
#         save_steps=500,
#         evaluation_strategy="steps",
#         eval_steps=500,
#         save_total_limit=5,
#         learning_rate=dynamic_params["lr"],
#         weight_decay=0.01,  # 增加权重衰减
#         warmup_ratio=0.1,  # 学习率预热
#         lr_scheduler_type="cosine",  # 使用余弦学习率调度
#         logging_steps=10,
#         tf32=True,
#         gradient_checkpointing=True,
#         max_grad_norm=0.5,
#         metric_for_best_model="eval_loss",
#         load_best_model_at_end=True,
#     )

# # 定义评价函数
# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     return {"accuracy": accuracy_score(labels.flatten(), predictions.flatten())}

# # 主训练循环
# best_eval_loss = float("inf")
# best_iteration = None

# for iteration in range(1, 4):  # 3次迭代
#     print(f"Starting iteration {iteration}")
    
#     training_args = get_training_args(iteration, "/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_2")
#     dynamic_params = get_dynamic_params(iteration)

#     # 初始化DPOMultiTrainer
#     trainer = DPOMultiTrainer(
#         model=model,
#         ref_model=ref_model,
#         args=training_args,
#         beta=dynamic_params["beta"],
#         train_dataset=preprocessed_train_dataset,
#         eval_dataset=preprocessed_val_dataset,
#         tokenizer=tokenizer,
#         max_length=4096,
#         max_prompt_length=512,
#         max_target_length=3072,
#         generate_during_eval=False,
#         compute_metrics=compute_metrics,
#     )

#     # 开始训练
#     trainer.train()

#     # 检查当前训练的评价损失
#     eval_metrics = trainer.evaluate()
#     eval_loss = eval_metrics.get("eval_loss")

#     # 如果这是当前最好的模型，则记录当前迭代
#     if eval_loss is not None and eval_loss < best_eval_loss:
#         best_eval_loss = eval_loss
#         best_iteration = iteration
#         print(f"New best model found at iteration {iteration} with eval loss: {eval_loss}")

# # 保存效果最好的模型
# if best_iteration is not None:
#     trainer.save_model(f"/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_1/best_model_iteration_{best_iteration}")
#     print(f"Best model saved from iteration {best_iteration}")
# else:
#     print("No improvement found during training.")

# print("Training completed")

# # 这个结果很不行  怀疑有严重的overfit
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from datasets import load_dataset
# import torch

# # 加载预训练模型和tokenizer
# model_path = "/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model"
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
# # ref_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
# ref_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)


# # 将参考模型设置为不可训练
# for param in ref_model.parameters():
#     param.requires_grad = False

# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # 加载和预处理数据集
# data_path = "/home/bhui/ML/ruimeng/ETO-main/data_pm/7B_weak+golden_pm.json"
# dataset = load_dataset("json", data_files=data_path)

# def preprocess_function(example):
#     def format_conversation(messages):
#         conversation = []
#         for msg in messages:
#             if msg['from'] == 'human':
#                 conversation.append({"role": "user", "content": msg['value']})
#             elif msg['from'] == 'gpt':
#                 conversation.append({"role": "assistant", "content": msg['value']})
#         return conversation

#     prompt = format_conversation(example['prompt'])
#     chosen = format_conversation(example['chosen'])
#     rejected = format_conversation(example['rejected'])

#     prompt_str = format_chat(prompt)
#     chosen_str = format_chat(prompt + chosen)
#     rejected_str = format_chat(prompt + rejected)

#     # 对 prompt、chosen 和 rejected 进行编码
#     prompt_tokens = tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True)
#     chosen_tokens = tokenizer(chosen_str, return_tensors="pt", padding=True, truncation=True)
#     rejected_tokens = tokenizer(rejected_str, return_tensors="pt", padding=True, truncation=True)

#     # 创建 labels
#     chosen_labels = chosen_tokens.input_ids.clone()
#     chosen_labels[chosen_labels == tokenizer.pad_token_id] = -100
#     chosen_labels[:, :len(prompt_tokens.input_ids[0])] = -100

#     rejected_labels = rejected_tokens.input_ids.clone()
#     rejected_labels[rejected_labels == tokenizer.pad_token_id] = -100
#     rejected_labels[:, :len(prompt_tokens.input_ids[0])] = -100

#     return {
#         "prompt_input_ids": prompt_tokens.input_ids[0],
#         "prompt_attention_mask": prompt_tokens.attention_mask[0],
#         "chosen_input_ids": chosen_tokens.input_ids[0],
#         "chosen_attention_mask": chosen_tokens.attention_mask[0],
#         "chosen_labels": chosen_labels[0],
#         "rejected_input_ids": rejected_tokens.input_ids[0],
#         "rejected_attention_mask": rejected_tokens.attention_mask[0],
#         "rejected_labels": rejected_labels[0],
#     }

# def format_chat(messages):
#     formatted = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
#     for i, message in enumerate(messages):
#         if message["role"] == "user":
#             formatted += f"{message['content']} [/INST] "
#         elif message["role"] == "assistant":
#             formatted += f"{message['content']} </s><s>[INST] "
#     formatted = formatted.rstrip(" </s><s>[INST] ")
#     return formatted

# # 预处理数据集
# preprocessed_dataset = dataset['train'].map(preprocess_function, remove_columns=dataset['train'].column_names)

# # 设置训练参数
# training_args = TrainingArguments(
#     output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_1",
#     num_train_epochs=3,
#     per_device_train_batch_size=1,  
#     gradient_accumulation_steps=8,  
#     save_steps=500,
#     save_total_limit=5,
#     learning_rate=5e-6,
#     weight_decay=0.0,
#     warmup_ratio=0.05,
#     lr_scheduler_type="constant_with_warmup",
#     logging_steps=10,
#     tf32=True,
#     gradient_checkpointing=True,  # 启用渐度检查点
#     max_grad_norm=0.5,  # 设置渐度的最大规正化值
# )

# # 初始化DPOMultiTrainer
# trainer = DPOMultiTrainer(
#     model=model,
#     ref_model=ref_model,
#     args=training_args,
#     beta=0.1,
#     train_dataset=preprocessed_dataset,
#     tokenizer=tokenizer,
#     max_length=4096,
#     max_prompt_length=512,
#     max_target_length=3072,
#     generate_during_eval=False,  # 训练评估时生成文本
# )

# # 开始训练
# trainer.train()

# 这个结果也不太好  要优化一下  但可以考虑换成Adam优化器之类的  因为过拟合现象有所解决
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from datasets import load_dataset
# import torch
# from peft import get_peft_model, LoraConfig, TaskType, PeftModel

# # 加载参考模型（已包含SFT LoRA权重）
# ref_model_path = "/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model"
# ref_model = AutoModelForCausalLM.from_pretrained(ref_model_path, device_map="auto", torch_dtype=torch.float16)

# # 将参考模型设置为不可训练
# for param in ref_model.parameters():
#     param.requires_grad = False

# # 加载基础模型（用于训练的policy model）
# base_model_path = "/home/bhui/ML/ruimeng/ETO-main/Llama-2-7b-chat-hf"  # 使用适合你的基础模型路径
# policy_model = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto", torch_dtype=torch.float16)

# # 为policy model配置LoRA
# lora_config = LoraConfig(
#     r=128,
#     lora_alpha=256,
#     target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
#     lora_dropout=0.1,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )

# # 将LoRA应用到policy model
# policy_model = get_peft_model(policy_model, lora_config)

# tokenizer = AutoTokenizer.from_pretrained(ref_model_path)

# # 加载和预处理数据集
# data_path = "/home/bhui/ML/ruimeng/ETO-main/data_pm/7B_weak+golden_pm.json"
# dataset = load_dataset("json", data_files=data_path)

# def preprocess_function(example):
#     def format_conversation(messages):
#         conversation = []
#         for msg in messages:
#             if msg['from'] == 'human':
#                 conversation.append({"role": "user", "content": msg['value']})
#             elif msg['from'] == 'gpt':
#                 conversation.append({"role": "assistant", "content": msg['value']})
#         return conversation

#     prompt = format_conversation(example['prompt'])
#     chosen = format_conversation(example['chosen'])
#     rejected = format_conversation(example['rejected'])

#     prompt_str = format_chat(prompt)
#     chosen_str = format_chat(prompt + chosen)
#     rejected_str = format_chat(prompt + rejected)

#     # 对 prompt、chosen 和 rejected 进行编码
#     prompt_tokens = tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True)
#     chosen_tokens = tokenizer(chosen_str, return_tensors="pt", padding=True, truncation=True)
#     rejected_tokens = tokenizer(rejected_str, return_tensors="pt", padding=True, truncation=True)

#     # 创建 labels
#     chosen_labels = chosen_tokens.input_ids.clone()
#     chosen_labels[chosen_labels == tokenizer.pad_token_id] = -100
#     chosen_labels[:, :len(prompt_tokens.input_ids[0])] = -100

#     rejected_labels = rejected_tokens.input_ids.clone()
#     rejected_labels[rejected_labels == tokenizer.pad_token_id] = -100
#     rejected_labels[:, :len(prompt_tokens.input_ids[0])] = -100

#     return {
#         "prompt_input_ids": prompt_tokens.input_ids[0],
#         "prompt_attention_mask": prompt_tokens.attention_mask[0],
#         "chosen_input_ids": chosen_tokens.input_ids[0],
#         "chosen_attention_mask": chosen_tokens.attention_mask[0],
#         "chosen_labels": chosen_labels[0],
#         "rejected_input_ids": rejected_tokens.input_ids[0],
#         "rejected_attention_mask": rejected_tokens.attention_mask[0],
#         "rejected_labels": rejected_labels[0],
#     }

# def format_chat(messages):
#     formatted = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
#     for i, message in enumerate(messages):
#         if message["role"] == "user":
#             formatted += f"{message['content']} [/INST] "
#         elif message["role"] == "assistant":
#             formatted += f"{message['content']} </s><s>[INST] "
#     formatted = formatted.rstrip(" </s><s>[INST] ")
#     return formatted

# # 预处理数据集
# preprocessed_dataset = dataset['train'].map(preprocess_function, remove_columns=dataset['train'].column_names)

# # 设置训练参数
# training_args = TrainingArguments(
#     output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_lora",
#     num_train_epochs=3,
#     per_device_train_batch_size=2,  
#     gradient_accumulation_steps=8,  
#     save_steps=500,
#     save_total_limit=5,
#     learning_rate=1e-5,
#     weight_decay=0.0,
#     warmup_ratio=0.05,
#     lr_scheduler_type="constant_with_warmup",
#     logging_steps=10,
#     tf32=True,
#     gradient_checkpointing=True,  # 启用梯度检查点
#     max_grad_norm=0.5,  # 设置梯度的最大规范化值
# )

# # 初始化DPOMultiTrainer
# trainer = DPOMultiTrainer(
#     model=policy_model,
#     ref_model=ref_model,
#     args=training_args,
#     beta=0.1,
#     train_dataset=preprocessed_dataset,
#     tokenizer=tokenizer,
#     max_length=4096,
#     max_prompt_length=512,
#     max_target_length=3072,
#     generate_during_eval=False,  # 训练评估时不生成文本
# )

# # 开始训练
# trainer.train()

# # 保存DPO LoRA权重
# dpo_lora_path = "/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_lora/final_lora"
# trainer.model.save_pretrained(dpo_lora_path)

# print("DPO训练完成，LoRA权重已保存。")

# # 将DPO LoRA权重合并到参考模型
# print("开始合并DPO LoRA权重到参考模型...")
# ref_model = PeftModel.from_pretrained(ref_model, dpo_lora_path)
# merged_model = ref_model.merge_and_unload()

# # 保存最终合并后的模型
# final_model_path = "/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_lora/final_merged_model"
# merged_model.save_pretrained(final_model_path)

# print(f"合并完成。最终模型已保存至 {final_model_path}")



# 这个是简单写的DPO 能跑但是效果不好
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from datasets import load_dataset

# # 加载预训练模型和tokenizer
# model_path = "/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model"
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# # 加载和预处理数据集
# data_path = "/home/bhui/ML/ruimeng/ETO-main/data_pm/7B_weak+golden_pm.json"
# dataset = load_dataset("json", data_files=data_path)

# def preprocess_function(example):
#     def format_conversation(messages):
#         conversation = []
#         for msg in messages:
#             if msg['from'] == 'human':
#                 conversation.append({"role": "user", "content": msg['value']})
#             elif msg['from'] == 'gpt':
#                 conversation.append({"role": "assistant", "content": msg['value']})
#         return conversation

#     prompt = format_conversation(example['prompt'])
#     chosen = format_conversation(example['chosen'])
#     rejected = format_conversation(example['rejected'])

#     prompt_str = format_chat(prompt)
#     chosen_str = format_chat(prompt + chosen)
#     rejected_str = format_chat(prompt + rejected)

#     # 对 prompt、chosen 和 rejected 进行编码
#     prompt_tokens = tokenizer(prompt_str, return_tensors="pt", padding=True, truncation=True)
#     chosen_tokens = tokenizer(chosen_str, return_tensors="pt", padding=True, truncation=True)
#     rejected_tokens = tokenizer(rejected_str, return_tensors="pt", padding=True, truncation=True)

#     # 创建 labels
#     chosen_labels = chosen_tokens.input_ids.clone()
#     chosen_labels[chosen_labels == tokenizer.pad_token_id] = -100
#     chosen_labels[:, :len(prompt_tokens.input_ids[0])] = -100

#     rejected_labels = rejected_tokens.input_ids.clone()
#     rejected_labels[rejected_labels == tokenizer.pad_token_id] = -100
#     rejected_labels[:, :len(prompt_tokens.input_ids[0])] = -100

#     return {
#         "prompt_input_ids": prompt_tokens.input_ids[0],
#         "prompt_attention_mask": prompt_tokens.attention_mask[0],
#         "chosen_input_ids": chosen_tokens.input_ids[0],
#         "chosen_attention_mask": chosen_tokens.attention_mask[0],
#         "chosen_labels": chosen_labels[0],
#         "rejected_input_ids": rejected_tokens.input_ids[0],
#         "rejected_attention_mask": rejected_tokens.attention_mask[0],
#         "rejected_labels": rejected_labels[0],
#     }

# def format_chat(messages):
#     formatted = "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
#     for i, message in enumerate(messages):
#         if message["role"] == "user":
#             formatted += f"{message['content']} [/INST] "
#         elif message["role"] == "assistant":
#             formatted += f"{message['content']} </s><s>[INST] "
#     formatted = formatted.rstrip(" </s><s>[INST] ")
#     return formatted

# # 预处理数据集
# preprocessed_dataset = dataset['train'].map(preprocess_function, remove_columns=dataset['train'].column_names)

# # 设置训练参数
# training_args = TrainingArguments(
#     output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_1",
#     num_train_epochs=3,
#     per_device_train_batch_size=1,  
#     gradient_accumulation_steps=8,  
#     save_steps=500,
#     save_total_limit=5,
#     learning_rate=5e-6,
#     weight_decay=0.0,
#     warmup_ratio=0.1,
#     lr_scheduler_type="constant_with_warmup",
#     logging_steps=10,
#     tf32=True,
#     gradient_checkpointing=True,  # 启用梯度检查点
# )

# # 初始化DPOMultiTrainer
# trainer = DPOMultiTrainer(
#     model=model,
#     args=training_args,
#     beta=0.1,
#     train_dataset=preprocessed_dataset,
#     tokenizer=tokenizer,
#     max_length=4096,
#     max_prompt_length=512,
#     max_target_length=3072,
# )

# # 开始训练
# trainer.train()


# 这段用的是加载融合后的模型 添加一个新的lora层 grad_nam变成了inf
# import sys
# print(sys.path)
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from datasets import load_dataset
# import torch
# import os
# from dataclasses import dataclass, field
# from typing import Optional, Dict, List
# import math
# from fastchat.conversation import SeparatorStyle, get_conv_template
# from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training

# IGNORE_TOKEN_ID = -100

# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(default="/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model")
#     ref_model_name_or_path: Optional[str] = field(default="/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model")
#     trust_remote_code: bool = field(default=False)
#     use_flash_attention_2: bool = field(default=False)

# @dataclass
# class DataArguments:
#     data_path: str = field(default="/home/bhui/ML/ruimeng/ETO-main/data_pm/7B_weak+golden_pm.json")

# @dataclass
# class TrainingArguments(TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="paged_adamw_8bit")
#     model_max_length: int = field(default=4096)
#     max_prompt_length: int = field(default=512)
#     max_target_length: int = field(default=3072)

# def get_rope_scaling(model_max_length, orig_ctx_len):
#     if model_max_length > orig_ctx_len:
#         scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
#         return {"type": "linear", "factor": scaling_factor}
#     return None

# def mask_labels(conversation, target, tokenizer, conv):
#     sep = conv.sep + conv.roles[1] + (": " if conv.sep_style == SeparatorStyle.ADD_COLON_TWO else " ")
#     total_len = int(target.ne(tokenizer.pad_token_id).sum())

#     turns = conversation.split(conv.sep2)
#     cur_len = 1
#     target[:cur_len] = IGNORE_TOKEN_ID
#     for i, turn in enumerate(turns):
#         if not turn:
#             break

#         turn_len = len(tokenizer(turn).input_ids) - 1
#         parts = turn.split(sep)
#         if len(parts) != 2:
#             break
#         parts[0] += sep

#         instruction_len = len(tokenizer(parts[0]).input_ids) - 2
#         if i != 0 and conv.roles[0] == 'USER':
#             instruction_len -= 1

#         target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
#         cur_len += turn_len + (3 if conv.sep2 == ' </s><s>' else 1)
#         if i != 0 and conv.roles[0] == 'USER':
#             cur_len -= 1

#     target[cur_len:] = IGNORE_TOKEN_ID

#     if cur_len < tokenizer.model_max_length and cur_len != total_len:
#         target[:] = IGNORE_TOKEN_ID
#         print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. #turn = {len(turns) - 1}. (ignored)")

#     return target

# def preprocess_function(
#     example: Dict,
#     tokenizer: AutoTokenizer,
#     max_length: int,
#     max_prompt_length: int,
#     conv_template: str,
# ) -> Dict:
#     conv = get_conv_template(conv_template)
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     def format_example(messages: List[Dict]) -> str:
#         conv.messages = []
#         for i, message in enumerate(messages):
#             role = roles[message['from']]
#             assert role == conv.roles[i % 2]
#             conv.append_message(role, message['value'])
#         return conv.get_prompt()

#     prompt = format_example(example['prompt'])
#     chosen = format_example(example['prompt'] + example['chosen'])
#     rejected = format_example(example['prompt'] + example['rejected'])

#     prompt_tokens = tokenizer(prompt, return_tensors="pt", max_length=max_prompt_length, truncation=True)
#     chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
#     rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)

#     chosen_labels = mask_labels(chosen, chosen_tokens.input_ids[0].clone(), tokenizer, conv)
#     rejected_labels = mask_labels(rejected, rejected_tokens.input_ids[0].clone(), tokenizer, conv)

#     return {
#         "prompt_input_ids": prompt_tokens.input_ids[0],
#         "prompt_attention_mask": prompt_tokens.attention_mask[0],
#         "chosen_input_ids": chosen_tokens.input_ids[0],
#         "chosen_attention_mask": chosen_tokens.attention_mask[0],
#         "chosen_labels": chosen_labels,
#         "rejected_input_ids": rejected_tokens.input_ids[0],
#         "rejected_attention_mask": rejected_tokens.attention_mask[0],
#         "rejected_labels": rejected_labels,
#     }

# def create_peft_config(model):
#     peft_config = LoraConfig(
#         task_type="CAUSAL_LM",
#         r=8,  # 减小 r 值
#         lora_alpha=16,  # 设置为 r 的 4 倍
#         lora_dropout=0.1,
#         bias="none",
#         target_modules=["q_proj", "v_proj"]
#     )
#     model = prepare_model_for_kbit_training(model)
#     model = get_peft_model(model, peft_config)
#     return model

# def check_and_set_trainable_parameters(model):
#     trainable_params = 0
#     all_param = 0
#     for name, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#         else:
#             param.requires_grad = True  # 确保所有参数都是可训练的
    
#     print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

# def main():
#     model_args = ModelArguments()
#     data_args = DataArguments()
#     training_args = TrainingArguments(
#         output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_weak+golden_lora",
#         num_train_epochs=3,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=8,
#         save_steps=500,
#         save_total_limit=5,
#         learning_rate=5e-5,  # 进一步增加学习率
#         weight_decay=0.01,
#         warmup_ratio=0.1,
#         max_grad_norm=1.0,
#         lr_scheduler_type="cosine",
#         logging_steps=1,  # 更频繁的日志记录
#         tf32=True,
#         gradient_checkpointing=True,
#         fp16=True,
#         optim="adamw_torch"  # 使用 adamw_torch 而不是 paged_adamw_8bit
#     )


#     # 加载模型和分词器
#     base_model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         trust_remote_code=model_args.trust_remote_code,
#         device_map="auto",
#         torch_dtype=torch.float16
#     )
#     model = create_peft_config(base_model)
    
#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

#     # 设置 RoPE 缩放因子
#     config = model.config
#     orig_ctx_len = getattr(config, "max_position_embeddings", None)
#     if orig_ctx_len:
#         config.rope_scaling = get_rope_scaling(training_args.model_max_length, orig_ctx_len)

#     # 加载参考模型
#     if model_args.ref_model_name_or_path:
#         ref_base_model = AutoModelForCausalLM.from_pretrained(
#             model_args.ref_model_name_or_path,
#             trust_remote_code=model_args.trust_remote_code,
#             device_map="auto",
#             torch_dtype=torch.float16
#         )
#         ref_model = create_peft_config(ref_base_model)
#     else:
#         ref_model = None

#     # 检查和设置可训练参数
#     print("Checking model parameters:")
#     check_and_set_trainable_parameters(model)

#     # 加载和预处理数据集
#     dataset = load_dataset("json", data_files=data_args.data_path)
#     conv_template = "llama-2"

#     preprocess = lambda example: preprocess_function(
#         example,
#         tokenizer,
#         training_args.model_max_length,
#         training_args.max_prompt_length,
#         conv_template
#     )

#     preprocessed_dataset = dataset['train'].map(
#         preprocess,
#         remove_columns=dataset['train'].column_names,
#         batched=False,
#         num_proc=4
#     )
    
#     def print_trainable_parameters(model):
#         trainable_params = 0
#         all_param = 0
#         for _, param in model.named_parameters():
#             all_param += param.numel()
#             if param.requires_grad:
#                 trainable_params += param.numel()
#         print(
#             f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#         )

#     # 初始化 DPOMultiTrainer
#     trainer = DPOMultiTrainer(
#         model=model,
#         ref_model=ref_model,
#         args=training_args,
#         beta=0.1,  # 根据论文，将 beta 设置为 0.1
#         train_dataset=preprocessed_dataset,
#         tokenizer=tokenizer,
#         max_length=training_args.model_max_length,
#         max_prompt_length=training_args.max_prompt_length,
#         max_target_length=training_args.max_target_length,
#     )

#     # 开始训练
#     trainer.train()

#     # 保存 LoRA 权重
#     trainer.model.save_pretrained(os.path.join(training_args.output_dir, "lora_weights"))

#     # 合并 LoRA 权重到基础模型并保存
#     merged_model = trainer.model.merge_and_unload()
#     merged_model.save_pretrained(os.path.join(training_args.output_dir, "merged_model"))
#     tokenizer.save_pretrained(os.path.join(training_args.output_dir, "merged_model"))

# if __name__ == "__main__":
#     main()

# import sys
# print(sys.path)
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from datasets import load_dataset
# import torch
# import os
# from dataclasses import dataclass, field
# from typing import Optional, Dict, List
# import math
# from fastchat.conversation import SeparatorStyle, get_conv_template
# from peft import PeftModel, prepare_model_for_kbit_training

# IGNORE_TOKEN_ID = -100

# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(default="/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model")
#     ref_model_name_or_path: Optional[str] = field(default="/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model")
#     trust_remote_code: bool = field(default=False)
#     use_flash_attention_2: bool = field(default=False)

# @dataclass
# class DataArguments:
#     data_path: str = field(default="/home/bhui/ML/ruimeng/ETO-main/data_pm/7B_weak+golden_pm.json")

# @dataclass
# class CustomTrainingArguments(TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="paged_adamw_8bit")
#     model_max_length: int = field(default=4096)
#     max_prompt_length: int = field(default=512)
#     max_target_length: int = field(default=3072)

# def get_rope_scaling(model_max_length, orig_ctx_len):
#     if model_max_length > orig_ctx_len:
#         scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
#         return {"type": "linear", "factor": scaling_factor}
#     return None

# def mask_labels(conversation, target, tokenizer, conv):
#     sep = conv.sep + conv.roles[1] + (": " if conv.sep_style == SeparatorStyle.ADD_COLON_TWO else " ")
#     total_len = int(target.ne(tokenizer.pad_token_id).sum())

#     turns = conversation.split(conv.sep2)
#     cur_len = 1
#     target[:cur_len] = IGNORE_TOKEN_ID
#     for i, turn in enumerate(turns):
#         if not turn:
#             break

#         turn_len = len(tokenizer(turn).input_ids) - 1
#         parts = turn.split(sep)
#         if len(parts) != 2:
#             break
#         parts[0] += sep

#         instruction_len = len(tokenizer(parts[0]).input_ids) - 2
#         if i != 0 and conv.roles[0] == 'USER':
#             instruction_len -= 1

#         target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
#         cur_len += turn_len + (3 if conv.sep2 == ' </s><s>' else 1)
#         if i != 0 and conv.roles[0] == 'USER':
#             cur_len -= 1

#     target[cur_len:] = IGNORE_TOKEN_ID

#     if cur_len < tokenizer.model_max_length and cur_len != total_len:
#         target[:] = IGNORE_TOKEN_ID
#         print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. #turn = {len(turns) - 1}. (ignored)")

#     return target

# def preprocess_function(
#     example: Dict,
#     tokenizer: AutoTokenizer,
#     max_length: int,
#     max_prompt_length: int,
#     conv_template: str,
# ) -> Dict:
#     conv = get_conv_template(conv_template)
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     def format_example(messages: List[Dict]) -> str:
#         conv.messages = []
#         for i, message in enumerate(messages):
#             role = roles[message['from']]
#             assert role == conv.roles[i % 2]
#             conv.append_message(role, message['value'])
#         return conv.get_prompt()

#     prompt = format_example(example['prompt'])
#     chosen = format_example(example['prompt'] + example['chosen'])
#     rejected = format_example(example['prompt'] + example['rejected'])

#     prompt_tokens = tokenizer(prompt, return_tensors="pt", max_length=max_prompt_length, truncation=True)
#     chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
#     rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)

#     chosen_labels = mask_labels(chosen, chosen_tokens.input_ids[0].clone(), tokenizer, conv)
#     rejected_labels = mask_labels(rejected, rejected_tokens.input_ids[0].clone(), tokenizer, conv)

#     return {
#         "prompt_input_ids": prompt_tokens.input_ids[0],
#         "prompt_attention_mask": prompt_tokens.attention_mask[0],
#         "chosen_input_ids": chosen_tokens.input_ids[0],
#         "chosen_attention_mask": chosen_tokens.attention_mask[0],
#         "chosen_labels": chosen_labels,
#         "rejected_input_ids": rejected_tokens.input_ids[0],
#         "rejected_attention_mask": rejected_tokens.attention_mask[0],
#         "rejected_labels": rejected_labels,
#     }

# def main():
#     model_args = ModelArguments()
#     data_args = DataArguments()
#     training_args = CustomTrainingArguments(
#         output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_weak+golden",
#         num_train_epochs=3,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         save_steps=500,
#         save_total_limit=5,
#         learning_rate=1e-5,
#         weight_decay=0.0,
#         warmup_ratio=0.1,
#         max_grad_norm=1.0,
#         lr_scheduler_type="constant_with_warmup",
#         logging_steps=10,
#         tf32=True,
#         gradient_checkpointing=True,
#         fp16=True,
#         optim="paged_adamw_8bit"
#     )

#     # 加载模型和分词器
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         trust_remote_code=model_args.trust_remote_code,
#         device_map="auto",
#         torch_dtype=torch.float16
#     )
#     model = prepare_model_for_kbit_training(model)  # 这里不再添加新的 LoRA 层
    
#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

#     # 设置 RoPE 缩放因子
#     config = model.config
#     orig_ctx_len = getattr(config, "max_position_embeddings", None)
#     if orig_ctx_len:
#         config.rope_scaling = get_rope_scaling(training_args.model_max_length, orig_ctx_len)

#     # 加载参考模型
#     if model_args.ref_model_name_or_path:
#         ref_model = AutoModelForCausalLM.from_pretrained(
#             model_args.ref_model_name_or_path,
#             trust_remote_code=model_args.trust_remote_code,
#             device_map="auto",
#             torch_dtype=torch.float16
#         )
#         ref_model = prepare_model_for_kbit_training(ref_model)  # 参考模型同样不添加新的 LoRA 层
#     else:
#         ref_model = None

#     # 加载和预处理数据集
#     dataset = load_dataset("json", data_files=data_args.data_path)
#     conv_template = "llama-2"

#     preprocess = lambda example: preprocess_function(
#         example,
#         tokenizer,
#         training_args.model_max_length,
#         training_args.max_prompt_length,
#         conv_template
#     )

#     preprocessed_dataset = dataset['train'].map(
#         preprocess,
#         remove_columns=dataset['train'].column_names,
#         batched=False,
#         num_proc=4
#     )

#     # 初始化 DPOMultiTrainer
#     trainer = DPOMultiTrainer(
#         model=model,
#         ref_model=ref_model,
#         args=training_args,
#         beta=0.1,  # 根据论文，将 beta 设置为 0.1
#         train_dataset=preprocessed_dataset,
#         tokenizer=tokenizer,
#         max_length=training_args.model_max_length,
#         max_prompt_length=training_args.max_prompt_length,
#         max_target_length=training_args.max_target_length,
#     )

#     # 开始训练
#     trainer.train()

#     # 保存合并后的模型
#     trainer.model.save_pretrained(os.path.join(training_args.output_dir, "merged_model"))
#     tokenizer.save_pretrained(os.path.join(training_args.output_dir, "merged_model"))

# if __name__ == "__main__":
#     main()
    



# #这个可以跑但是有点大了
# import sys
# print(sys.path)
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
# from datasets import load_dataset
# import torch
# from dataclasses import dataclass, field
# from typing import Optional, Dict, List
# import math
# from fastchat.conversation import SeparatorStyle, get_conv_template

# IGNORE_TOKEN_ID = -100

# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(default="/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model")
#     ref_model_name_or_path: Optional[str] = field(default="/home/bhui/ML/ruimeng/ETO-main/output/lora_sft_sci_weak_llama2/merged_model")
#     trust_remote_code: bool = field(default=False)
#     use_flash_attention_2: bool = field(default=False)

# @dataclass
# class DataArguments:
#     data_path: str = field(default="/home/bhui/ML/ruimeng/ETO-main/data_pm/7B_weak+golden_pm.json")

# @dataclass
# class TrainingArguments(TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     model_max_length: int = field(default=4096)
#     max_prompt_length: int = field(default=512)
#     max_target_length: int = field(default=3072)

# def get_rope_scaling(model_max_length, orig_ctx_len):
#     if model_max_length > orig_ctx_len:
#         scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
#         return {"type": "linear", "factor": scaling_factor}
#     return None

# def mask_labels(conversation, target, tokenizer, conv):
#     sep = conv.sep + conv.roles[1] + (": " if conv.sep_style == SeparatorStyle.ADD_COLON_TWO else " ")
#     total_len = int(target.ne(tokenizer.pad_token_id).sum())

#     turns = conversation.split(conv.sep2)
#     cur_len = 1
#     target[:cur_len] = IGNORE_TOKEN_ID
#     for i, turn in enumerate(turns):
#         if not turn:
#             break

#         turn_len = len(tokenizer(turn).input_ids) - 1
#         parts = turn.split(sep)
#         if len(parts) != 2:
#             break
#         parts[0] += sep

#         instruction_len = len(tokenizer(parts[0]).input_ids) - 2
#         if i != 0 and conv.roles[0] == 'USER':
#             instruction_len -= 1

#         target[cur_len: cur_len + instruction_len] = IGNORE_TOKEN_ID
#         cur_len += turn_len + (3 if conv.sep2 == ' </s><s>' else 1)
#         if i != 0 and conv.roles[0] == 'USER':
#             cur_len -= 1

#     target[cur_len:] = IGNORE_TOKEN_ID

#     if cur_len < tokenizer.model_max_length and cur_len != total_len:
#         target[:] = IGNORE_TOKEN_ID
#         print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. #turn = {len(turns) - 1}. (ignored)")

#     return target

# def preprocess_function(
#     example: Dict,
#     tokenizer: AutoTokenizer,
#     max_length: int,
#     max_prompt_length: int,
#     conv_template: str,
# ) -> Dict:
#     conv = get_conv_template(conv_template)
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     def format_example(messages: List[Dict]) -> str:
#         conv.messages = []
#         for i, message in enumerate(messages):
#             role = roles[message['from']]
#             assert role == conv.roles[i % 2]
#             conv.append_message(role, message['value'])
#         return conv.get_prompt()

#     prompt = format_example(example['prompt'])
#     chosen = format_example(example['prompt'] + example['chosen'])
#     rejected = format_example(example['prompt'] + example['rejected'])

#     prompt_tokens = tokenizer(prompt, return_tensors="pt", max_length=max_prompt_length, truncation=True)
#     chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=max_length, truncation=True)
#     rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=max_length, truncation=True)

#     chosen_labels = mask_labels(chosen, chosen_tokens.input_ids[0].clone(), tokenizer, conv)
#     rejected_labels = mask_labels(rejected, rejected_tokens.input_ids[0].clone(), tokenizer, conv)

#     return {
#         "prompt_input_ids": prompt_tokens.input_ids[0],
#         "prompt_attention_mask": prompt_tokens.attention_mask[0],
#         "chosen_input_ids": chosen_tokens.input_ids[0],
#         "chosen_attention_mask": chosen_tokens.attention_mask[0],
#         "chosen_labels": chosen_labels,
#         "rejected_input_ids": rejected_tokens.input_ids[0],
#         "rejected_attention_mask": rejected_tokens.attention_mask[0],
#         "rejected_labels": rejected_labels,
#     }

# def main():
#     model_args = ModelArguments()
#     data_args = DataArguments()
#     training_args = TrainingArguments(
#         output_dir="/home/bhui/ML/ruimeng/ETO-main/dpo_output/7B_DPO_sci_weak+golden",
#         num_train_epochs=3,
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=4,
#         save_steps=500,
#         save_total_limit=5,
#         learning_rate=1e-6,
#         weight_decay=0.0,
#         warmup_ratio=0.1,
#         max_grad_norm=1.0,
#         lr_scheduler_type="constant_with_warmup",
#         logging_steps=10,
#         tf32=True,
#         gradient_checkpointing=True,
#     )

#     # 加载模型和分词器
#     model = AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         trust_remote_code=model_args.trust_remote_code,
#         device_map="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

#     # 设置 RoPE 缩放因子
#     config = model.config
#     orig_ctx_len = getattr(config, "max_position_embeddings", None)
#     if orig_ctx_len:
#         config.rope_scaling = get_rope_scaling(training_args.model_max_length, orig_ctx_len)

#     # 加载参考模型
#     if model_args.ref_model_name_or_path:
#         ref_model = AutoModelForCausalLM.from_pretrained(
#             model_args.ref_model_name_or_path,
#             trust_remote_code=model_args.trust_remote_code,
#             device_map="auto"
#         )
#     else:
#         ref_model = None

#     # 加载和预处理数据集
#     dataset = load_dataset("json", data_files=data_args.data_path)
#     conv_template = "llama-2"

#     preprocess = lambda example: preprocess_function(
#         example,
#         tokenizer,
#         training_args.model_max_length,
#         training_args.max_prompt_length,
#         conv_template
#     )

#     preprocessed_dataset = dataset['train'].map(
#         preprocess,
#         remove_columns=dataset['train'].column_names,
#         batched=False,
#         num_proc=4
#     )

#     # 初始化 DPOMultiTrainer
#     trainer = DPOMultiTrainer(
#         model=model,
#         ref_model=ref_model,
#         args=training_args,
#         beta=0.1,  # 根据论文，将 beta 设置为 0.1
#         train_dataset=preprocessed_dataset,
#         tokenizer=tokenizer,
#         max_length=training_args.model_max_length,
#         max_prompt_length=training_args.max_prompt_length,
#         max_target_length=training_args.max_target_length,
#     )

#     # 开始训练
#     trainer.train()

#     # 保存最终模型
#     trainer.save_model(training_args.output_dir)
#     tokenizer.save_pretrained(training_args.output_dir)

# if __name__ == "__main__":
#     main()







# 有grad_norm等于0的问题
# # ./train.sh > DPO.txt 2>&1
# import sys
# print(sys.path)
# from dataclasses import dataclass, field
# import json
# import math
# import pathlib
# from typing import Dict, Optional, Sequence
# from functools import partial

# import numpy as np
# import torch
# from torch.utils.data import Dataset

# import transformers
# from transformers import Trainer
# from transformers.trainer_pt_utils import LabelSmoother
# # from fastchat.train.dpo_trainer import DPOTrainer
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from datasets import load_dataset

# from fastchat.conversation import SeparatorStyle
# from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

# IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default="/home/bhui/ML/ruimeng/ETO-main/output/weak_sft_llama2")
#     ref_model_name_or_path: Optional[str] = field(
#         default="/home/bhui/ML/ruimeng/ETO-main/output/weak_sft_llama2",
#         metadata={"help": "Path to the reference model"}
#     )
#     trust_remote_code: bool = field(
#         default=None,
#         metadata={
#             "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
#         },
#     )
#     padding_side: str = field(
#         default="right", metadata={"help": "The padding side in tokenizer"}
#     )
#     beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

# @dataclass
# class DataArguments:
#     data_path: str = field(
#         default="/home/bhui/ML/ruimeng/ETO-main/data_pm/webshop_weak_model_pm(weak+ground_truth).json", 
#         metadata={"help": "Path to the training data."}
#     )
#     lazy_preprocess: bool = False

# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     model_max_length: int = field(
#         default=4096,
#         metadata={
#             "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
#         },
#     )
#     max_prompt_length: int = field(
#         default=512,
#         metadata={
#             "help": "Maximum prompt length."
#         },
#     )
#     max_target_length: int = field(
#         default=3072,
#         metadata={
#             "help": "Maximum target length."
#         },
#     )
#     gradient_accumulation_steps: int = field(
#         default=16,
#         metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
#     )
#     evaluation_strategy: str = field(
#         default="no",
#         metadata={"help": "The evaluation strategy to use."},
#     )
#     save_strategy: str = field(
#         default="steps",
#         metadata={"help": "The checkpoint save strategy to use."},
#     )
#     save_steps: int = field(
#         default=500,
#         metadata={"help": "Save checkpoint every X updates steps."},
#     )

# def mask_labels(conversation, target, tokenizer, conv):
#     print(f"Conversation: {conversation}")
#     print(f"Separator: {conv.sep}")
#     print(f"Roles: {conv.roles}")
#     if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
#         sep = conv.sep + conv.roles[1] + ": "
#     elif conv.sep_style == SeparatorStyle.LLAMA2:
#         sep = conv.sep + conv.roles[1] + " "
#     elif conv.sep_style == SeparatorStyle.CHATGLM:
#         sep = conv.sep
#     elif conv.sep_style == SeparatorStyle.CHATML:
#         sep = conv.sep + conv.roles[1] + "\n"
#     else:
#         # raise NotImplementedError
#         print(f"Unhandled separator style: {conv.sep_style}")
#         sep = conv.sep + conv.roles[1] + ": "  # 默认分隔符
    
#     total_len = int(target.ne(tokenizer.pad_token_id).sum())

#     turns = conversation.split(conv.sep2)
#     cur_len = 1
#     target[:cur_len] = IGNORE_TOKEN_ID
#     for i, turn in enumerate(turns):
#         if turn == "":
#             break

#         turn_len = len(tokenizer(turn).input_ids) - 1

#         parts = turn.split(sep)

#         if len(parts) != 2:
#             break
#         parts[0] += sep
        
#         instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#         if i != 0 and conv.roles[0] == 'USER':
#             instruction_len -= 1

#         target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

#         if conv.sep2 == '</s>':
#             cur_len += turn_len + 1 
#         elif conv.sep2 == ' </s><s>':
#             cur_len += turn_len + 3
#         else:
#             raise NotImplementedError
        
#         if i != 0 and conv.roles[0] == 'USER':
#             cur_len -= 1

#     target[cur_len:] = IGNORE_TOKEN_ID

#     if cur_len < tokenizer.model_max_length:
#         if cur_len != total_len:
#             print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. #turn = {len(turns) - 1}. (ignored)")
#             target[:] = IGNORE_TOKEN_ID

#     return target

# def preprocess_multi_turn(
#     source,
#     tokenizer: transformers.PreTrainedTokenizer,
#     model_path: str,
# ) -> Dict:
#     conv = get_model_adapter(model_path).get_default_conv_template(model_path)
#     print(f"Conversation template: {conv}")
#     print(f"Separator style: {conv.sep_style}")
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     conv.messages = []
#     for j, sentence in enumerate(source['prompt']):
#         role = roles[sentence["from"]]
#         assert role == conv.roles[j % 2]
#         conv.append_message(role, sentence["value"])
#     prompt = conv.get_prompt()

#     conv.messages = []
#     for j, sentence in enumerate(source['prompt'] + source['chosen']):
#         role = roles[sentence["from"]]
#         assert role == conv.roles[j % 2]
#         conv.append_message(role, sentence["value"])
#     chosen = conv.get_prompt()

#     conv.messages = []
#     for j, sentence in enumerate(source['prompt'] + source['rejected']):
#         role = roles[sentence["from"]]
#         assert role == conv.roles[j % 2]
#         conv.append_message(role, sentence["value"])
#     rejected = conv.get_prompt()

#     prompt_tokens = tokenizer(prompt, return_tensors="pt")

#     chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
#     chosen_labels = chosen_tokens.input_ids[0].clone()
#     chosen_labels = mask_labels(chosen, chosen_labels, tokenizer, conv)
#     chosen_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
#     rejected_labels = rejected_tokens.input_ids[0].clone()
#     rejected_labels = mask_labels(rejected, rejected_labels, tokenizer, conv)
#     rejected_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     return dict(
#         chosen_input_ids=chosen_tokens['input_ids'][0].tolist(),
#         chosen_attention_mask=chosen_tokens['attention_mask'][0].tolist(),
#         chosen_labels=chosen_labels.tolist(),
#         rejected_input_ids=rejected_tokens['input_ids'][0].tolist(),
#         rejected_attention_mask=rejected_tokens['attention_mask'][0].tolist(),
#         rejected_labels=rejected_labels.tolist(),
#         prompt_input_ids=prompt_tokens['input_ids'][0].tolist(),
#         prompt_attention_mask=prompt_tokens['attention_mask'][0].tolist(),
#     )

# def train():
#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments)
#     )
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     # 动态设置 beta 和学习率
#     if training_args.num_train_epochs == 1:
#         model_args.beta = 0.1
#         training_args.learning_rate = 1e-6
#     else:
#         model_args.beta = 0.5
#         training_args.learning_rate = 5e-7

#     config = transformers.AutoConfig.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=training_args.cache_dir,
#         trust_remote_code=model_args.trust_remote_code,
#     )
#     config.use_cache = False

#     model = transformers.AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         config=config,
#         cache_dir=training_args.cache_dir,
#         trust_remote_code=model_args.trust_remote_code,
#         device_map="auto",
#     )

#     # 添加这段代码来检查模型参数
#     print("Checking model parameters:")
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             print(f"Parameter {name}: shape={param.shape}, requires_grad=True")
#         else:
#             print(f"Parameter {name}: shape={param.shape}, requires_grad=False")

#     model_ref = transformers.AutoModelForCausalLM.from_pretrained(
#         model_args.ref_model_name_or_path,
#         config=config,
#         cache_dir=training_args.cache_dir,
#         trust_remote_code=model_args.trust_remote_code,
#         device_map="auto",
#     )

#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=training_args.cache_dir,
#         model_max_length=training_args.model_max_length,
#         padding_side=model_args.padding_side,
#         use_fast=False,
#         trust_remote_code=model_args.trust_remote_code,
#     )

#     if tokenizer.pad_token != tokenizer.unk_token:
#         tokenizer.pad_token = tokenizer.unk_token

#     dataset = load_dataset("json", data_files=data_args.data_path)
#     preprocess = partial(preprocess_multi_turn, tokenizer=tokenizer, model_path=model_args.model_name_or_path)
#     train_dataset = dataset["train"].map(preprocess)

#     trainer = DPOMultiTrainer(
#         model,
#         model_ref,
#         args=training_args,
#         beta=model_args.beta,
#         train_dataset=train_dataset,
#         tokenizer=tokenizer,
#         max_length=training_args.model_max_length,
#         max_target_length=training_args.max_target_length,
#         max_prompt_length=training_args.max_prompt_length,
#     )

#     if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
#         trainer.train(resume_from_checkpoint=True)
#     else:
#         trainer.train()

#     model.config.use_cache = True
#     trainer.save_state()
#     trainer.save_model()

# if __name__ == "__main__":
#     train()

# from dataclasses import dataclass, field
# import json
# import math
# import pathlib
# from typing import Dict, Optional, Sequence
# from functools import partial

# import numpy as np
# import torch
# from torch.utils.data import Dataset

# # from fastchat.train.llama2_flash_attn_monkey_patch import (
# #     replace_llama_attn_with_flash_attn,
# # )

# # replace_llama_attn_with_flash_attn()

# import transformers
# from transformers import Trainer
# from transformers.trainer_pt_utils import LabelSmoother
# # from trl import DPOTrainer
# from fastchat.train.dpo_trainer import DPOMultiTrainer
# from datasets import load_dataset

# from fastchat.conversation import SeparatorStyle
# from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

# IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# @dataclass
# class ModelArguments:
#     model_name_or_path: Optional[str] = field(default="/home/bhui/ML/ruimeng/ETO-main/output/weak_sft_llama2")
#     ref_model_name_or_path: Optional[str] = field(default="/home/bhui/ML/ruimeng/ETO-main/output/weak_sft_llama2")
#     trust_remote_code: bool = field(
#         default=False,
#         metadata={
#             "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
#         },
#     )
#     padding_side: str = field(
#         default="right", metadata={"help": "The padding side in tokenizer"}
#     )
#     beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})


# @dataclass
# class DataArguments:
#     data_path: str = field(
#         default="/home/bhui/ML/ruimeng/ETO-main/data_pm/webshop_weak_model_pm(weak+ground_truth).json", metadata={"help": "Path to the training data."}
#     )
#     eval_data_path: str = field(
#         default="/home/bhui/ML/ruimeng/ETO-main/data_pm/webshop_weak_model_pm(weak+ground_truth).json", metadata={"help": "Path to the evaluation data."}
#     )
#     lazy_preprocess: bool = False


# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     model_max_length: int = field(
#         default=512,
#         metadata={
#             "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
#         },
#     )
#     max_prompt_length: int = field(
#         default=512,
#         metadata={
#             "help": "Maximum target length."
#         },
#     )
#     max_target_length: int = field(
#         default=2048,
#         metadata={
#             "help": "Maximum target length."
#         },
#     )


# local_rank = None


# def rank0_print(*args):
#     if local_rank == 0:
#         print(*args)


# def trainer_save_model_safe(trainer: transformers.Trainer):
#     from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
#     from torch.distributed.fsdp import StateDictType, FullStateDictConfig

#     save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
#     with FSDP.state_dict_type(
#         trainer.model, StateDictType.FULL_STATE_DICT, save_policy
#     ):
#         trainer.save_model()


# def mask_labels(conversation, target, tokenizer, conv):
#     if conv.sep_style == SeparatorStyle.ADD_COLON_TWO:
#         sep = conv.sep + conv.roles[1] + ": "
#     elif conv.sep_style == SeparatorStyle.LLAMA2:
#         sep = conv.sep + conv.roles[1] + " "
#     else:
#         raise NotImplementedError
    
#     total_len = int(target.ne(tokenizer.pad_token_id).sum())

#     turns = conversation.split(conv.sep2)
#     cur_len = 1
#     target[:cur_len] = IGNORE_TOKEN_ID
#     for i, turn in enumerate(turns):
#         if turn == "":
#             break

#         # remove <s>
#         turn_len = len(tokenizer(turn).input_ids) - 1

#         parts = turn.split(sep)

#         if len(parts) != 2:
#             break
#         parts[0] += sep
        
#         # remove <s> and the "_" in the end
#         instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#         # magic number for vicuna, since different subtoken for "USER"
#         if i != 0 and conv.roles[0] == 'USER':
#             # The legacy and non-legacy modes handle special tokens differently
#             instruction_len -= 1

#         # Ignore the user instructions
#         target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

#         # add the length of turn sep
#         if conv.sep2 == '</s>':
#             cur_len += turn_len + 1 
#         elif conv.sep2 == ' </s><s>':
#             cur_len += turn_len + 3
#         else:
#             raise NotImplementedError
        
#         # magic number for vicuna, since different subtoken for "USER"
#         if i != 0 and conv.roles[0] == 'USER':
#             # The legacy and non-legacy modes handle special tokens differently
#             cur_len -= 1

#     target[cur_len:] = IGNORE_TOKEN_ID

#     if False:  # Inspect and check the correctness of masking
#         z = target.clone()
#         z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
#         rank0_print(conversation)
#         rank0_print(tokenizer.decode(z))
#         exit()

#     if cur_len < tokenizer.model_max_length:
#         if cur_len != total_len:
#             z = target.clone()
#             z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
#             rank0_print(conversation)
#             print("#" * 50)
#             rank0_print(tokenizer.decode(z))
#             target[:] = IGNORE_TOKEN_ID
#             rank0_print(
#                 f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                 f" #turn = {len(turns) - 1}. (ignored)"
#             )

#     return target


# def preprocess_multi_turn(
#     source,
#     tokenizer: transformers.PreTrainedTokenizer,
#     model_path: str,
# ) -> Dict:
#     conv = get_model_adapter(model_path).get_default_conv_template(model_path)
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conv.messages = []
#     for j, sentence in enumerate(source['prompt']):
#         role = roles[sentence["from"]]
#         assert role == conv.roles[j % 2]
#         conv.append_message(role, sentence["value"])
#     prompt = conv.get_prompt()

#     conv.messages = []
#     for j, sentence in enumerate(source['prompt'] + source['chosen']):
#         role = roles[sentence["from"]]
#         assert role == conv.roles[j % 2]
#         conv.append_message(role, sentence["value"])
#     chosen = conv.get_prompt()

#     conv.messages = []
#     for j, sentence in enumerate(source['prompt'] + source['rejected']):
#         role = roles[sentence["from"]]
#         assert role == conv.roles[j % 2]
#         conv.append_message(role, sentence["value"])
#     rejected = conv.get_prompt()

#     # Tokenize conversations
#     prompt_tokens = tokenizer(prompt, return_tensors="pt")

#     chosen_tokens = tokenizer(chosen, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
#     # chosen_tokens = tokenizer(chosen, return_tensors="pt")
#     chosen_labels = chosen_tokens.input_ids[0].clone()
#     chosen_labels = mask_labels(chosen, chosen_labels, tokenizer, conv)
#     chosen_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     rejected_tokens = tokenizer(rejected, return_tensors="pt", max_length=tokenizer.model_max_length, truncation=True)
#     rejected_labels = rejected_tokens.input_ids[0].clone()
#     rejected_labels = mask_labels(rejected, rejected_labels, tokenizer, conv)
#     rejected_labels[:len(prompt_tokens['input_ids'][0])] = IGNORE_TOKEN_ID

#     if False:  # Inspect and check the correctness of masking
#         z = chosen_labels.clone()
#         z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
#         rank0_print(chosen)
#         rank0_print(tokenizer.decode(z))
#         z = rejected_labels.clone()
#         z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
#         rank0_print(rejected)
#         rank0_print(tokenizer.decode(z))
#         exit()

#     return dict(
#         chosen_input_ids=chosen_tokens['input_ids'][0].tolist(),
#         chosen_attention_mask=chosen_tokens['attention_mask'][0].tolist(),
#         chosen_labels=chosen_labels.tolist(),
#         rejected_input_ids=rejected_tokens['input_ids'][0].tolist(),
#         rejected_attention_mask=rejected_tokens['attention_mask'][0].tolist(),
#         rejected_labels=rejected_labels.tolist(),
#         prompt_input_ids=prompt_tokens['input_ids'][0].tolist(),
#         prompt_attention_mask=prompt_tokens['attention_mask'][0].tolist(),
#     )


# def train():
#     global local_rank

#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments)
#     )
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#     local_rank = training_args.local_rank

#     # Set RoPE scaling factor
#     config = transformers.AutoConfig.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=training_args.cache_dir,
#         trust_remote_code=model_args.trust_remote_code,
#     )
#     orig_ctx_len = getattr(config, "max_position_embeddings", None)
#     if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
#         scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
#         config.rope_scaling = {"type": "linear", "factor": scaling_factor}
#     config.use_cache = False

#     # Load model and tokenizer
#     model = transformers.AutoModelForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         config=config,
#         cache_dir=training_args.cache_dir,
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation="flash_attention_2",
#     )

#     model_ref = transformers.AutoModelForCausalLM.from_pretrained(
#         model_args.ref_model_name_or_path,
#         config=config,
#         cache_dir=training_args.cache_dir,
#         trust_remote_code=model_args.trust_remote_code,
#         attn_implementation="flash_attention_2",
#     )

#     tokenizer = transformers.AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=training_args.cache_dir,
#         model_max_length=training_args.model_max_length,
#         padding_side=model_args.padding_side,
#         use_fast=False,
#         trust_remote_code=model_args.trust_remote_code,
#     )

#     if tokenizer.pad_token != tokenizer.unk_token:
#         tokenizer.pad_token = tokenizer.unk_token

#     # Load data
#     dataset = load_dataset("json", data_files=data_args.data_path)
#     preprocess = partial(preprocess_multi_turn, tokenizer=tokenizer, model_path=model_args.model_name_or_path)
#     train_dataset = dataset["train"].map(preprocess)

#     # Start trainner
#     trainer = DPOMultiTrainer(
#         model,
#         model_ref,
#         args=training_args,
#         beta=model_args.beta,
#         train_dataset=train_dataset,
#         tokenizer=tokenizer,
#         max_length=training_args.model_max_length,
#         max_target_length=training_args.max_target_length,
#         max_prompt_length=training_args.max_prompt_length,
#         generate_during_eval=True,
#     )

#     # trainer.ref_model = trainer.accelerator.prepare(trainer.ref_model)

#     if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
#         trainer.train(resume_from_checkpoint=True)
#     else:
#         trainer.train()

#     # Save model
#     model.config.use_cache = True
#     trainer.save_state()
#     if trainer.is_deepspeed_enabled:
#         trainer.save_model()
#     else:
#         trainer_save_model_safe(trainer)


# if __name__ == "__main__":
#     train()