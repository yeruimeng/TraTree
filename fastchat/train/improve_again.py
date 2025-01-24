import os
import argparse
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from typing import Dict
from tqdm import tqdm  # 添加tqdm用于进度条
from collections import deque
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(42)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="output/lora_sft_sci_strong_llama2/merged_model")
    parser.add_argument("--data_path", type=str, default="all_update_advantage_trajectories.json")  # 修改为包含优势值的数据文件
    parser.add_argument("--output_dir", type=str, default="mcts/lora_wts_sci_mcts_optimal_again")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lazy_preprocess", action="store_true", help="Use lazy preprocessing")
    return parser.parse_args()


def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    advantages_list = []

    # 添加进度条
    for source in sources:
        conversation = []

        # 处理 'prompt' 部分
        if 'prompt' in source:
            for turn in source['prompt']:
                if turn['from'] == 'human':
                    conversation.append({"role": "user", "content": turn['value']})
                elif turn['from'] in ['gpt', 'assistant']:
                    # 为 prompt 部分的 gpt 回复添加默认的 advantage 值
                    conversation.append({"role": "assistant", "content": turn['value'], "advantage": 0.0})
                else:
                    # 处理其他可能的 'from' 值，确保不遗漏
                    conversation.append({"role": "assistant", "content": turn['value'], "advantage": 0.0})

        # 处理 'conversations' 部分
        if 'conversations' in source:
            for turn in source['conversations']:
                if turn['from'] == 'human':
                    conversation.append({"role": "user", "content": turn['value']})
                elif turn['from'] in ['gpt', 'assistant']:
                    # 获取优势值，如果没有则默认为0.0
                    advantage = turn.get('advantage', 0.0)
                    conversation.append({"role": "assistant", "content": turn['value'], "advantage": advantage})
                else:
                    # 处理其他可能的 'from' 值，确保不遗漏
                    advantage = turn.get('advantage', 0.0)
                    conversation.append({"role": "assistant", "content": turn['value'], "advantage": advantage})

        # 将对话转换为模型输入格式
        formatted_conversation = format_conversation(conversation, tokenizer)
        input_ids_list.extend(formatted_conversation["input_ids"])
        attention_mask_list.extend(formatted_conversation["attention_mask"])
        labels_list.extend(formatted_conversation["labels"])
        advantages_list.extend(formatted_conversation["advantages"])

    # 数据验证
    for i, advantage in enumerate(advantages_list):
        if not isinstance(advantage, float):
            print(f"Advantage at index {i} is not a float: {advantage}")
            advantages_list[i] = 0.0

    return dict(
        input_ids=input_ids_list,
        attention_mask=attention_mask_list,
        labels=labels_list,
        advantages=advantages_list,
    )


def format_conversation(conversation, tokenizer):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    advantages_list = []

    # 初始系统提示
    formatted = f"{B_INST} {B_SYS}You are a helpful assistant.{E_SYS}\n\n"

    # 初始化上一条消息的内容
    last_user_message = ""

    for i, message in enumerate(conversation):
        if message["role"] == "user":
            user_content = f"{message['content']} {E_INST} "
            formatted += user_content
            last_user_message = user_content
        elif message["role"] == "assistant":
            assistant_content = f"{message['content']} {E_INST} "
            formatted += assistant_content

            encoding = tokenizer(
                formatted,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=tokenizer.model_max_length,
            )

            input_ids = encoding.input_ids[0]
            attention_mask = encoding.attention_mask[0]

            labels = input_ids.clone()
            user_input_ids = tokenizer(last_user_message, return_tensors="pt").input_ids[0][:-1]
            assistant_input_ids = tokenizer(assistant_content, return_tensors="pt").input_ids[0][:-1]
            start_idx = len(input_ids) - len(assistant_input_ids) - 1
            labels[:start_idx] = IGNORE_TOKEN_ID
            labels[labels == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

            advantage = message.get('advantage', 0.0)
            # print(f"Raw advantage: {advantage}")
            try:
                advantage = float(advantage)
            except (TypeError, ValueError):
                print(f"Error converting advantage to float: {e}")
                advantage = 0.0

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            advantages_list.append(advantage)

            last_user_message = ""

    assert len(input_ids_list) == len(advantages_list), "Mismatch between input_ids and advantages"

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
        "advantages": advantages_list,
    }


class AdvantageDataset(Dataset):
    def __init__(self, data_dict: Dict):
        super(AdvantageDataset, self).__init__()
        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.labels = data_dict["labels"]
        self.advantages = data_dict["advantages"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        print(f"Dataset item {i}, advantage: {self.advantages[i]}")
        return dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            labels=self.labels[i],
            advantages=torch.tensor(self.advantages[i], dtype=torch.float),
        )


class LazyAdvantageDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazyAdvantageDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
            
        data_dict = preprocess([self.raw_data[i]], self.tokenizer)
        ret = dict(
            input_ids=data_dict["input_ids"][0],
            attention_mask=data_dict["attention_mask"][0],
            labels=data_dict["labels"][0],
            advantages=torch.tensor(data_dict["advantages"][0], dtype=torch.float),
        )
        self.cached_data_dict[i] = ret
        return ret

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    dataset_cls = LazyAdvantageDataset if data_args.lazy_preprocess else AdvantageDataset
    print("Loading data...")

    with open(data_args.data_path, "r", encoding='utf-8') as f:
        train_json = json.load(f)
    
    print(f"\nFound {len(train_json)} conversations in dataset")
    
    if not data_args.lazy_preprocess:
        print("\nPre-processing all conversations at once...")
        data_dict = preprocess(train_json, tokenizer)
        train_dataset = dataset_cls(data_dict)
        print("\nPreprocessing completed!")
    else:
        print("\nUsing lazy loading mode - conversations will be processed as needed")
        train_dataset = dataset_cls(train_json, tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=None)

class BacktrackingAdvantageEstimator:
    def __init__(self, gamma=0.99, lambda_=0.95, memory_size=1000):
        self.gamma = gamma          # GAE折扣因子
        self.lambda_ = lambda_      # GAE平滑参数
        self.memory_size = memory_size
        self.advantage_history = deque(maxlen=memory_size)
        self.state_history = {}     # 记录状态转换历史
        self.total_steps = 0
        self.ema_advantage = None   # 添加EMA
        self.ema_alpha = 0.1        # EMA平滑因子
        
    def update_state_history(self, current_state, previous_state, advantage):
        """记录状态转换和对应的优势值"""
        if previous_state not in self.state_history:
            self.state_history[previous_state] = []
        self.state_history[previous_state].append((current_state, advantage))
    
    def get_backtracking_factor(self, state, advantage):
        """基于历史状态转换计算回溯因子"""
        if state not in self.state_history:
            return 1.0
        
        transitions = self.state_history[state]
        if not transitions:
            return 1.0
            
        # 计算历史平均优势值
        avg_historical_advantage = np.mean([adv for _, adv in transitions])
        
        # 动态回溯因子：根据差异程度调整权重
        diff = avg_historical_advantage - advantage
        if diff > 0:
            return 1.0 + np.tanh(diff)  # 使用tanh限制增幅
        return 1.0
    
    def compute_gae(self, rewards, values, masks):
        """计算广义优势估计"""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * masks[t] - values[t]
            last_gae = delta + self.gamma * self.lambda_ * masks[t] * last_gae
            advantages[t] = last_gae
            
        return advantages
    
    def process_advantage(self, advantage, state=None):
        """改进的优势值处理"""
        self.total_steps += 1
        self.advantage_history.append(advantage)
        
        # 更新EMA
        if self.ema_advantage is None:
            self.ema_advantage = advantage
        else:
            self.ema_advantage = self.ema_alpha * advantage + (1 - self.ema_alpha) * self.ema_advantage
        
        if len(self.advantage_history) > 0:
            advantage_array = np.array(self.advantage_history)
            
            # 使用更稳健的统计方法
            median = np.median(advantage_array)
            mad = np.median(np.abs(advantage_array - median))
            
            if mad > 0:
                # 结合EMA和MAD的缩放
                scaled_advantage = (advantage - self.ema_advantage) / (mad * 1.4826)
            else:
                # 如果MAD接近0，使用min-max缩放
                max_abs = max(abs(min(advantage_array)), abs(max(advantage_array)))
                scaled_advantage = advantage / (max_abs + 1e-8)
            
            # 使用softer的非线性变换
            nonlinear_advantage = 2.0 / (1.0 + np.exp(-scaled_advantage)) - 1.0
            
            # 转换为tensor
            result = torch.tensor(nonlinear_advantage, dtype=torch.float)
        else:
            result = torch.tensor(advantage, dtype=torch.float)
        
        # 应用回溯因子
        if state is not None:
            backtrack_factor = self.get_backtracking_factor(state, advantage)
            result = result * backtrack_factor
        
        return result
    
    def get_dynamic_alpha(self):
        """改进的动态alpha计算"""
        base_alpha = 0.5
        warmup_steps = 1000
        decay_rate = 0.98
        
        # 添加周期性调整
        cycle_length = 2000
        cycle_factor = 0.1 * (1 + np.cos(2 * np.pi * self.total_steps / cycle_length))
        
        # 计算基础alpha
        alpha = base_alpha * (1 - np.exp(-self.total_steps / warmup_steps))
        alpha *= decay_rate ** (self.total_steps / warmup_steps)
        
        # 添加周期性波动
        alpha = alpha * (1 + cycle_factor)
        
        return min(max(0.1, alpha), base_alpha)

class ImprovedAdvantageTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.advantage_estimator = BacktrackingAdvantageEstimator()
        self.step_count = 0
        
    def compute_loss(self, model, inputs, return_outputs=False):
        self.step_count += 1
        advantages = inputs.pop("advantages").to(model.device)
        states = inputs.pop("states", None)
        
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 基础损失计算
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_TOKEN_ID)
        token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1))
        sequence_loss = token_loss.view(shift_labels.size()).mean(dim=1)
        
        # 1. 改进的稳健缩放
        def improved_robust_scale(x):
            # 确保输入是有效的
            if torch.all(x == 0):
                return torch.zeros_like(x)
                
            median = x.median()
            abs_dev = (x - median).abs()
            mad = abs_dev.median()
            
            # 如果MAD太小，使用最大最小值缩放
            if mad < 1e-6:
                max_val = x.max()
                min_val = x.min()
                if max_val == min_val:
                    return torch.zeros_like(x)
                scaled = (x - min_val) / (max_val - min_val + 1e-6) * 2 - 1
            else:
                scaled = (x - median) / (mad * 1.4826 + 1e-6)
            
            # 使用softer的tanh
            return torch.tanh(scaled * 0.5)  # 使用0.5作为缩放因子使变换更平滑
        
        scaled_advantages = improved_robust_scale(advantages)
        
        # 2. 改进的动态alpha
        warmup_steps = 1000
        base_alpha = 0.5
        curr_step = self.step_count
        if curr_step < warmup_steps:
            alpha = base_alpha * (curr_step / warmup_steps)
        else:
            # 使用更平滑的周期调整
            alpha = base_alpha * (0.9 + 0.1 * torch.cos(torch.tensor(curr_step / 1000 * 3.14159)))
        
        # 3. 改进的权重计算
        advantage_weights = torch.ones_like(scaled_advantages)
        positive_mask = scaled_advantages > 0
        negative_mask = scaled_advantages < 0
        
        # 对正负优势值分别处理
        advantage_weights[positive_mask] = torch.exp(-alpha * scaled_advantages[positive_mask])
        advantage_weights[negative_mask] = torch.exp(alpha * scaled_advantages[negative_mask])
        
        # 4. 改进的置信度权重
        advantage_magnitude = torch.abs(scaled_advantages)
        confidence_weight = torch.clamp(advantage_magnitude, 0, 1)
        
        # 5. 熵正则化
        probs = torch.softmax(shift_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean(dim=1)
        entropy_weight = 0.01 * torch.exp(-torch.tensor(curr_step / 5000.0))
        
        # 6. 改进的课程学习
        curriculum_factor = torch.clamp(torch.tensor(curr_step / 2000.0), 0, 1)
        
        # 结合所有组件的损失计算
        weighted_loss = sequence_loss * (1.0 + advantage_weights * confidence_weight)
        weighted_loss = curriculum_factor * weighted_loss + (1 - curriculum_factor) * sequence_loss
        
        # 7. KL散度约束
        kl_div = torch.tensor(0.0, device=model.device)
        if curr_step > warmup_steps:
            with torch.no_grad():
                old_probs = torch.softmax(shift_logits.detach(), dim=-1)
            kl_div = torch.nn.functional.kl_div(
                torch.log(probs + 1e-10), 
                old_probs,
                reduction='batchmean'
            )
        
        kl_weight = 0.01 if curr_step > warmup_steps else 0.0
        
        # 最终损失
        final_loss = (
            weighted_loss.mean() +
            entropy_weight * entropy.mean() +
            kl_weight * kl_div
        )
        
        # 记录更详细的训练统计
        if self.state.global_step % 100 == 0:
            with torch.no_grad():
                self.log({
                    "train/alpha": alpha if isinstance(alpha, float) else alpha.item(),
                    "train/scaled_advantages_mean": scaled_advantages.mean().item(),
                    "train/scaled_advantages_std": scaled_advantages.std().item(),
                    "train/advantage_weights_mean": advantage_weights.mean().item(),
                    "train/confidence_weight_mean": confidence_weight.mean().item(),
                    "train/entropy": entropy.mean().item(),
                    "train/kl_div": kl_div.item(),
                    "train/original_loss": sequence_loss.mean().item(),
                    "train/weighted_loss": weighted_loss.mean().item(),
                    "train/final_loss": final_loss.item(),
                    "train/curriculum_factor": curriculum_factor.item(),
                    # 添加更多调试信息
                    "train/positive_advantages": (scaled_advantages > 0).float().mean().item(),
                    "train/negative_advantages": (scaled_advantages < 0).float().mean().item(),
                    "train/max_advantage": scaled_advantages.max().item(),
                    "train/min_advantage": scaled_advantages.min().item()
                })
        
        return (final_loss, outputs) if return_outputs else final_loss

class ModifiedDataCollatorWithAdvantages:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features):
        print(f"Features: {features}")
        # 基本的batch处理保持不变
        batch = self.tokenizer.pad(
            {
                "input_ids": [f["input_ids"] for f in features],
                "attention_mask": [f["attention_mask"] for f in features],
                "labels": [f["labels"] for f in features]
            },
            padding='longest',
            return_tensors="pt"
        )
        
        # 修改这里的优势值处理
        advantages = []
        for feature in features:
            if 'advantages' in feature:
                adv = feature['advantages']
                print(f"Feature advantage: {adv}")
                advantages.append(torch.tensor(adv, dtype=torch.float))
            else:
                print("Advantage not found in feature!")
                advantages.append(torch.tensor(0.0, dtype=torch.float))
            adv = feature.get("advantages", torch.tensor(0.0))
            print(f"Feature advantage: {adv}")
            if isinstance(adv, (int, float)):
                adv = torch.tensor(adv, dtype=torch.float)
            advantages.append(adv)
        
        batch["advantages"] = torch.stack(advantages)
        
        # 添加额外的调试信息
        if len(advantages) > 0:
            print(f"Batch advantages: {batch['advantages']}")
            
        return batch

class DataCollatorWithAdvantages:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [feature['input_ids'] for feature in features]
        attention_mask = [feature['attention_mask'] for feature in features]
        labels = [feature['labels'] for feature in features]
        
        advantages = []
        for feature in features:
            if 'advantages' in feature:
                advantages.append(feature['advantages'])
            else:
                advantages.append(torch.tensor(0.0, dtype=torch.float))
        
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels},
            padding='longest',
            return_tensors="pt",
        )

        batch['advantages'] = torch.stack(advantages)
        return batch


class AdvantageTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        advantages = inputs.pop("advantages").to(model.device)
        print(f"Advantages in compute_loss: {advantages}")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_TOKEN_ID)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.size())
        loss = loss.mean(dim=1)

        # 确保优势值不是 nan
        advantages = torch.nan_to_num(advantages, 0.0)
        # 使用更安全的标准化方式
        std = advantages.std()
        if std > 0:
            advantages = (advantages - advantages.mean()) / std
        # 不进行截断
        # advantages = torch.clamp(advantages, -1.0, 1.0)

        # 使用新的损失调整公式
        alpha = 0.5  # 可以根据需要调整
        adjusted_loss = loss * (1.0 - alpha * advantages)
        adjusted_loss = torch.clamp(adjusted_loss, min=0.0)
        adjusted_loss = adjusted_loss.mean()

        # 如果损失是 nan,使用原始损失
        if torch.isnan(adjusted_loss) or adjusted_loss.item() == 0.0:
            adjusted_loss = loss.mean()

        return (adjusted_loss, outputs) if return_outputs else adjusted_loss


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        model_max_length=args.model_max_length,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, config)

    print("Prepare the dataset...")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=args)
    train_dataset = data_module["train_dataset"]

    # 打印前几个样本进行验证
    print("Sample data from the dataset:")
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Sample {i}:")
        print(f"  input_ids: {sample['input_ids']}")
        print(f"  attention_mask: {sample['attention_mask']}")
        print(f"  labels: {sample['labels']}")
        print(f"  advantages: {sample['advantages']}")
        print("---")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=True,
        save_strategy="steps",
        save_steps=1000,  # 修改保存步数为1000以匹配你的checkpoint
        logging_steps=10,
        evaluation_strategy="no",
        remove_unused_columns=False,  # 添加这一行
        )


    data_collator = DataCollatorWithAdvantages(tokenizer=tokenizer)

    trainer = ImprovedAdvantageTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=ModifiedDataCollatorWithAdvantages(tokenizer)  # 确保这里传入的是自定义的数据整理器
)


    # 检查是否有现有的checkpoint
    checkpoint_dir = os.path.join(args.output_dir, "checkpoint-555")

    if os.path.exists(checkpoint_dir):
        print(f"Resuming training from checkpoint: {checkpoint_dir}")
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        print("No checkpoint found, starting training from scratch.")
        trainer.train()

    # 保存 LoRA 权重
    model.save_pretrained(os.path.join(args.output_dir, "lora_weights"))

    # 合并 LoRA 权重到原始模型并保存
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(os.path.join(args.output_dir, "merged_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "merged_model"))


if __name__ == "__main__":
    args = parse_args()
    train(args)