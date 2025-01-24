import os
import argparse
import json
import math
from typing import Dict, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
)
from transformers.trainer_pt_utils import LabelSmoother
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from torch import nn
from transformers import default_data_collator
from tqdm import tqdm

# 定义常量
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--ref_model_name", type=str, required=True, help="Path to the reference model.")
    parser.add_argument("--data_path", type=str, default="data/webshop_sft.json")
    parser.add_argument("--output_dir", type=str, default="output/strong_sft_llama2")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lazy_preprocess", action="store_true")
    parser.add_argument("--beta", type=float, default=0.2, help="Initial beta value for loss calculation.")
    parser.add_argument("--scaling_factor", type=float, default=8000.0, help="Initial scaling factor for reward_diff.")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha.")
    return parser.parse_args()

def format_conversation(conversation):
    """
    手动格式化对话，应用自定义的对话模板。
    """
    formatted = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
    for i, message in enumerate(conversation):
        if message["role"] == "user":
            formatted += f"{message['content']} [/INST] "
        elif message["role"] == "assistant":
            formatted += f"{message['content']} </s><s>[INST] "
    formatted = formatted.rstrip(" </s><s>[INST] ")
    return formatted

def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    conversations = []
    for source in sources:
        conversation = []
        for turn in source:
            if turn['from'] == 'human':
                conversation.append({"role": "user", "content": turn['value']})
            elif turn['from'] == 'gpt':
                conversation.append({"role": "assistant", "content": turn['value']})
        conversations.append(conversation)

    formatted_conversations = [format_conversation(conv) for conv in conversations]

    encodings = tokenizer(
        formatted_conversations,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        print("Formatting inputs...")
        sources = []
        self.reward_diff = []
        for example in raw_data:
            if "prompt" not in example or "reward_diff" not in example:
                raise ValueError(f"Data sample missing 'prompt' or 'reward_diff': {example}")
            sources.append(example["prompt"])
            self.reward_diff.append(example["reward_diff"])
        
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            attention_mask=self.attention_mask[i],
            labels=self.labels[i],
            reward_diff=torch.tensor(self.reward_diff[i], dtype=torch.float)
        )


class LazySupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        if "prompt" not in self.raw_data[i] or "reward_diff" not in self.raw_data[i]:
            raise ValueError(f"Data sample missing 'prompt' or 'reward_diff': {self.raw_data[i]}")

        ret = preprocess([self.raw_data[i]["prompt"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            attention_mask=ret["attention_mask"][0],
            labels=ret["labels"][0],
            reward_diff=torch.tensor(self.raw_data[i]["reward_diff"], dtype=torch.float)
        )
        self.cached_data_dict[i] = ret
        return ret

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=None)

class DynamicMCTSLoss(nn.Module):
    def __init__(self, model, ref_model, tokenizer, initial_beta=0.2, initial_scaling_factor=5000.0, total_steps=10000):
        super(DynamicMCTSLoss, self).__init__()
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.initial_beta = initial_beta
        self.initial_scaling_factor = initial_scaling_factor
        self.total_steps = total_steps
        self.current_step = 0
        
        # 创建日志目录
        self.log_dir = "training_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建不同类型的日志文件
        self.training_log = open(os.path.join(self.log_dir, "training_metrics.log"), "w")
        self.loss_history_log = open(os.path.join(self.log_dir, "loss_history.log"), "w")
        self.parameter_log = open(os.path.join(self.log_dir, "parameter_trajectory.log"), "w")
        
        # 初始化跟踪指标
        self.loss_history = []
        self.beta_history = []
        self.scaling_factor_history = []
        self.reward_history = []
        
    def log_metrics(self, losses, reward_diff):
        """记录训练指标"""
        # 记录基本训练指标
        log_entry = (f"Step {self.current_step}:\n"
                    f"Beta: {losses['current_beta']:.4f}\n"
                    f"Scaling Factor: {losses['current_scaling_factor']:.2f}\n"
                    f"CE Loss: {losses['ce_loss']:.4f}\n"
                    f"KL Div: {losses['kl_div']:.4f}\n"
                    f"Weighted CE: {losses['weighted_ce']:.4f}\n"
                    f"Reward Diff: {losses['avg_reward_diff']:.4f}\n"
                    f"Normalized Reward: {losses['avg_normalized_reward']:.4f}\n"
                    f"Final Loss: {losses['final_loss'].item():.4f}\n"
                    f"-" * 50 + "\n")
        self.training_log.write(log_entry)
        self.training_log.flush()
        
        # 记录损失历史
        self.loss_history.append({
            'step': self.current_step,
            'ce_loss': losses['ce_loss'],
            'kl_div': losses['kl_div'],
            'final_loss': losses['final_loss'].item()
        })
        
        # 记录参数轨迹
        param_entry = (f"{self.current_step}\t"
                      f"{losses['current_beta']:.4f}\t"
                      f"{losses['current_scaling_factor']:.2f}\n")
        self.parameter_log.write(param_entry)
        self.parameter_log.flush()
        
        # 更新历史记录
        self.reward_history.append(reward_diff.mean().item())

    def compute_dynamic_parameters(self):
        """计算动态beta和scaling_factor值"""
        # beta从0.05增加到0.1，增长幅度为0.15
        beta = min(0.1, self.initial_beta + 0.15 * (self.current_step / self.total_steps))
        
        # scaling_factor从initial_scaling_factor降低到8000  webshop 从20降到5
        scaling_factor = max(5.0, self.initial_scaling_factor / (1 + 0.0005 * self.current_step))
        
        self.beta_history.append(beta)
        self.scaling_factor_history.append(scaling_factor)
        
        return beta, scaling_factor

    def forward(self, input_ids, attention_mask, labels, reward_diff):
        try:
            # 计算CE loss
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss_ce = outputs.loss

            # 计算KL divergence
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                ref_logits = ref_outputs.logits

            logits = outputs.logits
            p = F.log_softmax(logits, dim=-1)
            q = F.softmax(ref_logits, dim=-1)
            kl_div = F.kl_div(p, q, reduction='batchmean')

            # 动态调整参数
            beta, scaling_factor = self.compute_dynamic_parameters()

            # 处理reward_diff
            min_reward = torch.min(reward_diff)
            shifted_reward = reward_diff - min_reward
            R_diff_normalized = torch.log1p(shifted_reward / scaling_factor)

            weighted_ce = R_diff_normalized * loss_ce
            final_loss = weighted_ce.mean() + beta * kl_div

            self.current_step += 1

            loss_dict = {
                'final_loss': final_loss,
                'ce_loss': loss_ce.mean().item(),
                'kl_div': kl_div.item(),
                'weighted_ce': weighted_ce.mean().item(),
                'avg_reward_diff': reward_diff.mean().item(),
                'avg_normalized_reward': R_diff_normalized.mean().item(),
                'current_beta': beta,
                'current_scaling_factor': scaling_factor
            }

            # 记录训练指标
            self.log_metrics(loss_dict, reward_diff)

            return loss_dict
            
        except Exception as e:
            print(f"Error in DynamicMCTSLoss.forward: {str(e)}")
            raise e

    def plot_training_curves(self):
        """在训练结束时绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建一个2x2的子图布局
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 绘制损失曲线
            steps = [x['step'] for x in self.loss_history]
            ce_losses = [x['ce_loss'] for x in self.loss_history]
            kl_losses = [x['kl_div'] for x in self.loss_history]
            final_losses = [x['final_loss'] for x in self.loss_history]
            
            axes[0, 0].plot(steps, ce_losses, label='CE Loss')
            axes[0, 0].plot(steps, final_losses, label='Final Loss')
            axes[0, 0].set_title('Loss History')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss Value')
            axes[0, 0].legend()
            
            # 绘制KL散度
            axes[0, 1].plot(steps, kl_losses)
            axes[0, 1].set_title('KL Divergence History')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('KL Divergence')
            
            # 绘制beta变化
            axes[1, 0].plot(self.beta_history)
            axes[1, 0].set_title('Beta Trajectory')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Beta Value')
            
            # 绘制scaling factor变化
            axes[1, 1].plot(self.scaling_factor_history)
            axes[1, 1].set_title('Scaling Factor Trajectory')
            axes[1, 1].set_xlabel('Steps')
            axes[1, 1].set_ylabel('Scaling Factor Value')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error plotting training curves: {str(e)}")

    def __del__(self):
        """确保文件正确关闭"""
        self.training_log.close()
        self.loss_history_log.close()
        self.parameter_log.close()

def train(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        model_max_length=args.model_max_length,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型和配置
    policy_config = AutoConfig.from_pretrained(args.model_name_or_path)
    orig_ctx_len = getattr(policy_config, "max_position_embeddings", None)
    if orig_ctx_len and args.model_max_length > orig_ctx_len:
        scaling_fac = float(math.ceil(args.model_max_length / orig_ctx_len))
        policy_config.rope_scaling = {"type": "linear", "factor": scaling_fac}
    policy_config.use_cache = not args.gradient_checkpointing

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=policy_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()

    # 配置 LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    policy_model = get_peft_model(policy_model, lora_config)

    # 加载参考模型
    ref_config = AutoConfig.from_pretrained(args.ref_model_name)
    if orig_ctx_len and args.model_max_length > orig_ctx_len:
        scaling_fac = float(math.ceil(args.model_max_length / orig_ctx_len))
        ref_config.rope_scaling = {"type": "linear", "factor": scaling_fac}
    ref_config.use_cache = True

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_name,
        config=ref_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    for param in ref_model.parameters():
        param.requires_grad = False

    # 加载数据
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=args)
    train_dataset = data_module["train_dataset"]

    # 计算 warmup steps
    num_update_steps_per_epoch = len(train_dataset) // args.gradient_accumulation_steps
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * args.warmup_ratio)

    # 定义动态损失函数
    criterion = DynamicMCTSLoss(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        initial_beta=args.beta,
        initial_scaling_factor=args.scaling_factor,
        total_steps=max_train_steps
    )

    # 定义优化器和调度器
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )

    # 打印训练配置信息
    print("\nTraining Configuration:")
    print(f"Number of batches: {len(train_dataset)}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Number of training steps per epoch: {num_update_steps_per_epoch}")
    print(f"Total training steps: {max_train_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.model_max_length}")
    print(f"Initial Beta: {args.beta}")
    print(f"Initial Scaling Factor: {args.scaling_factor}\n")

    policy_model.train()
    global_step = 0

    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.per_device_train_batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=default_data_collator
    )

    for epoch in range(args.num_train_epochs):
        print(f"Starting epoch {epoch + 1}/{args.num_train_epochs}")
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", total=len(dataloader))

        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 将数据移动到设备
                input_ids = batch['input_ids'].to(policy_model.device)
                attention_mask = batch['attention_mask'].to(policy_model.device)
                labels = batch['labels'].to(policy_model.device)
                reward_diff = batch['reward_diff'].to(policy_model.device)

                # 获取损失
                losses = criterion(input_ids, attention_mask, labels, reward_diff)
                
                # 确保所有必需的键都存在
                required_keys = ['final_loss', 'ce_loss', 'kl_div', 'weighted_ce', 
                               'avg_reward_diff', 'avg_normalized_reward', 
                               'current_beta', 'current_scaling_factor']
                
                if not all(key in losses for key in required_keys):
                    missing_keys = [key for key in required_keys if key not in losses]
                    print(f"Warning: Missing keys in losses: {missing_keys}")
                    continue

                # 计算梯度
                final_loss = losses['final_loss'] / args.gradient_accumulation_steps
                final_loss.backward()

                # 更新累积损失
                epoch_loss += final_loss.item()

                # 在达到梯度累积步数后更新模型
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # 更新进度条
                    progress_bar.set_postfix({
                        'loss': f'{final_loss.item() * args.gradient_accumulation_steps:.4f}',
                        'ce': f'{losses["ce_loss"]:.4f}',
                        'kl': f'{losses["kl_div"]:.4f}',
                        'w_ce': f'{losses["weighted_ce"]:.4f}',
                        'r_diff': f'{losses["avg_reward_diff"]:.4f}',
                        'norm_r': f'{losses["avg_normalized_reward"]:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                        'beta': f'{losses["current_beta"]:.4f}',
                        'scale': f'{losses["current_scaling_factor"]:.2f}'
                    })

                    # 定期记录详细信息
                    if global_step % args.logging_steps == 0:
                        print(f"\nStep {global_step} loss details:")
                        for key in required_keys:
                            if key in losses:
                                print(f"  {key}: {losses[key]:.4f}")
                        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

                    # 保存检查点
                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-step-{global_step}')
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        
                        print(f"\nSaving checkpoint at step {global_step}...")
                        policy_model.save_pretrained(checkpoint_dir)
                        tokenizer.save_pretrained(checkpoint_dir)
                        print(f"Checkpoint saved at {checkpoint_dir}")

            except Exception as e:
                print(f"\nError in batch {batch_idx}: {str(e)}")
                print("Full error details:")
                import traceback
                print(traceback.format_exc())
                continue

        # 打印每个epoch的平均损失
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch + 1} finished with average loss: {avg_epoch_loss:.4f}")

    print("\nTraining completed. Starting model saving process...")

    # 保存最终的LoRA权重
    print("Saving final LoRA weights...")
    lora_save_path = os.path.join(args.output_dir, "final_lora")
    policy_model.save_pretrained(lora_save_path)
    print("Final LoRA weights saved successfully.")

    # 尝试合并并保存完整模型
    print("Starting to merge LoRA weights into the main model...")
    try:
        merged_model = policy_model.merge_and_unload()
        final_model_path = os.path.join(args.output_dir, "final_merged_model")
        merged_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Final merged model saved to {final_model_path}")
    except Exception as e:
        print(f"Error during model merging: {str(e)}")
        print("Saving unmerged model instead...")
        final_model_path = os.path.join(args.output_dir, "final_unmerged_model")
        policy_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Unmerged model saved to {final_model_path}")

    print("\nTraining process completed successfully!")

if __name__ == "__main__":
    args = parse_args()
    train(args)
