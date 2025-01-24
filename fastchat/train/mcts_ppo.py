import os
import json
import math
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)
import transformers
from tqdm import tqdm
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

# 定义Llama对话模板常量
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# 基础系统提示
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant to do some scientific experiment in an environment.
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway
You should explore the environment and find the items you need to complete the experiment.
You can teleport to any room in one step.
All containers in the environment have already been opened, you can directly get items from the containers."""

@dataclass
class TrainingConfig:
    # 基础模型配置
    model_name: str = "meta-llama/Llama-2-13b-chat-hf"
    ref_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    model_max_length: int = 4096
    
    # LoRA配置
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ])
    
    # PPO配置
    num_epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 32
    max_grad_norm: float = 1.0
    
    # PPO超参数
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip_ratio: float = 0.2
    c1: float = 1.0  # 价值损失系数
    c2: float = 0.01  # 熵损失系数
    target_kl: float = 0.01
    
    # 训练配置
    use_4bit: bool = False
    use_8bit: bool = True
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    train_data_path: str = "update_contrastive_trajectoriess_6.json"
    output_dir: str = "ppo/ppo_sci_wts"
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    max_steps: int = 10000
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

def format_conversation(conversation: List[Dict[str, str]]) -> str:
    """格式化对话为Llama-2-chat格式"""
    formatted = ""
    
    if not conversation:
        return formatted
        
    # 处理system消息
    if conversation[0]["role"] == "system":
        formatted = f"{B_INST} {B_SYS}{conversation[0]['content']}{E_SYS}"
        if len(conversation) > 1:
            formatted += f"{conversation[1]['content']}{E_INST} "
        conversation = conversation[2:]
    
    # 处理后续的对话
    for i, msg in enumerate(conversation):
        if msg["role"] == "user":
            if i == len(conversation) - 1:
                formatted += f"{B_INST} {msg['content'].strip()} {E_INST}"
            else:
                formatted += f"{B_INST} {msg['content'].strip()} {E_INST} "
        elif msg["role"] == "assistant":
            formatted += f"{msg['content'].strip()} "
            
    return formatted.strip()

class TreeNode:
    def __init__(self, state=None, action=None, thought=None, parent=None):
        self.state = state
        self.action = action
        self.thought = thought
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_reward = 0.0
        
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        return child

class TrajectoryTreeEnv:
    def __init__(self, root_node: TreeNode, embedding_model: SentenceTransformer):
        self.root = root_node
        self.current_node = root_node
        self.embedding_model = embedding_model
        self.history = []
        self.conversation_history = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]
        self.reset()
        
    def reset(self):
        self.current_node = self.root
        self.history = []
        self.conversation_history = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
        ]
        return self._get_state_representation()
        
    def step(self, action: str):
        # 将action添加到对话历史
        self.conversation_history.append({
            "role": "assistant",
            "content": action
        })
        
        # 找到下一个节点
        next_node = self._find_best_matching_child(action)
        
        if next_node is None:
            return self._get_state_representation(), -1.0, True, {}
            
        # 计算奖励
        reward = self._calculate_reward(next_node)
        
        # 更新当前状态
        self.current_node = next_node
        self.history.append(next_node)
        
        # 如果有新的状态，添加到对话历史
        if next_node.state:
            self.conversation_history.append({
                "role": "user",
                "content": next_node.state
            })
            
        done = len(next_node.children) == 0
        
        return self._get_state_representation(), reward, done, {}
        
    def _get_state_representation(self) -> str:
        """返回当前状态的字符串表示"""
        return format_conversation(self.conversation_history)
        
    def _find_best_matching_child(self, action: str) -> Optional[TreeNode]:
        if not self.current_node.children:
            return None
            
        best_similarity = float('-inf')
        best_child = None
        
        # 计算action的embedding
        action_embedding = self.embedding_model.encode(action)
        
        # 找到最匹配的子节点
        for child in self.current_node.children:
            if not child.action:
                continue
                
            child_action_embedding = self.embedding_model.encode(child.action)
            similarity = np.dot(action_embedding, child_action_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_child = child
                
        # 只有当相似度超过阈值时才返回子节点
        return best_child if best_similarity > 0.7 else None
        
    def _calculate_reward(self, node: TreeNode) -> float:
        if node.visit_count == 0:
            return 0.0
            
        # 计算当前路径质量
        path_quality = 0.0
        current = node
        depth = 0
        
        while current.parent:
            visit_ratio = current.visit_count / max(1, current.parent.visit_count)
            avg_reward = current.total_reward / max(1, current.visit_count)
            path_quality += visit_ratio * avg_reward * (0.95 ** depth)
            current = current.parent
            depth += 1
            
        return path_quality

class ActorCriticLoRA(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=False,
            model_max_length=config.model_max_length,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Loading model configuration...")
        model_config = AutoConfig.from_pretrained(config.model_name)
        
        if config.gradient_checkpointing:
            model_config.use_cache = False
            
        print("Loading base model...")
        # 准备模型加载参数
        model_kwargs = {
            "config": model_config,
            "device_map": "auto",
        }
        
        if config.use_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["torch_dtype"] = torch.float16
        elif config.use_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32
            
        if config.use_flash_attention:
            model_kwargs["use_flash_attention_2"] = True
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            **model_kwargs
        )
        
        if config.gradient_checkpointing:
            self.base_model.gradient_checkpointing_enable()
            
        # 准备LoRA配置
        print("Preparing LoRA config...")
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 为8bit训练准备模型
        if config.use_8bit or config.use_4bit:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
            
        # 应用LoRA
        print("Applying LoRA...")
        self.model = get_peft_model(self.base_model, peft_config)
        
        # 添加价值头
        self.value_head = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(self.model.device)
        
        # 打印训练参数信息
        print("\nTrainable parameters:")
        self.model.print_trainable_parameters()
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 获取最后一层的隐藏状态
        last_hidden = outputs.hidden_states[-1]
        
        # 计算动作概率
        logits = outputs.logits[:, -1, :]  # 只使用最后一个token的输出
        action_probs = F.softmax(logits, dim=-1)
        
        # 计算状态价值
        pooled = last_hidden.mean(dim=1)
        value = self.value_head(pooled)
        
        return action_probs, value
        
    def get_policy_output(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            action_probs, _ = self.forward(input_ids, attention_mask)
        return action_probs
        
    def get_value(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        with torch.no_grad():
            _, value = self.forward(input_ids, attention_mask)
        return value
        
    def save_pretrained(self, save_dir: str):
        """保存模型权重和配置"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存LoRA权重
        self.model.save_pretrained(save_dir)
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # 保存价值头
        torch.save(
            self.value_head.state_dict(),
            os.path.join(save_dir, "value_head.pt")
        )
        
    def load_pretrained(self, load_dir: str):
        """加载预训练的权重"""
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(
            self.model,
            load_dir,
            is_trainable=True
        )
        
        # 加载价值头
        value_head_path = os.path.join(load_dir, "value_head.pt")
        if os.path.exists(value_head_path):
            self.value_head.load_state_dict(torch.load(value_head_path))

class PPOTrainer:
    def __init__(
        self, 
        env: TrajectoryTreeEnv, 
        model: ActorCriticLoRA, 
        config: TrainingConfig
    ):
        self.env = env
        self.model = model
        self.config = config
        
        # 设置优化器
        trainable_params = [
            {"params": self.model.value_head.parameters()},
            {"params": self.model.model.parameters()}
        ]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 用于跟踪训练状态
        self.global_step = 0
        self.train_stats = {
            'actor_loss': [],
            'critic_loss': [],
            'entropy_loss': [],
            'total_loss': [],
            'rewards': [],
            'episode_lengths': [],
            'kl_divergences': []
        }
        
    def collect_rollout(self, num_steps: int) -> Dict:
        """收集轨迹数据"""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        state = self.env.reset()
        done = False
        
        for _ in range(num_steps):
            if done:
                state = self.env.reset()
                done = False
                
            # 将状态转换为模型输入
            state_inputs = self.model.tokenizer(
                state,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.config.model_max_length
            ).to(self.model.device)
            
            # 获取动作和价值预测
            with torch.no_grad():
                action_probs, value = self.model(
                    state_inputs.input_ids,
                    state_inputs.attention_mask
                )
                
            # 采样动作
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # 将token ID转换为文本
            action_text = self.model.tokenizer.decode(action.item())
            
            # 与环境交互
            next_state, reward, done, _ = self.env.step(action_text)
            
            # 存储transition
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            state = next_state
            
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_probs': log_probs,
            'dones': dones
        }

    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        dones: List[bool]
    ) -> tuple:
        """计算广义优势估计"""
        advantages = []
        returns = []
        advantage = 0
        next_value = 0 if dones[-1] else values[-1]
        
        for r, v, done in zip(reversed(rewards), reversed(values), reversed(dones)):
            td_error = r + self.config.gamma * next_value * (1 - done) - v
            advantage = td_error + self.config.gamma * self.config.gae_lambda * (1 - done) * advantage
            next_value = v
            
            advantages.insert(0, advantage)
            returns.insert(0, advantage + v)
            
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.model.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.model.device)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def update(self, rollout: Dict):
        """更新策略和价值网络"""
        states = rollout['states']
        actions = torch.tensor(rollout['actions'], device=self.model.device)
        old_log_probs = torch.tensor(rollout['log_probs'], device=self.model.device)
        
        # 计算优势和回报
        advantages, returns = self.compute_gae(
            rollout['rewards'],
            rollout['values'],
            rollout['dones']
        )
        
        # 处理states
        state_inputs = self.model.tokenizer(
            states,
            padding=True,
            truncation=True,
            max_length=self.config.model_max_length,
            return_tensors='pt'
        ).to(self.model.device)
        
        # PPO更新
        for _ in range(self.config.num_epochs):
            # 获取新的动作概率和价值
            action_probs, values = self.model(
                state_inputs.input_ids,
                state_inputs.attention_mask
            )
            
            # 计算新的log概率
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            # 计算ratio
            ratio = (new_log_probs - old_log_probs).exp()
            
            # 计算actor损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1 - self.config.clip_ratio,
                1 + self.config.clip_ratio
            ) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 计算critic损失
            value_pred_clipped = rollout['values'] + torch.clamp(
                values.squeeze() - rollout['values'],
                -self.config.value_clip_ratio,
                self.config.value_clip_ratio
            )
            value_losses = (values.squeeze() - returns) ** 2
            value_losses_clipped = (value_pred_clipped - returns) ** 2
            critic_loss = torch.max(value_losses, value_losses_clipped).mean()
            
            # 计算熵损失
            entropy_loss = -dist.entropy().mean()
            
            # 计算总损失
            loss = (
                actor_loss + 
                self.config.c1 * critic_loss + 
                self.config.c2 * entropy_loss
            )
            
            # 更新模型
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            # 记录损失
            self.train_stats['actor_loss'].append(actor_loss.item())
            self.train_stats['critic_loss'].append(critic_loss.item())
            self.train_stats['entropy_loss'].append(entropy_loss.item())
            self.train_stats['total_loss'].append(loss.item())
            
            # 计算KL散度
            approx_kl = ((old_log_probs - new_log_probs) ** 2).mean().item()
            self.train_stats['kl_divergences'].append(approx_kl)
            
            # 如果KL散度过大，提前停止更新
            if approx_kl > self.config.target_kl:
                break

    def train(self):
        """训练主循环"""
        self.model.train()
        
        pbar = tqdm(total=self.config.max_steps, desc="Training")
        steps_taken = 0
        
        try:
            while steps_taken < self.config.max_steps:
                # 收集rollout数据
                rollout = self.collect_rollout(self.config.batch_size)
                
                # 更新模型
                self.update(rollout)
                
                steps_taken += len(rollout['states'])
                self.global_step += 1
                pbar.update(len(rollout['states']))
                
                # 记录统计信息
                self.train_stats['rewards'].extend(rollout['rewards'])
                self.train_stats['episode_lengths'].append(len(rollout['states']))
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
                    
                # 打印日志
                if self.global_step % self.config.logging_steps == 0:
                    self.log_stats()
                    
        except Exception as e:
            print(f"Error during training: {str(e)}")
            self.save_checkpoint(emergency=True)
            raise e
            
        finally:
            pbar.close()
            self.save_final_model()