import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm

# 从 build_tree_env.py 导入 => 生成 "id_to_env" (多任务)
from build_tree_env import build_all_envs_for_ids

###################################################
# 1) ValueHead
###################################################
class ValueHead(nn.Module):
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
    def forward(self, hidden):
        return self.mlp(hidden)

###################################################
# 2) PPOTrainer (多任务版本, 无测试环节)
###################################################
class PPOTrainer:
    def __init__(
        self,
        actor_model,
        ref_model,
        value_head,
        tokenizer,
        id_to_env: dict,   # 多任务: 不只是一个env
        output_dir: str,
        lr=5e-5,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.3,
        kl_coef=0.01,
        target_kl=0.05,
        min_kl_coef=0.001,
        max_kl_coef=0.05,
        n_steps=10,
        ppo_epochs=3,
        save_interval=5000
    ):
        """
        参数说明:
          actor_model: 13B + LoRA, 要训练的模型
          ref_model:   7B 参考模型(冻结)
          value_head:  价值网络
          tokenizer:   同actor使用的tokenizer
          id_to_env:   { id_str: TreeEnv }, 多任务环境
          output_dir:  保存checkpoint的位置
          lr:          学习率
          gamma, lam:  PPO中的折扣和GAE参数
          clip_ratio:  PPO ratio clip
          kl_coef:     初始KL系数
          ...
          n_steps:     每次采样的步数
          ppo_epochs:  每次更新时的epoch数
          save_interval: 每多少episodes保存一次checkpoint
        """
        self.actor_model = actor_model
        self.ref_model = ref_model
        self.value_head = value_head
        self.tokenizer = tokenizer

        self.id_to_env = id_to_env
        self.all_ids = list(id_to_env.keys())

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.kl_coef = kl_coef
        self.target_kl = target_kl
        self.min_kl_coef = min_kl_coef
        self.max_kl_coef = max_kl_coef
        self.n_steps = n_steps
        self.ppo_epochs = ppo_epochs
        self.save_interval = save_interval
        
        self.best_reward = float('-inf')
        
        # 记录统计(没有单独测试集,只记录train过程)
        self.metrics = {
            'episode_rewards': [],
            'kl_divergences': [],
            'policy_losses': [],
            'value_losses': [],
            'kl_coefs': []
        }
        
        self.optimizer = torch.optim.AdamW(
            list(self.actor_model.parameters()) + list(self.value_head.parameters()),
            lr=lr
        )

    def save_checkpoint(self, episode, is_best=False):
        """保存checkpoint(含LoRA权重)"""
        checkpoint = {
            'episode': episode,
            'actor_model_state': self.actor_model.state_dict(),
            'value_head_state': self.value_head.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'metrics': self.metrics,
            'best_reward': self.best_reward
        }
        ckpt_path = self.output_dir / f"checkpoint_ep{episode}.pt"
        torch.save(checkpoint, ckpt_path)
        print(f"[save_checkpoint] episode={episode} => {ckpt_path}")

        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f" => best model saved to {best_path}")

            # 保存LoRA权重
            lora_path = self.output_dir / "best_lora_weights"
            self.actor_model.save_pretrained(lora_path)
            print(f" => LoRA权重保存到 {lora_path}")

            # 合并LoRA并保存(可选)
            merged_path = self.output_dir / "best_merged_model"
            merged_model = self.actor_model.merge_and_unload()
            merged_model.save_pretrained(merged_path)
            self.tokenizer.save_pretrained(merged_path)
            print(f" => 合并后完整模型保存到 {merged_path}")

    def load_checkpoint(self, checkpoint_path):
        """如果要恢复训练可用"""
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.actor_model.load_state_dict(ckpt['actor_model_state'])
        self.value_head.load_state_dict(ckpt['value_head_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.metrics = ckpt['metrics']
        self.best_reward = ckpt['best_reward']
        print(f"[load_checkpoint] 加载自 {checkpoint_path}, best_reward={self.best_reward}")

    def collect_rollout(self, max_steps=10):
        """一次rollout:随机 pick 一个id => env"""
        chosen_id = random.choice(self.all_ids)
        env = self.id_to_env[chosen_id]

        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        
        s = env.reset()
        done = False
        
        for _ in range(max_steps):
            if done:
                break
            
            inputs = self.tokenizer(s, return_tensors='pt', truncation=True)
            inputs = {k: v.to(self.actor_model.device) for k,v in inputs.items()}

            with torch.no_grad():
                out = self.actor_model(**inputs, output_hidden_states=True, return_dict=True)
                hidden = out.hidden_states[-1][:, -1, :]
                logits = out.logits[:, -1, :]
                val = self.value_head(hidden).squeeze(0)
            
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            a_text = self.tokenizer.decode(action.item())
            s_next, r, done, info = env.step(a_text)
            
            states.append(s)
            actions.append(action.item())
            rewards.append(r)
            dones.append(done)
            log_probs.append(log_prob.detach())
            values.append(val.detach())

            s = s_next
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "log_probs": log_probs,
            "values": values
        }

    def compute_gae(self, rewards, values, dones, next_value=0.0):
        advantages, returns = [], []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma*next_value*mask - values[t]
            gae = delta + self.gamma*self.lam*mask*gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.actor_model.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.actor_model.device)
        return advantages, returns

    def update(self, rollout):
        actions = torch.tensor(rollout['actions'], device=self.actor_model.device)
        old_log_probs = torch.stack(rollout['log_probs']).to(self.actor_model.device)
        values_t = torch.stack(rollout['values']).to(self.actor_model.device)
        
        advantages, returns = self.compute_gae(
            rollout['rewards'],
            values_t,
            rollout['dones'],
            next_value=0.0
        )
        
        epoch_metrics = {'policy_losses': [], 'value_losses': [], 'kl_divs': []}
        
        for _ in range(self.ppo_epochs):
            new_log_probs_list = []
            new_values_list = []
            ref_log_probs_list = []

            for i, state_text in enumerate(rollout['states']):
                inputs = self.tokenizer(state_text, return_tensors='pt', truncation=True)
                inputs = {k: v.to(self.actor_model.device) for k,v in inputs.items()}

                out_actor = self.actor_model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_actor = out_actor.hidden_states[-1][:, -1, :]
                logits_actor = out_actor.logits[:, -1, :]

                with torch.no_grad():
                    out_ref = self.ref_model(**inputs, output_hidden_states=True, return_dict=True)
                    hidden_ref = out_ref.hidden_states[-1][:, -1, :]
                    logits_ref = out_ref.logits[:, -1, :]

                val_actor = self.value_head(hidden_actor).squeeze(0)
                val_ref = self.value_head(hidden_ref).squeeze(0).detach()

                # 若actor显著优于ref => 减小kl
                if val_actor > val_ref * 1.2:
                    self.kl_coef *= 0.9
                    self.kl_coef = max(self.kl_coef, self.min_kl_coef)

                dist_actor = Categorical(F.softmax(logits_actor, dim=-1))
                dist_ref = Categorical(F.softmax(logits_ref, dim=-1))

                a_tensor = torch.tensor([actions[i]], device=logits_actor.device)
                new_lp = dist_actor.log_prob(a_tensor)
                ref_lp = dist_ref.log_prob(a_tensor)

                new_log_probs_list.append(new_lp)
                new_values_list.append(val_actor)
                ref_log_probs_list.append(ref_lp)

            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list)
            ref_log_probs = torch.stack(ref_log_probs_list)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)*advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            v_clip = values_t + torch.clamp(new_values - values_t, -self.clip_ratio, self.clip_ratio)
            v_loss1 = (new_values - returns)**2
            v_loss2 = (v_clip - returns)**2
            value_loss = 0.5*torch.max(v_loss1, v_loss2).mean()

            kl = (new_log_probs - ref_log_probs).mean()
            kl_penalty = self.kl_coef * kl
            loss = policy_loss + value_loss + kl_penalty

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_model.parameters()) + list(self.value_head.parameters()),
                0.5
            )
            self.optimizer.step()

            epoch_metrics['policy_losses'].append(policy_loss.item())
            epoch_metrics['value_losses'].append(value_loss.item())
            epoch_metrics['kl_divs'].append(kl.item())

        return epoch_metrics

    def plot_training_curves(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

        # Episode Rewards
        ax1.plot(self.metrics['episode_rewards'], label='EpisodeReward')
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')

        # KL Divergence
        ax2.plot(self.metrics['kl_divergences'], label='KL Divergence')
        ax2.set_title('KL Divergence')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('KL')

        # KL Coefs
        ax3.plot(self.metrics['kl_coefs'], label='KL Coef')
        ax3.set_title('KL Coefficient')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Coef')

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png")
        plt.close()

    def save_training_log(self):
        log_path = self.output_dir / 'training_log.json'
        log_data = {
            'metrics': self.metrics,
            'best_reward': self.best_reward,
            'final_kl_coef': self.kl_coef
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

    def train(self, total_episodes=10000):
        """
        只做多任务训练,不额外测试. 
        记录的best_reward是训练rollout中出现的最大episode reward.
        """
        print(f"开始训练(多任务) => total_episodes={total_episodes}, output_dir={self.output_dir}")
        print(f"初始: lr={self.optimizer.param_groups[0]['lr']}, kl_coef={self.kl_coef}")

        for ep in tqdm(range(total_episodes), desc="Training"):
            rollout = self.collect_rollout(max_steps=self.n_steps)
            ep_reward = sum(rollout["rewards"])

            metrics = self.update(rollout)

            # 记录
            self.metrics['episode_rewards'].append(ep_reward)
            self.metrics['policy_losses'].append(np.mean(metrics['policy_losses']))
            self.metrics['value_losses'].append(np.mean(metrics['value_losses']))
            self.metrics['kl_divergences'].append(np.mean(metrics['kl_divs']))
            self.metrics['kl_coefs'].append(self.kl_coef)

            # 若是当前episode reward比best更好 => 保存best
            if ep_reward > self.best_reward:
                self.best_reward = ep_reward
                self.save_checkpoint(ep, is_best=True)

            # 定期保存checkpoint
            if (ep+1) % self.save_interval == 0:
                self.save_checkpoint(ep+1)
                self.plot_training_curves()
                self.save_training_log()
        
        # 训练结束,再保存一次
        self.save_checkpoint(total_episodes)
        self.plot_training_curves()
        self.save_training_log()
        print("训练完成!")

def main():
    output_dir = "ppo/multi_env_training"
    
    print("构建多任务 env (id_to_env) ...")
    id_to_env = build_all_envs_for_ids()
    print(f"共找到 {len(id_to_env)} 个 id => 多任务 PPO")

    # Reference模型
    print("加载Reference模型 (7B)...")
    ref_model_name = "sci_output/lora_sft_sci_weak_llama2/merged_model"
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name, use_fast=False)
    if ref_tokenizer.pad_token is None:
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
    ref_config = AutoConfig.from_pretrained(ref_model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_model_name, config=ref_config,
        load_in_8bit=True, device_map="auto"
    )
    ref_model.requires_grad_(False)

    # Actor模型 (13B) + LoRA
    print("加载Actor模型 (13B)...")
    actor_model_name = "sci_output/lora_sft_sci_strong_llama2/merged_model"
    actor_tokenizer = AutoTokenizer.from_pretrained(actor_model_name, use_fast=False)
    if actor_tokenizer.pad_token is None:
        actor_tokenizer.pad_token = actor_tokenizer.eos_token
    actor_config = AutoConfig.from_pretrained(actor_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        actor_model_name, config=actor_config,
        load_in_8bit=True, device_map="auto"
    )
    lora_config = LoraConfig(
        r=256,
        lora_alpha=128,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    actor_model = get_peft_model(base_model, lora_config)

    # ValueHead
    hidden_size = actor_model.config.hidden_size
    value_head = ValueHead(hidden_size).to("cuda:0")

    # 构建多任务PPOTrainer (无测试集)
    trainer = PPOTrainer(
        actor_model=actor_model,
        ref_model=ref_model,
        value_head=value_head,
        tokenizer=actor_tokenizer,
        id_to_env=id_to_env,
        output_dir=output_dir,
        lr=5e-5,
        kl_coef=0.01,
        n_steps=10,
        ppo_epochs=3,
        save_interval=5000
    )

    # 开始多任务训练
    trainer.train(total_episodes=20000)

if __name__ == "__main__":
    main()
