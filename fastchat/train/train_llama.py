# # # python fastchat/train/train.py
# # # ./train.sh

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
import argparse
import json
import math
import pathlib
from typing import Dict, Optional, Sequence, Union
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import LabelSmoother
from accelerate import dispatch_model

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/home/bhui/ML/ruimeng/ETO-main/Llama-2-7b-chat-hf")  # 修改模型路径
    parser.add_argument("--data_path", type=str, default="data/webshop_sft.json")
    parser.add_argument("--output_dir", type=str, default="output/new_weak_sft")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lazy_preprocess", action="store_true")
    return parser.parse_args()

def preprocess(sources, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    conversations = []
    for source in sources:
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for turn in source:
            if turn['from'] == 'human':
                messages.append({"role": "user", "content": turn['value']})
            elif turn['from'] == 'gpt':
                messages.append({"role": "assistant", "content": turn['value']})
        conversations.append(messages)

    # Apply chat template specific to Llama 2
    conversations = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Llama 2 specific parameters
            system_prompt="<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
            end_of_conversation_token="</s>",
            # Other parameters as needed
        )
        for messages in conversations
    ]

    encodings = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    labels = input_ids.clone()

    # Set padding tokens to be ignored in loss calculation
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
        sources = [example["conversations"] for example in raw_data]
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
        )

class LazySupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            attention_mask=ret["attention_mask"][0],
            labels=ret["labels"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=None)


# class CustomTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs.logits
#         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN_ID)
#         loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

#         # 1. 检查 loss 是否为 NaN
#         if torch.isnan(loss).any():
#             raise ValueError(f"NaN loss detected: {loss}")

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        
        # 将 logits 转换为 float32 以避免 NaN 问题
        logits = outputs.logits.float()  
        
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN_ID)
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # 1. 检查 loss 是否为 NaN
        if torch.isnan(loss).any():
            raise ValueError(f"NaN loss detected: {loss}")

        return (loss, outputs) if return_outputs else loss

    def _backward(self, loss, optimizer, gradient_accumulation_steps):
        loss.backward()

        # 2. 进行梯度裁剪
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)

        # 3. 检查参数是否包含 NaN
        for name, param in self.model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                raise ValueError(f"NaN gradient detected in parameter: {name}, gradient: {param.grad}")

        # 更新参数
        if (self.state.global_step + 1) % gradient_accumulation_steps == 0 or (
            self.state.global_step < self.state.max_steps and (
                self.control.should_epoch_end or self.control.should_training_end
            )
        ):
            optimizer.step()
            optimizer.zero_grad()

        # 4. 检查更新后的参数是否包含 NaN
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                raise ValueError(f"NaN detected in parameter: {name} after optimizer step")

    def add_custom_callback(self):
        """
        Add a custom callback to the trainer.
        """
        class CustomCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                """
                Log additional information.
                """
                if torch.isnan(logs["loss"]).any():
                    raise ValueError("NaN loss detected during logging.")
                # Add any other checks here

        self.add_callback(CustomCallback)

    def train(self, resume_from_checkpoint: Optional[Union[str, bool]] = None):
        """
        Override the train method to add custom callbacks.
        """
        self.add_custom_callback()
        super().train(resume_from_checkpoint)

def train(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=args.model_max_length,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map={"": 0, "transformer": 1},  # 明确指定哪些层在哪个GPU上
        # device_map="auto", 
        torch_dtype=torch.float32,
        use_cache=False, 
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=args)
    train_dataset = data_module["train_dataset"]

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="epoch",
        evaluation_strategy="no",
        max_grad_norm=0.5,
        fp16=False,
    )

    # Use the custom trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    args = parse_args()
    train(args)