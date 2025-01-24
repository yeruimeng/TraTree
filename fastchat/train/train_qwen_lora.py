import os
import argparse
import json
import torch
import numpy as np
import random
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_pt_utils import LabelSmoother
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# Optional: For reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    # Qwen model
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    # Data and output paths
    parser.add_argument("--data_path", type=str, default="data/sft_data.json")
    parser.add_argument("--output_dir", type=str, default="output/qwen_lora_sft")
    # Training hyperparameters
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--model_max_length", type=int, default=4096)
    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    # Data preprocessing
    parser.add_argument("--lazy_preprocess", action="store_true", help="Use lazy preprocessing.")
    return parser.parse_args()

def apply_qwen_chat_template(tokenizer, conversation: List[Dict], system_text: str):
    """
    Convert a list of {from: 'human'/'gpt', value: str} 
    into Qwen's chat prompt format using apply_chat_template.
    """
    messages = []
    # A global system prompt (you can customize)
    messages.append({"role": "system", "content": system_text})

    for turn in conversation:
        if turn["from"] == "human":
            messages.append({"role": "user", "content": turn["value"]})
        elif turn["from"] == "gpt":
            messages.append({"role": "assistant", "content": turn["value"]})

    # Qwen's helper to insert special tokens, roles, etc.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False  # For training, we usually include the entire conversation
    )
    return text

def preprocess(tokenizer, sources, system_text: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
    """
    sources: list of conversation examples, 
             each example is a list of dict [{from:..., value:...}, ...].
    """
    # Convert each conversation to the Qwen prompt text
    prompt_texts = [
        apply_qwen_chat_template(tokenizer, conv, system_text=system_text) 
        for conv in sources
    ]
    
    # Tokenize
    encodings = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    
    input_ids = encodings.input_ids
    attention_mask = encodings.attention_mask
    
    # Create labels (same as input_ids, but ignore pad)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer, system_text="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
        super().__init__()
        # raw_data is a list of items, each has "conversations"
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(tokenizer, sources, system_text)
        self.input_ids = data_dict["input_ids"]
        self.attention_mask = data_dict["attention_mask"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "labels": self.labels[i],
        }

class LazySupervisedDataset(Dataset):
    """
    Processes each sample on-the-fly (useful for large datasets).
    """
    def __init__(self, raw_data, tokenizer, system_text="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
        super().__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.system_text = system_text
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        
        example = self.raw_data[i]
        conversation = example["conversations"]
        data_dict = preprocess(self.tokenizer, [conversation], self.system_text)
        # data_dict has batch dimension = 1, so take the first row
        item = {
            "input_ids": data_dict["input_ids"][0],
            "attention_mask": data_dict["attention_mask"][0],
            "labels": data_dict["labels"][0],
        }
        self.cached_data_dict[i] = item
        return item

def make_supervised_data_module(tokenizer, data_args):
    print("Loading data from:", data_args.data_path)
    raw_data = json.load(open(data_args.data_path, "r", encoding="utf-8"))
    
    if data_args.lazy_preprocess:
        dataset = LazySupervisedDataset(raw_data, tokenizer=tokenizer)
    else:
        dataset = SupervisedDataset(raw_data, tokenizer=tokenizer)
    
    return dict(train_dataset=dataset, eval_dataset=None)

def train(args):
    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        model_max_length=args.model_max_length,
    )
    # Ensure pad_token is defined (Qwen has <|padding|> by default, but let's be safe)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Loading the base model
    print("Loading base model:", args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Optional: Prepare model for 4-bit or 8-bit training if you wish
    # If you only want fp16 training, you can remove the following line.
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Adjust if Qwen layer names differ
    )
    model = get_peft_model(model, lora_config)
    
    # Create dataset
    data_module = make_supervised_data_module(tokenizer, args)
    train_dataset = data_module["train_dataset"]
    # (Optional) no eval_dataset provided here, you can split or add your eval if desired

    # Training setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=True,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        evaluation_strategy="no",  # or "steps" if you have eval data
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()

    # Save LoRA weights
    print("Saving LoRA adapter weights...")
    model.save_pretrained(os.path.join(args.output_dir, "lora_weights"))
    
    # Merge the LoRA weights into the base model and save the merged model
    print("Merging LoRA weights and saving the full model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(os.path.join(args.output_dir, "merged_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "merged_model"))

if __name__ == "__main__":
    args = parse_args()
    train(args)
