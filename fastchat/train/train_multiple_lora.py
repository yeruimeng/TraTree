import os
import argparse
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from typing import Dict, List

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Llama-2-7b-chat-hf")
    parser.add_argument("--data_path", type=str, default="data/webshop_sft.json")
    parser.add_argument("--output_dir", type=str, default="output/multi_lora_sft_weak_llama2")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lazy_preprocess", action="store_true", help="Use lazy preprocessing")
    parser.add_argument("--num_lora_models", type=int, default=3, help="Number of LoRA models to train")
    parser.add_argument("--sampling_ratio", type=float, default=0.8, help="Ratio of data to sample for each LoRA model")
    parser.add_argument("--sampling_method", type=str, default="random", choices=["random", "bootstrap", "stratified"], help="Sampling method to use")
    parser.add_argument("--stratify_on_length", action="store_true", help="Stratify based on conversation length")
    parser.add_argument("--stratify_on_complexity", action="store_true", help="Stratify based on conversation complexity")
    return parser.parse_args()

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

    def format_conversation(conversation):
        formatted = f"{B_INST} {B_SYS}You are a helpful assistant.{E_SYS}\n\n"
        for i, message in enumerate(conversation):
            if message["role"] == "user":
                formatted += f"{message['content']} {E_INST} "
            elif message["role"] == "assistant":
                formatted += f"{message['content']} </s><s>{B_INST} "
        formatted = formatted.rstrip(f" </s><s>{B_INST} ")
        return formatted

    conversations = [format_conversation(conv) for conv in conversations]

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

    labels[labels == tokenizer.pad_token_id] = IGNORE_TOKEN_ID

    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

class SupervisedDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
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

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    sampling_ratio: float,
    sampling_method: str,
    stratify_on_length: bool,
    stratify_on_complexity: bool,
) -> Dict:
    dataset_cls = LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    print("Loading data...")
    train_json = json.load(open(data_args.data_path, "r"))

    # Implement sampling methods
    if sampling_method == 'random':
        num_samples = int(len(train_json) * sampling_ratio)
        sampled_data = random.sample(train_json, num_samples)
    elif sampling_method == 'bootstrap':
        num_samples = int(len(train_json) * sampling_ratio)
        sampled_data = random.choices(train_json, k=num_samples)
    elif sampling_method == 'stratified':
        # Compute stratification keys
        stratify_key = []
        for example in train_json:
            length = sum(len(turn['value']) for turn in example['conversations'])
            complexity = len(example['conversations'])
            key = ''
            if stratify_on_length:
                key += f"length_{length // 100}"  # Binning lengths
            if stratify_on_complexity:
                key += f"_complexity_{complexity}"
            if key == '':
                key = 'default'
            stratify_key.append(key)

        # Group examples by strata
        strata = {}
        for key, example in zip(stratify_key, train_json):
            if key not in strata:
                strata[key] = []
            strata[key].append(example)

        # Sample proportionally from each stratum
        sampled_data = []
        for key, examples in strata.items():
            num_examples_in_stratum = len(examples)
            sample_size = max(int(num_examples_in_stratum * sampling_ratio), 1)  # Ensure at least one sample
            sampled_examples = random.sample(examples, min(sample_size, num_examples_in_stratum))
            sampled_data.extend(sampled_examples)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    train_dataset = dataset_cls(sampled_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None)

def train_lora_model(args, model_index: int):
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
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, config)

    print(f"Prepare the dataset for LoRA model {model_index}...")
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=args,
        sampling_ratio=args.sampling_ratio,
        sampling_method=args.sampling_method,
        stratify_on_length=args.stratify_on_length,
        stratify_on_complexity=args.stratify_on_complexity,
    )
    train_dataset = data_module["train_dataset"]

    output_dir = os.path.join(args.output_dir, f"lora_model_{model_index}")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=True,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        evaluation_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save LoRA weights
    model.save_pretrained(os.path.join(output_dir, "lora_weights"))

    # Merge LoRA weights with the original model and save
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(os.path.join(output_dir, "merged_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "merged_model"))

def train(args):
    for i in range(args.num_lora_models):
        print(f"Training LoRA model {i+1}/{args.num_lora_models}")
        train_lora_model(args, i)

if __name__ == "__main__":
    args = parse_args()
    train(args)
