#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_lora_llama.py \
    --model_name_or_path ./model \
    --data_path ./data \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 2e-5 \
    --lora_r 256 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --model_max_length 4096 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 