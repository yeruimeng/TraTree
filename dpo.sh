#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3  
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

mkdir -p ./output

BASE_MODEL_PATH="./base"
REF_MODEL_PATH="./ref"
DATA_PATH="./data"
OUTPUT_DIR="./output"

python dpo_train.py \
    --base_model_path $BASE_MODEL_PATH \
    --ref_model_path $REF_MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR

