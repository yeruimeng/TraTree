
# # mcts是这个指令
# #!/bin/bash

# # 清除任何现有的 CUDA 设置
# unset CUDA_VISIBLE_DEVICES

# # 显示初始 GPU 状态
# echo "Initial GPU status:"
# nvidia-smi


# # 设置 CUDA 设备
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# echo "Set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# # 显示设置后的 GPU 状态
# echo "GPU status after setting CUDA_VISIBLE_DEVICES:"
# nvidia-smi

# # 验证 PyTorch 是否正确识别 GPU
# python -c "import torch; print('PyTorch sees', torch.cuda.device_count(), 'GPUs')"

# # Run the training script
# python fastchat/train/best_of_N.py \
#     --model_name_or_path webshop_output/lora_sft_strong_llama2_web_v1/merged_model \
#     --data_path sft_7B_webshop_Best_of_N.json \
#     --output_dir baseline/Best_of_N_webshop \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 2e-4 \
#     --lora_r 128 \
#     --lora_alpha 128 \
#     --lora_dropout 0.05 \
#     --model_max_length 4096 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.03 \
#     --weight_decay 0.0 



# # #!/bin/bash

# # # Set CUDA devices for GPU
# # export CUDA_VISIBLE_DEVICES=1,2,3

# # # Create output directory if it doesn't exist
# # OUTPUT_DIR="mcts/contrastive_mcts_optimal"
# # mkdir -p $OUTPUT_DIR

# # # Run the training script
# # python fastchat/train/mcts_train_improve.py \
# #     --model_name_or_path output/lora_sft_sci_strong_llama2/merged_model \
# #     --data_path contrastive_trajectories_path_pair.json \
# #     --output_dir $OUTPUT_DIR \
# #     --num_train_epochs 3 \
# #     --per_device_train_batch_size 1 \
# #     --gradient_accumulation_steps 16 \
# #     --learning_rate 2e-4 \
# #     --lora_r 128 \
# #     --lora_alpha 256 \
# #     --lora_dropout 0.05 \
# #     --model_max_length 4096 \
# #     --warmup_ratio 0.1 \
# #     --weight_decay 0.0 \
# #     --save_strategy steps \
# #     --save_steps 50 \
# #     --logging_steps 10 \
# #     --save_total_limit 5 

# # # Check if training completed successfully
# # if [ $? -eq 0 ]; then
# #     echo "Training completed successfully!"
# #     echo "Models saved to: $OUTPUT_DIR"
# #     echo "- LoRA weights: $OUTPUT_DIR/lora_weights"
# #     echo "- Merged model: $OUTPUT_DIR/merged_model"
# #     echo "- Best model: $OUTPUT_DIR/best_model"
# # else
# #     echo "Training failed!"
# #     exit 1
# # fi


# # --lora_alpha一开始是256的 那个时候效果比较好  但是跑不动  先用这个试试

# # #!/bin/bash
# # # Set CUDA devices for GPU
# # export CUDA_VISIBLE_DEVICES=0,1,2,3

# # # Run the training script
# # python fastchat/train/mcts_train_improve.py \
# #     --model_name_or_path output/lora_sft_sci_strong_llama2/merged_model \
# #     --data_path contrastive_trajectories_path_pair.json \
# #     --output_dir mcts/contrastive_mcts_optimal \
# #     --num_train_epochs 3 \
# #     --per_device_train_batch_size 2 \
# #     --gradient_accumulation_steps 16 \
# #     --learning_rate 2e-4 \
# #     --lora_r 128 \
# #     --lora_alpha 256 \
# #     --lora_dropout 0.1 \
# #     --model_max_length 4096 \
# #     --lr_scheduler_type cosine \
# #     --weight_decay 0.0 \
# #     --warmup_ratio 0.1 \
# #     --lazy_preprocess \
# #     --lazy_preprocess \
# #     --contrast_weight 0.5 \
# #     --temperature 0.1


# # #!/bin/bash
# # export CUDA_VISIBLE_DEVICES=0,1,2,3

# # python fastchat/train/mcts_train_improve.py \
# #     --model_name_or_path output/lora_sft_sci_strong_llama2/merged_model \
# #     --data_path enhanced_optimal_trajectories.json \
# #     --output_dir mcts/contrastive_mcts_optimal \
# #     --num_train_epochs 3 \
# #     --per_device_train_batch_size 2 \
# #     --gradient_accumulation_steps 8 \
# #     --learning_rate 2e-4 \
# #     --lora_r 128 \
# #     --lora_alpha 256 \
# #     --lora_dropout 0.1 \
# #     --model_max_length 4096 \
# #     --lr_scheduler_type cosine \
# #     --weight_decay 0.0 \
# #     --warmup_ratio 0.1 \
# #     --temperature 0.7 \
# #     --kl_weight 0.1 \
# #     --min_temperature 0.5 \
# #     --max_temperature 1.0 \
# #     --min_kl_weight 0.05 \
# #     --max_kl_weight 0.2 \
# #     --save_steps 500 \
# #     --logging_steps 10






# # #!/bin/bash
# # Set CUDA devices for GPU
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# # Run the training script
# python fastchat/train/train_lora_llama.py \
#     --model_name_or_path Llama-2-13b-chat-hf \
#     --data_path data/alfworld_sft.json \
#     --instruction_path instruction.txt \
#     --output_dir output/lora_sft_weak_llama_sci \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 2 \
#     --learning_rate 1e-5 \
#     --lora_r 64 \
#     --lora_alpha 128 \
#     --lora_dropout 0.05 \
#     --model_max_length 4096 \
#     --lr_scheduler_type cosine \
#     --weight_decay 0.0 \
#     --warmup_ratio 0.1 \
#     --lazy_preprocess


#!/bin/bash


# # Set CUDA devices for GPU
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# # Run the training script
# python fastchat/train/mcts_train_improve.py \
#     --model_name_or_path output/lora_sft_sci_strong_llama2/merged_model \
#     --data_path all_advantage_trajectories.json \
#     --output_dir mcts/lora_wts_sci_mcts_optimal \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --learning_rate 5e-6 \
#     --lora_r 128 \
#     --lora_alpha 256 \
#     --lora_dropout 0.1 \
#     --model_max_length 4096 \
#     --lr_scheduler_type cosine \
#     --weight_decay 0.0 \
#     --warmup_ratio 0.1 \
#     --lazy_preprocess


# # 下面是lora sft的  可以用
# Set CUDA devices for GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Run the training script
python fastchat/train/train_lora_llama.py \
    --model_name_or_path Llama-2-13b-chat-hf \
    --data_path data/alfworld_sft.json \
    --output_dir alfworld_output/lora_sft_strong_model \
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