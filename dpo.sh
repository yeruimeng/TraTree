# #!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2,3 
# python fastchat/train/mcts_train_improve.py \
#     --model_name sci_output/lora_sft_sci_strong_llama2/merged_model \
#     --ref_model_name sci_output/lora_sft_sci_weak_llama2/merged_model \
#     --train_file update_contrastive_trajectoriess_6.json \
#     --output_dir mcts/mcts_6_pair_reward_diff_sci_wts \
#     --per_device_train_batch_size 2 \
#     --num_train_epochs 3 \
#     --learning_rate 2e-5 \
#     --gradient_accumulation_steps 16 \
#     --warmup_ratio 0.03 \
#     --gradient_checkpointing \
#     --max_length 4096 \
#     --beta 0.2 \
#     --scaling_factor 25000.0 \
#     --lora_r 128 \
#     --lora_alpha 256 \
#     --save_steps 300 \
#     --logging_steps 50


# CUDA_VISIBLE_DEVICES=0,1,2,3 
# python fastchat/train/mcts_train_improve.py \
#   --model_name sci_output/lora_sft_sci_strong_llama2/merged_model \
#   --ref_model_name sci_output/lora_sft_sci_weak_llama2/merged_model \
#   --train_file update_contrastive_trajectoriess.json \
#   --output_dir mcts/mcts_6_pair_reward_diff_sci_wts \
#   --per_device_train_batch_size 2 \
#   --num_train_epochs 3 \
#   --learning_rate 2e-5 \
#   --gradient_accumulation_steps 16 \
#   --warmup_ratio 0.03 \
#   --gradient_checkpointing



# 下面这个是dpo训练时用的sh脚本