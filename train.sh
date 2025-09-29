#!/bin/bash
#SBATCH --job-name=hbench-sft
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:4
#SBATCH --mem=48GB
#SBATCH --time=12:00:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.8/targets/x86_64-linux/lib/
# module load cuda-11.8
# module load gcc-7.4

# Set Hugging Face environment variables
# export HF_HOME=/data/user_data/cyang3/.hf_cache
# export HF_HUB_CACHE=/data/hf_cache/hub
# export HF_DATASETS_CACHE=/data/hf_cache/datasets
# export HF_HUB_OFFLINE=1

source .env
echo "WANDB_API_KEY: $WANDB_API_KEY"

deepspeed --module openrlhf.cli.train_sft \
   --save_path /data/tir/projects/tir5/users/cyang3/checkpoint/healthbench_gpt5_n2/Qwen2.5-7B-Instruct \
   --ckpt_path /data/tir/projects/tir5/users/cyang3/checkpoint/healthbench_gpt5_n2/Qwen2.5-7B-Instruct \
   --save_steps 102 \
   --logging_steps 1 \
   --eval_steps 51 \
   --train_batch_size 64 \
   --micro_train_batch_size 2 \
   --pretrain Qwen/Qwen2.5-7B-Instruct \
   --bf16 \
   --max_epochs 3 \
   --max_len 4096 \
   --zero_stage 3 \
   --learning_rate 1e-5 \
   --dataset malusamayo/healthbench_gpt5_n2 \
   --input_key prompt \
   --output_key answer \
   --dataset_split train \
   --eval_dataset malusamayo/healthbench_gpt5_n2 \
   --eval_split validation \
   --max_samples 10000 \
   --apply_chat_template \
   --packing_samples \
   --save_hf_ckpt \
   --max_ckpt_num 10 \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --wandb_project healthbench-sft \

# Support HF tokenizer.apply_chat_template
# --apply_chat_template 
# --tokenizer_chat_template {HF Chat Template}

# Support RingAttention
# pip install ring_flash_attn
#   --ring_attn_size 2 \
#   --ring_head_stride 2 \

# Multi-turn fine-tuning loss
# --multiturn

# Can also be used for continued pre-training
# --pretrain_mode