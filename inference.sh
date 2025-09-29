#!/bin/bash
#SBATCH --job-name=hbench-infer
#SBATCH --nodes=1
#SBATCH --gres=gpu:A6000:1
#SBATCH --mem=48GB
#SBATCH --time=8:00:00

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.8/targets/x86_64-linux/lib/
export RAY_TMPDIR=/home/cyang3/OpenRLHF/.ray_tmp
# module load cuda-11.8
# module load gcc-7.4

SPLITS=("train" "validation" "test")

# MODEL=Qwen/Qwen2.5-7B-Instruct
# MODEL=/data/tir/projects/tir5/users/cyang3/checkpoint/healthbench_gpt5/Qwen2.5-7B-Instruct/global_step51_hf
# MODEL=/data/tir/projects/tir5/users/cyang3/checkpoint/healthbench_gpt5/Qwen2.5-7B-Instruct/global_step102_hf
MODEL=/data/tir/projects/tir5/users/cyang3/checkpoint/healthbench_gpt5/Qwen2.5-7B-Instruct/

for split in "${SPLITS[@]}"; do
  echo "Processing split: $split"

  deepspeed --module openrlhf.cli.batch_inference \
    --eval_task generate_vllm \
    --pretrain $MODEL \
    --bf16 \
    --max_new_tokens 2048 \
    --prompt_max_len 2048 \
    --dataset malusamayo/healthbench_gpt5 \
    --dataset_split "$split" \
    --input_key prompt \
    --apply_chat_template \
    --temperature 0.7 \
    --zero_stage 0 \
    --best_of_n 4 \
    --enable_prefix_caching \
    --tp_size 1 \
    --max_num_seqs 64 \
    --output_path "healthbench_qwen2.5_generate_vllm_epoch3_${split}.jsonl"
done
