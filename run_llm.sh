#!/bin/bash
#SBATCH --gres=gpu:80g:1
#SBATCH --mem=100G
#SBATCH --output=launch_out.txt
#SBATCH --qos=short
#SBATCH --job-name=run_llm

source vdb_venv/bin/activate

vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --max-model-len 35904