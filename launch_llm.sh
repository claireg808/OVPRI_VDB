#!/bin/bash
#SBATCH --gres=gpu:80g:1
#SBATCH --mem=100G
#SBATCH --output=launch_output.txt
#SBATCH --qos=short
#SBATCH --job-name=launch

source vdb_venv/bin/activate

vllm serve Qwen/Qwen3-Embedding-4B --max-model-len 32000