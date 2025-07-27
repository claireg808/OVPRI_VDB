#!/bin/bash
#SBATCH --gres=gpu:80g:1
#SBATCH --mem=100G
#SBATCH --output=launch_out.txt
#SBATCH --qos=short
#SBATCH --job-name=run_llm

source vdb_venv/bin/activate

export HUGGINGFACE_HUB_TOKEN=$(cat ~/OVPRI_VDB/.hf_token)

huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"

vllm serve meta-llama/Meta-Llama-3-8B-Instruct --served-model-name llama3