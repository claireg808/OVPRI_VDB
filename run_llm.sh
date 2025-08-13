#!/bin/bash
#SBATCH --gres=gpu:80g:1
#SBATCH --mem=100G
#SBATCH --output=launch_out.txt
#SBATCH --qos=short
#SBATCH --job-name=run_llm

module load python/3.11

source vdb_venv/bin/activate

export HUGGINGFACE_HUB_TOKEN=$(cat ~/OVPRI_VDB/.hf_token)
hf auth login --token "$HUGGINGFACE_HUB_TOKEN"

# start vLLM in the background
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --served-model-name llama3 \
    > vllm_out.txt 2>&1 &

# start Streamlit
streamlit run front_end.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0