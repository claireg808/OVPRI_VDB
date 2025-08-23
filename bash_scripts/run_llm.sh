#!/bin/bash
#SBATCH --gres=gpu:80g:2
#SBATCH --mem=100G
#SBATCH --output=rag_and_frontend/launch_out.log
#SBATCH --qos=short
#SBATCH --job-name=run_llm

source vdb_venv/bin/activate

export HUGGINGFACE_HUB_TOKEN=$(cat ~/OVPRI_VDB/.hf_token)
hf auth login --token "$HUGGINGFACE_HUB_TOKEN"

# start vLLM in the background
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --served-model-name llama3 \
    > rag_and_frontend/vllm_out.log 2>&1 &

# start Streamlit
streamlit run rag_and_frontend/front_end.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0