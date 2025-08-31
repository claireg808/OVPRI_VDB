#!/bin/bash
#SBATCH --gres=gpu:80g:1
#SBATCH --mem=100G
#SBATCH --output=rag/launch_out.log
#SBATCH --qos=short
#SBATCH --job-name=run_llm

source vdb_venv/bin/activate

export HUGGINGFACE_HUB_TOKEN=$(cat ~/OVPRI_VDB/.hf_token)
hf auth login --token "$HUGGINGFACE_HUB_TOKEN"

# start vLLM in the background
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --served-model-name llama3 \
    > rag/vllm_out.log 2>&1 &

echo "Waiting for vLLM to start..."
sleep 60
echo "Launching frontend"

# start frontend
flask --app rag/receive_messages run \
    --debug \
    --host=0.0.0.0 \
    --port=5000