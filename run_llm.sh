#!/bin/bash
#SBATCH --gres=gpu:80g:1
#SBATCH --mem=100G
#SBATCH --output=launch_out.txt
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --job-name=run_llm


source vdb_venv/bin/activate

# log system info
echo "Running on $(hostname)"
echo "Python: $(which python)"
echo "Working dir: $(pwd)"

export HUGGINGFACE_HUB_TOKEN=$(cat ~/OVPRI_VDB/.hf_token)
huggingface-cli login --token "$HUGGINGFACE_HUB_TOKEN"

# start vLLM in the background
echo "Starting vLLM..."
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --served-model-name llama3 \
    > vllm.log 2>&1 &

sleep 10

# start Streamlit
echo "Starting Streamlit..."
streamlit run front_end.py \
    --server.port 8501 \
    --server.headless true \
    --server.address 0.0.0.0