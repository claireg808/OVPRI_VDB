#!/bin/bash
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=100G
#SBATCH --output=rag_out.txt
#SBATCH --qos=short
#SBATCH --job-name=rag

source vdb_venv/bin/activate

python rag.py