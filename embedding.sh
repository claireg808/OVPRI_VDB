#!/bin/bash
#SBATCH --gres=gpu:80g:1
#SBATCH --mem=100G
#SBATCH --output=emb_out.txt
#SBATCH --qos=short
#SBATCH --job-name=embed

source vdb_venv/bin/activate

python embedding.py