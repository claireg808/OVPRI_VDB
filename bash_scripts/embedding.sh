#!/bin/bash
#SBATCH --gres=gpu:40g:1
#SBATCH --mem=100G
#SBATCH --output=pre_processing/emb_out.log
#SBATCH --qos=short
#SBATCH --job-name=embed

source vdb_venv/bin/activate

python pre_processing/embedding.py