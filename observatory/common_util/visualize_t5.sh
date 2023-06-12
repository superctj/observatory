#!/bin/bash
#SBATCH --job-name=Vis_t5
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=1-00:00
#SBATCH --account=jag0
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --array=1
source ~/miniconda3/bin/activate
conda init
conda activate tabeval
python visualize_embeddings.py --root_dir '/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD/t5-base' --mcv_threshold 0.2 --cosine_similarity_threshold 0.98 --n_components 2