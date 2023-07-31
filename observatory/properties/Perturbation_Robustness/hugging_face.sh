#!/bin/bash
#SBATCH --job-name=Perturbation_Robustness
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=2-00:00
#SBATCH --account=jag0
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --array=1
source ~/miniconda3/bin/activate
conda init
conda activate tabeval
model_name=""
original_dir=""
abbreviation_dir=""
synonym_dir=""
abbreviation_save_dir=""
synonym_save_dir=""
python3 evaluate_Perturbation_Robustness.py \
--model_name $model_name \
-o $original_dir \
-c $abbreviation_dir  \
-s $abbreviation_save_dir  
python3 evaluate_Perturbation_Robustness.py \
--model_name $model_name \
-o $original_dir \
-c $synonym_dir  \
-s $synonym_save_dir