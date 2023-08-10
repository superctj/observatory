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
conda activate observatory
model_name="doduo"
original_dir="/home/zjsun/data/processed_db_data/original"
abbreviation_dir="/home/zjsun/data/processed_db_data/abbreviation"
synonym_dir="/home/zjsun/data/processed_db_data/synonym"
abbreviation_save_dir="/home/zjsun/test_Perturbation_Robustness"
synonym_save_dir="/home/zjsun/test_Perturbation_Robustness"
doduo_path="/home/zjsun/DuDuo/doduo"
python3 doduo_evaluate_Perturbation_Robustness.py \
--model_name $model_name \
-o $original_dir \
-c $abbreviation_dir  \
-s $abbreviation_save_dir  \
--doduo_path $doduo_path
python3 doduo_evaluate_Perturbation_Robustness.py \
--model_name $model_name \
-o $original_dir \
-c $synonym_dir  \
-s $synonym_save_dir \
--doduo_path $doduo_path