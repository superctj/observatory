#!/bin/bash
#SBATCH --job-name=Functional_Dependencies
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=5-00:00
#SBATCH --account=jag0
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --array=1
source ~/miniconda3/bin/activate
conda init
conda activate tabeval
root_dir=""
model_name="doduo"
save_dir=""
doduo_path=""
python Functional_Dependencies.py \
-m $model_name \
--root_dir $root_dir \
--save_dir $save_dir    \
--doduo_path $doduo_path