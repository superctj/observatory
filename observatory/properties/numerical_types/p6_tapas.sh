#!/bin/bash
#SBATCH --job-name=p6_tapas
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
python3 numerical_types.py  --root_dir "/home/zjsun/data/sotab_numerical_data_type_datasets/sotab_numerical_data_type_datasets" \
--model_name google/tapas-base --n 1000 