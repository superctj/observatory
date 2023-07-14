#!/bin/bash
#SBATCH --job-name=p6_t5
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
python3 numerical_types.py  --root_dir /nfs/turbo/coe-jag/zjsun/data/sotab_data_type_datasets/sotab_data_type_datasets \
--model_name t5-base --n 1000 --save_folder p6_non_text --metadata_path nontext_types_10-classes_metadata.csv
python3 numerical_types.py  --root_dir /nfs/turbo/coe-jag/zjsun/data/sotab_data_type_datasets/sotab_data_type_datasets \
--model_name t5-base --n 1000 --save_folder p6_text --metadata_path text_types_10-classes_metadata.csv