#!/bin/bash
#SBATCH --job-name=Heterogeneous_Context
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
root_dir="/home/zjsun/data/sotab_numerical_data_type_datasets/sotab_numerical_data_type_datasets"
model_name="doduo"
non_text_save_folder="/home/zjsun/test"
text_save_folder="/home/zjsun/test"
non_text_metadata_path="/nfs/turbo/coe-jag/zjsun/data/sotab_data_type_datasets/sotab_data_type_datasets/nontext_types_10-classes_metadata.csv"
text_metadata_path="/nfs/turbo/coe-jag/zjsun/data/sotab_data_type_datasets/sotab_data_type_datasets/text_types_10-classes_metadata.csv"
doduo_path="/home/zjsun/DuDuo/doduo"
python3 doduo_numerical_types.py  --root_dir $root_dir \
--model_name $model_name --n 1000 \
--save_folder $non_text_save_folder \
--metadata_path $non_text_metadata_path \
--doduo_path $doduo_path
python3 doduo_numerical_types.py  --root_dir $root_dir \
--model_name $model_name --n 1000 \
--save_folder $text_save_folder \
--metadata_path $text_metadata_path \
--doduo_path $doduo_path