#!/bin/bash
#SBATCH --job-name=Entity_Stability
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
conda activate observatory
model_name="turl"
save_dir="/home/zjsun/test"
data_dir="/home/zjsun/Turl"
file_path="/home/zjsun/row_insig/test_tables.jsonl"
config="/home/zjsun/Turl/table-base-config_v2.json"
ckpt_path="/home/zjsun/Turl/pytorch_model.bin"
python entity_stability.py \
-m $model_name \
--save_dir $save_dir \
--data_dir $data_dir \
--file_path $file_path \
--config $config \
--ckpt_path $ckpt_path
