#!/bin/bash
#SBATCH --job-name=Turl_SP
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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
conda activate observatory
data_dir="/home/zjsun/Turl"
config_path="/home/zjsun/Turl/table-base-config_v2.json"
ckpt_path="/home/zjsun/Turl/pytorch_model.bin"
save_dir="SP_TD_beta"
sample_portion=0.25
python3 sample_turl.py \
--data_dir $data_dir \
--config_path $config_path \
--ckpt_path $ckpt_path \
--cuda_device 0 \
-s $save_dir \
-p $sample_portion
-l 0