#!/bin/bash
#SBATCH --job-name=Turl_CI
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
conda activate tabeval
python3 col_shuffle_turl.py --data_dir "/home/congtj/observatory/data" --config_path "/home/congtj/observatory/observatory/models/TURL/configs/table-base-config_v2.json" --ckpt_path "/ssd/congtj/observatory/turl_models/pytorch_model.bin" --cuda_device 1 -s CI_TD -l 0