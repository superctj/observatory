#!/bin/bash
#SBATCH --job-name=p8_bert
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
python spider_fd_loader.py -m bert-base-uncased --root_dir /nfs/turbo/coe-jag/zjsun/data/spider_fd_artifact/fd_artifact