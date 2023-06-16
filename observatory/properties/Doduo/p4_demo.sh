#!/bin/bash
#SBATCH --job-name=p4_doduo
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
python3 evaluate_p4.py -o /home/zjsun/data/data_example/original -c /home/zjsun/data/data_example/changed_header    -s p4_SD_demo   -m doduo