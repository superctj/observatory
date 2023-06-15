#!/bin/bash
#SBATCH --job-name=p5_doduo
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
python3 nextiajd_loader.py --testbed "testbedXS" --root_dir "/nfs/turbo/coe-jag/zjsun/data/nextiajd_datasets" \
--model_name doduo --n 1000 --r 3