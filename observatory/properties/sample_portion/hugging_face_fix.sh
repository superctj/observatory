#!/bin/bash
#SBATCH --job-name=Hugging_face_fix_SP
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
python3 evaluate_sample_portion.py -r normal_TD  -s SP_TD_beta  -n 0  -p 0.75
python3 evaluate_sample_portion.py -r normal_TD  -s SP_TD_beta  -n 0  -p 0.25
python3 evaluate_sample_portion.py -r normal_TD  -s SP_TD_beta  -n 0  -p 0.5