#!/bin/bash
#SBATCH --job-name=zjsun
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=1-00:00
#SBATCH --account=jag0
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --array=1
source ~/miniconda3/bin/activate
conda init
conda activate tabeval
python3 row_shuffle_turl.py   -s RI_TD   -l 2354