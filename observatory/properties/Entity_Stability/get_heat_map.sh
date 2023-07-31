#!/bin/bash
#SBATCH --job-name=get_heat_map
#SBATCH --partition=standard
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
save_dir=""
dir1=""
dir2=""
label1=""
label2=""
python3 get_heat_map.py  --directories $dir1 $dir2 \
--labels $label1 $label2 \
--save_dir $save_dir \
--K_values 10 20 30 40 50 \