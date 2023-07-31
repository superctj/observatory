#!/bin/bash
#SBATCH --job-name=Sample_Fidelity
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
file_path=""
results_dir="" 
num_shuffle=""
model_name="" 
p=0.5
csv_dir="./Turl_csv_dataset"
python3 turl2normal.py --file_path $file_path\
--directory $csv_dir
python3 evaluate_Sample_Fidelity.py \
-r $csv_dir  \
-s $results_dir  \
-n $num_shuffle \
-m $model_name \
-p $p 