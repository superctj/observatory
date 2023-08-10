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
conda activate tabert
results_dir="/home/zjsun/test" 
num_shuffle=1000
model_name="tabert"
tabert_bin="/home/zjsun/TaBert/TaBERT/tabert_base_k3/model.bin"
p=0.5
csv_dir="/nfs/turbo/coe-jag/zjsun/data/normal_TD"
python3 tabert_evaluate_Sample_Fidelity.py \
-r $csv_dir  \
-s $results_dir  \
-n $num_shuffle \
-m $model_name \
-p $p \
--tabert_bin $tabert_bin