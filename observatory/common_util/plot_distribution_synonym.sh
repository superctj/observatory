#!/bin/bash
#SBATCH --job-name=Plot_distribution_synonym
#SBATCH --partition=standard
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
python plot_distribution.py --read_folder /home/zjsun/test_Perturbation_Robustness/Perturbation_Robustness/t5-base/results --save_folder ./p4_test --picture_name  t5_synonym.png --labels t5