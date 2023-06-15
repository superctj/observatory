#!/bin/bash
#SBATCH --job-name=Plot_p5
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
python plot_p5.py --results_file /nfs/turbo/coe-jag/zjsun/p5/testbedXS/bert-base-uncased/results.pt \
--save_dir ./p5_plot \
--pic_name bert_XS.png