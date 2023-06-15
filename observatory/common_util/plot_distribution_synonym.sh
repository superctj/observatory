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
python plot_distribution.py --read_folder /nfs/turbo/coe-jag/zjsun/p4/p4_SD_synonym/bert-base-uncased/results --save_folder ./p4_plot1 --picture_name  bert_synonym.png
python plot_distribution.py --read_folder /nfs/turbo/coe-jag/zjsun/p4/p4_SD_synonym/google/tapas-base/results --save_folder ./p4_plot1 --picture_name  tapas_synonym.png
python plot_distribution.py --read_folder /nfs/turbo/coe-jag/zjsun/p4/p4_SD_synonym/roberta-base/results --save_folder ./p4_plot1 --picture_name  roberta_synonym.png
python plot_distribution.py --read_folder /nfs/turbo/coe-jag/zjsun/p4/p4_SD_synonym/t5-base/results --save_folder ./p4_plot1 --picture_name  t5_synonym.png
python plot_distribution.py --read_folder /nfs/turbo/coe-jag/zjsun/p4/p4_SD_synonym/tabert/results --save_folder ./p4_plot1 --picture_name  tabert_synonym.png