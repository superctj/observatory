#!/bin/bash
#SBATCH --job-name=Plot_distribution_abbreviation
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
python plot_distribution.py  --read_folders /nfs/turbo/coe-jag/zjsun/p4/p4_SD_abbreviation/bert-base-uncased/results \
/nfs/turbo/coe-jag/zjsun/p4/p4_SD_abbreviation/google/tapas-base/results \
/nfs/turbo/coe-jag/zjsun/p4/p4_SD_abbreviation/roberta-base/results \
/nfs/turbo/coe-jag/zjsun/p4/p4_SD_abbreviation/t5-base/results \
/nfs/turbo/coe-jag/zjsun/p4/p4_SD_abbreviation/tabert/results \
--labels bert tapas roberta t5 tabert --save_folder ./p4_plot2 --picture_name abbreviation.png