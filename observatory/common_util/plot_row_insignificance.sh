#!/bin/bash
#SBATCH --job-name=Plot_row_insignificance
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
python plot_box_plot.py  --read_folders \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_beta/bert-base-uncased \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_beta/roberta-base \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_beta/t5-base \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_gamma/google/tapas-base \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_gamma/tabert \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD/Turl \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_gamma/doduo \
--labels BERT RoBERTa T5 TAPAS TaBERT TURL DODUO \
--save_folder ./row_insignificance \
--picture_name row_insignificance.png