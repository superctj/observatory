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
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_beta/bert-base-uncased/results \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_beta/roberta-base/results \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_beta/t5-base/results \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_gamma/google/tapas-base/results \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_gamma/tabert/results \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD/Turl/results \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_gamma/doduo/results \
--labels BERT RoBERTa T5 TAPAS TaBERT TURL DODUO \
--save_folder ./row_insignificance \
--picture_name row_insignificance.png