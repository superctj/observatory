#!/bin/bash
#SBATCH --job-name=Plot_column_insignificance
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
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/bert-base-uncased/results \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/roberta-base/results \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/t5-base/results \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD_beta/google/tapas-base/results \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD_beta/tabert/results \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/Turl/results \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD_beta/doduo/results \
--labels BERT RoBERTa T5 TAPAS TaBERT TURL DODUO \
--save_folder ./column_insignificance \
--picture_name column_insignificance.png