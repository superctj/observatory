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
/home/zjsun/test/Row_Order_Insignificance/t5-base/results \
/home/zjsun/test/Row_Order_Insignificance/doduo/results \
--labels T5 doduo \
--save_folder ./row_insignificance \
--picture_name row_insignificance.png