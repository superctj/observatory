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
python plot_p5.py --results_file /nfs/turbo/coe-jag/zjsun/p5/testbedS/t5-base/results.pt \
--save_dir ./p5_S \
--pic_name t5_S.png
python plot_p5.py --results_file /nfs/turbo/coe-jag/zjsun/p5/testbedS/roberta-base/results.pt \
--save_dir ./p5_S \
--pic_name roberta_S.png
python plot_p5.py --results_file /nfs/turbo/coe-jag/zjsun/p5/testbedS/google/tapas-base/results.pt \
--save_dir ./p5_S \
--pic_name tapas_S.png
python plot_p5.py --results_file /nfs/turbo/coe-jag/zjsun/p5/testbedS/bert-base-uncased/results.pt \
--save_dir ./p5_S \
--pic_name bert_S.png
python plot_p5.py --results_file /nfs/turbo/coe-jag/zjsun/p5/testbedS/tabert/results.pt \
--save_dir ./p5_S \
--pic_name tabert_S.png
python plot_p5.py --results_file /nfs/turbo/coe-jag/zjsun/p5/testbedS/doduo/results.pt \
--save_dir ./p5_S \
--pic_name doduo_S.png