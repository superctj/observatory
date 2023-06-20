#!/bin/bash
#SBATCH --job-name=Plot_coefficients_XS
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
python plot_coefficient.py  --save_dir="./p5/XS" \
--files \
/nfs/turbo/coe-jag/zjsun/p5/testbedXS/bert-base-uncased/results.pt \
/nfs/turbo/coe-jag/zjsun/p5/testbedXS/roberta-base/results.pt \
/nfs/turbo/coe-jag/zjsun/p5/testbedXS/t5-base/results.pt \
/nfs/turbo/coe-jag/zjsun/p5/testbedXS/google/tapas-base/results.pt \
/nfs/turbo/coe-jag/zjsun/p5/testbedXS/tabert/results.pt \
/nfs/turbo/coe-jag/zjsun/p5/testbedXS/doduo/results.pt \
--labels BERT RoBERTa T5 TAPAS TaBERT  DODUO