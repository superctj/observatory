#!/bin/bash
#SBATCH --job-name=p8_results
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=2-00:00
#SBATCH --account=jag0
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --array=1
source ~/miniconda3/bin/activate
conda init
conda activate tabeval
python3 p8_results.py  --folders \
/nfs/turbo/coe-jag/zjsun/FD/bert-base-uncased \
/nfs/turbo/coe-jag/zjsun/FD/roberta-base \
/nfs/turbo/coe-jag/zjsun/FD/t5-base \
/nfs/turbo/coe-jag/zjsun/FD/google/tapas-base \
/nfs/turbo/coe-jag/zjsun/FD/doduo \
--labels BERT RoBERTa T5 TAPAS   DODUO 