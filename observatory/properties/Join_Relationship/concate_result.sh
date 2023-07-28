#!/bin/bash
#SBATCH --job-name=concate_result
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
python3 concate_result.py --path /nfs/turbo/coe-jag/zjsun/p5/testbedS/tabert
python3 concate_result.py --path /nfs/turbo/coe-jag/zjsun/p5/testbedS/doduo
python3 concate_result.py --path /nfs/turbo/coe-jag/zjsun/p5/testbedS/bert-base-uncased
python3 concate_result.py --path /nfs/turbo/coe-jag/zjsun/p5/testbedS/google/tapas-base
python3 concate_result.py --path /nfs/turbo/coe-jag/zjsun/p5/testbedS/roberta-base
python3 concate_result.py --path /nfs/turbo/coe-jag/zjsun/p5/testbedS/t5-base