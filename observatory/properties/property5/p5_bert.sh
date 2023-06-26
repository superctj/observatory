#!/bin/bash
#SBATCH --job-name=p5_bert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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
python3 nextiajd_loader.py --testbed \
"testbedS" --root_dir "/nfs/turbo/coe-jag/zjsun/data/nextiajd_datasets" \
--model_name bert-base-uncased \
--n 1000 --start \
0 \
--num_tables 100000 \
--value 1000