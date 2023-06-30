#!/bin/bash
#SBATCH --job-name=p5_roberta
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=5-00:00
#SBATCH --account=jag0
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --array=1
source ~/miniconda3/bin/activate
conda init
conda activate tabeval
python3 nextiajd_loader.py --testbed \
"testbedS" --root_dir "/nfs/turbo/coe-jag/zjsun/data/nextiajd_datasets" \
--model_name roberta-base \
--n 1000 --start \
10000 \
--num_tables 20000 \
--value 1000
python3 nextiajd_loader.py --testbed \
"testbedS" --root_dir "/nfs/turbo/coe-jag/zjsun/data/nextiajd_datasets" \
--model_name roberta-base \
--n 1000 --start \
30000 \
--num_tables 20000 \
--value 1000
python3 nextiajd_loader.py --testbed \
"testbedS" --root_dir "/nfs/turbo/coe-jag/zjsun/data/nextiajd_datasets" \
--model_name roberta-base \
--n 1000 --start \
50000 \
--num_tables 20000 \
--value 1000