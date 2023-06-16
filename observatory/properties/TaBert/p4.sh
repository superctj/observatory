#!/bin/bash
#SBATCH --job-name=p4_tabert
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
conda activate tabert
python3 evaluate_p4.py -o /home/zjsun/data/processed_db_data/original -c /home/zjsun/data/processed_db_data/abbreviation    -s p4_SD_abbreviation   -m tabert
python3 evaluate_p4.py -o /home/zjsun/data/processed_db_data/original -c /home/zjsun/data/processed_db_data/synonym  -s p4_SD_synonym  -m tabert