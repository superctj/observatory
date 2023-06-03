#!/bin/bash
#SBATCH --job-name=Plot_CI
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
python plot_result.py --dirs /nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/bert-base-uncased/results /nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/google/tapas-base/results /nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/roberta-base/results /nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/t5-base/results /nfs/turbo/coe-jag/zjsun/col_insig/CI_TD_beta/google/tapas-base/results /nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/Turl/results  --labels bert tapas-old roberta t5 tapas-new* Turl* --avg_cosine_similarities_file CI_avg_cosine_similarities.txt --mcvs_file CI_mcvs.txt --plot_file CI_results.jpg --result_pairs_file CI.json