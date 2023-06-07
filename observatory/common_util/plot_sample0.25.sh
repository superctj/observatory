#!/bin/bash
#SBATCH --job-name=Plot_sample_0.25
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
python plot_result.py --dirs /nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_beta/bert-base-uncased/results /nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_beta/google/tapas-base/results /nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_beta/roberta-base/results /nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_beta/t5-base/results /nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_gamma/google/tapas-base/results /nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_beta/Turl/results --labels bert tapas-old roberta t5 tapas-new* Turl* --avg_cosine_similarities_file sample_0.25_avg_cosine_similarities.txt --mcvs_file sample_0.25_mcvs.txt --plot_file sample_0.25_results.jpg --result_pairs_file sample_0.25.json