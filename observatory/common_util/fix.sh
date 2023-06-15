#!/bin/bash
#SBATCH --job-name=fix_result
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
python3 fix_result.py  --parent_directories \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_beta/Turl \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_gamma/doduo \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_gamma/google/tapas-base \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.25/SP_TD_gamma/tabert \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.5/SP_TD_beta/bert-base-uncased \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.5/SP_TD_beta/roberta-base \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.5/SP_TD_beta/t5-base \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.5/SP_TD_beta/Turl \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.5/SP_TD_gamma/doduo \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.5/SP_TD_gamma/google/tapas-base \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.5/SP_TD_gamma/tabert \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.75/SP_TD_beta/bert-base-uncased \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.75/SP_TD_beta/roberta-base \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.75/SP_TD_beta/t5-base \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.75/SP_TD_beta/Turl \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.75/SP_TD_gamma/doduo \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.75/SP_TD_gamma/google/tapas-base \
/nfs/turbo/coe-jag/zjsun/sample_portion/0.75/SP_TD_gamma/tabert \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_beta/bert-base-uncased \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_beta/roberta-base \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_beta/t5-base \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD/Turl \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_gamma/google/tapas-base \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_gamma/doduo \
/nfs/turbo/coe-jag/zjsun/row_insig/RI_TD_gamma/tabert \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/bert-base-uncased \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/roberta-base \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/t5-base \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD/Turl \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD_beta/google/tapas-base \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD_beta/doduo \
/nfs/turbo/coe-jag/zjsun/col_insig/CI_TD_beta/tabert

