#!/bin/bash
#SBATCH --job-name=get_result
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
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6_non_text/bert-base-uncased/embeddings.pt --label bert --text False
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6_non_text/doduo/embeddings.pt --label doduo  --text False
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6_non_text/google/tapas-base/embeddings.pt --label tapas  --text False
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6_non_text/roberta-base/embeddings.pt --label roberta  --text False
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6_non_text/t5-base/embeddings.pt --label t5  --text False
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6_non_text/tabert/embeddings.pt --label tabert  --text False