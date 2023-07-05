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
/nfs/turbo/coe-jag/zjsun/p6/bert-base-uncased/embeddings.pt --label bert
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6/doduo/embeddings.pt --label doduo
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6/google/tapas-base/embeddings.pt --label tapas
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6/roberta-base/embeddings.pt --label roberta
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6/t5-base/embeddings.pt --label t5
python3 k_means_clustering.py  --embedding_file \
/nfs/turbo/coe-jag/zjsun/p6/tabert/embeddings.pt --label tabert


