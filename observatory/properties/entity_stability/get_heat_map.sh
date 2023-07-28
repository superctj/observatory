#!/bin/bash
#SBATCH --job-name=get_heat_map
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
python3 get_heat_map.py  --directories \
/nfs/turbo/coe-jag/zjsun/cell_embeddings/bert-base-uncased/entity_embeddings \
/nfs/turbo/coe-jag/zjsun/cell_embeddings/roberta-base/entity_embeddings \
/nfs/turbo/coe-jag/zjsun/cell_embeddings/t5-base/entity_embeddings \
/nfs/turbo/coe-jag/zjsun/cell_embeddings/google/tapas-base/entity_embeddings \
/nfs/turbo/coe-jag/zjsun/cell_embeddings/turl/entity_embeddings \
/nfs/turbo/coe-jag/zjsun/cell_embeddings/doduo/entity_embeddings \
--labels BERT RoBERTa T5 TAPAS  TURL DODUO \
--save_dir /home/zjsun/Plot/p7 \
--K_values 10 20 30 40 50 \
--if_only_plot True \
--if_double_entity True 