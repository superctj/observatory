#!/bin/bash
set -e

dataset="spider"
model_name="bert-base-uncased"
threshold=0.7
k=2

python topk_search_spider.py \
--dataset_dir="/data/spider_artifact/db_csv_extended/" \
--metadata_path="/data/spider_artifact/dev_join_data_extended.csv" \
--index_dir="/data/join_discovery_indexes/bert/indexes/${dataset}/" \
--output_dir="./outputs/${dataset}/" \
--model_name=${model_name} \
--lsh_threshold=${threshold} \
--top_k=${k}