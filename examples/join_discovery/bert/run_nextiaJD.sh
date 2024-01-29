#!/bin/bash

# Stop on errors
set -Eeuo pipefail

data_dir="/ssd/congtj/observatory/nextiajd_datasets"
testbed="testbedS"
model_name="bert-base-uncased" # "t5-base"
threshold=0.7
# k=10
# num_samples=-1

for num_samples in -1 10 100 1000
do
for k in 2 3 5 10
do
    python topk_search_nextiaJD.py \
    --dataset_dir="${data_dir}/${testbed}/datasets/" \
    --metadata_path="${data_dir}/${testbed}/datasetInformation_${testbed}.csv" \
    --ground_truth_path="${data_dir}/${testbed}/groundTruth_${testbed}.csv" \
    --index_dir="/ssd/congtj/observatory/experiments/join_discovery/${model_name}/indexes/${testbed}/" \
    --output_dir="./outputs/${testbed}/" \
    --model_name=${model_name} \
    --lsh_threshold=${threshold} \
    --top_k=${k} \
    --num_samples=${num_samples}
done
done