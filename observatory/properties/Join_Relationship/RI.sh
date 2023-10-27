#!/bin/bash
#SBATCH --job-name=CI_Join_Relationship
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
conda activate tabeval
# specify number of iterations
n=1
# specify starting number
start=0
# specify testbed and root_dir
testbed="testbedS"
root_dir="/nfs/turbo/coe-jag/zjsun/data/nextiajd_datasets"
model_name="t5-base"
save_dir="/nfs/turbo/coe-jag/zjsun/revision"
batch_size=256
num_tables=10
# loop n times
for (( i=0; i<$n; i++ ))
do
  # calculate the current start number
  current_start=$((start + (i * $num_tables)))

  python3 RI_nextiajd_loader.py \
  --testbed $testbed \
  --root_dir $root_dir \
  --model_name $model_name \
  --start \
  $current_start \
  --num_tables $num_tables \
  --value 10000 \
  --save_dir $save_dir \
  --num_shuffles 100  \
  --batch_size $batch_size

  echo "Completed iteration $((i+1)) out of $n"
done

echo "All iterations completed"