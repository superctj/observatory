#!/bin/bash
#SBATCH --job-name=Join_Relationship
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=5-00:00
#SBATCH --account=jag0
#SBATCH --output=/home/%u/logs/%x-%j.log
#SBATCH --array=1
source ~/miniconda3/bin/activate
conda init
conda activate tabeval
# specify number of iterations
n=4
# specify starting number
start=0
# specify testbed and root_dir
testbed=""
root_dir=""
model_name="doduo"
doduo_path="/home/zjsun/DuDuo/doduo"
# loop n times
for (( i=0; i<$n; i++ ))
do
  # calculate the current start number
  current_start=$((start + (i * 10000)))

  python3 doduo_nextiajd_loader.py \
  --testbed $testbed \
  --root_dir $root_dir \
  --model_name $model_name \
  --n 1000 --start \
  $current_start \
  --num_tables 10000 \
  --value 1000 \
  --doduo_path $doduo_path
  echo "Completed iteration $((i+1)) out of $n"
done

echo "All iterations completed"