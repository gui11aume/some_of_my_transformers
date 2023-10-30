#!/bin/bash
#SBATCH --requeue
#SBATCH --account=def-gfilion
#SBATCH --job-name=Treeformer_training
#SBATCH --output=$SCRATCH/Projects/Treeformer/output_file_%j.out
#SBATCH --error=$SCRATCH/Projects/Treeformer/error_file_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --partition=compute_full_node


module --ignore_cache load cuda/11.4.4
module --ignore_cache load anaconda3
source activate pytorch

cd $HOME/Projects/Treeformer

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

make
