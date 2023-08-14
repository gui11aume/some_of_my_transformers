#!/bin/bash
#SBATCH --account=def-gfilion
#SBATCH --job-name=test_job
#SBATCH --output=/scratch/g/gfilion/gfilion/output_file_%j.out
#SBATCH --error=/scratch/g/gfilion/gfilion/error_file_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --partition=compute_full_node

module --ignore_cache load cuda/11.4.4
module --ignore_cache load anaconda3
source activate pytorch

cd /scratch/g/gfilion/gfilion

export CUBLAS_WORKSPACE_CONFIG=:4096:2
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

srun python ./pretrain_MLM.py majestic_tokenizer.json flat_claims.txt.gz trained.pt

# Starting the job this way triggers the following warning
# (but the job still runs fine).

#SBATCH WARNING:
# Your job can not use all 32 cores associated with each gpu.
#SBATCH WARNING:
# You have requested the maximum amount of time.  If this jobs does not actually
# need this amount of time, it may run sooner if you request less time.
#SBATCH: 2 warnings were found.
#SBATCH: Job submission will be attempted despite warnings.
#SBATCH: For more info see https://docs.alliancecan.ca/wiki/Mist

# For more information see
# https://lightning.ai/docs/fabric/stable/guide/multi_node/slurm.html
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.plugins.environments.SLURMEnvironment.html

