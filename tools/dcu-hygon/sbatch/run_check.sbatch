#!/bin/bash
#SBATCH --job-name=check
#SBATCH --partition=kshdtest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=dcu:1
#SBATCH --mem=32G
#SBATCH --output=job.%j.log
#SBATCH --error=job.%j.log

module rm compiler/rocm/2.9
module load compiler/rocm/4.0.1
module list

source activate ~/miniconda3/envs/rocm401

srun rocm-smi && \
     python3 -c "import paddle; print('version =', paddle.__version__)" && \
     python3 -c "import paddle; print('commit =', paddle.version.commit)" && \
     python3 -c "import paddle; paddle.utils.run_check()"
