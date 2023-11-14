#!/bin/bash

#SBATCH --job-name=feature
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=96GB
#SBATCH --output=slurm_%j.out


# Begin execution
module purge


singularity exec --nv \
            --overlay /scratch/ab10313/singularity_overlays/overlay-50G-10M.ext3:ro \
            /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
            /bin/bash -c "source /ext3/env.sh; python /home/ab10313/Projects/submeso_ML/scripts/resolution/fcnn_select_arch_1_4.py 10"
