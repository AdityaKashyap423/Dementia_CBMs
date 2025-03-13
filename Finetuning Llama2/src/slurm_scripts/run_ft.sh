#!/bin/bash
#
#SBATCH --partition=p_nlp
#SBATCH --job-name=pl_ft
#SBATCH --output=/nlp/data/kashyap/%x.%j.out
#SBATCH --error=/nlp/data/kashyap/%x.%j.err
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --constraint=48GBgpu
#SBATCH --mem-per-cpu=14G
#SBATCH --cpus-per-task=8

srun python src/pl_ft.py $@