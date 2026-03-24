#!/bin/bash
#SBATCH --partition=ava_m.p
#SBATCH --nodelist=ava-m6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name=svg-diff
#SBATCH --output=slurm-%j.log

cd /home/jgroenev/repositories/svg_diffusion
python train_svg_diffusion.py
