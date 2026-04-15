#!/bin/bash
#SBATCH --partition=ava_m.p
#SBATCH --nodelist=ava-m5
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=svg-overfit
#SBATCH --output=slurm-%j.log

cd /home/jgroenev/repositories/svg_diffusion
python train_svg_diffusion.py --resume /home/jgroenev/repositories/svg_diffusion/checkpoints/559272/latest.pt --overfit
