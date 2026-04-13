#!/bin/bash
#SBATCH --partition=ava_m.p
#SBATCH --nodelist=ava-m5
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=svg-flow-matching
#SBATCH --output=slurm-%j.log

cd /home/jgroenev/repositories/svg_diffusion
python classifier_prediction_flow_matching.py
