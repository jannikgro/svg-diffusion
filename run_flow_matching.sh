#!/bin/bash
#SBATCH --partition=ava_m.p
#SBATCH --nodelist=ava-m3
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=svg-flow-matching
#SBATCH --output=slurm-%j.log

cd /home/jgroenev/repositories/svg_diffusion
python classifier_prediction_flow_matching.py --resume flow_matching_plots/flow_model.pt --epochs 100000 --start-epoch 12000 --batch-size 2048
