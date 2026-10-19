#!/bin/bash

#SBATCH --job-name=com_CNN_training
#SBATCH --output=eftest.out
#SBATCH --error=eftest.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

PROJECT="/ceph/project/p5_intelligent_seeds/P5/se_efnet"
SCRIPT="train_combined.py"

singularity exec --nv --bind $PROJECT/data:/data /ceph/container/pytorch/pytorch_26.08.sif \
	python3 $PROJECT/$SCRIPT \
	--seed 10 \
	--epochs 1 \
	--model-output $PROJECT/model.pth \
	--csv-output $PROJECT/output.csv \
	--dist-output $PROJECT/pred_output.csv