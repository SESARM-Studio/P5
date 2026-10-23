#!/bin/bash
set -euo pipefail

#SBATCH --job-name=com_CNN_training
#SBATCH --output=eftest.out
#SBATCH --error=eftest.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

PROJECT="/ceph/project/p5_intelligent_seeds/P5/experiment0"
DATA_PATH="/ceph/project/p5_intelligent_seeds/data"
SCRIPT="experiment0"

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"


singularity exec --nv --bind $PROJECT/../../data:/data /ceph/container/pytorch/pytorch_26.08.sif \
	"$PYTHON" -m $PROJECT.$SCRIPT \
	--seed $1 \
	--epochs 60 \
	--csv-output $PROJECT/outputs/output_$1.csv \
	--dist-output $PROJECT/outputs/pred_output_$1.csv \
	--training-id $2