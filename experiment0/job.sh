#!/bin/bash

#SBATCH --job-name=com_CNN_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="/ceph/project/p5_intelligent_seeds/P5"
EXPERIMENT_DIR="$PROJECT_ROOT/experiment0"
DATA_PATH="/ceph/project/p5_intelligent_seeds/data"
MODULE="experiment0.experiment0"
CONTAINER="/ceph/container/pytorch/pytorch_26.08.sif"

SEED="$1"
TRAINING_ID="$2"

echo "seed=$1"

cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

# If Singularity fails, try again. 
for try in {1..10}; do
  singularity exec /ceph/container/pytorch/pytorch_26.08.sif true 2>/dev/null && break
  echo "Singularity failed (try $try), trying again in 200 seconds" >&2
  sleep 200
done


singularity exec --nv --bind "$PROJECT_ROOT" --bind "$DATA_PATH:/data" "$CONTAINER" \
	"$PYTHON" -m "$MODULE" \
	--seed "$SEED" \
	--epochs 60 \
	--csv-output "$EXPERIMENT_DIR/outputs/output_$SEED.csv" \
	--dist-output "$EXPERIMENT_DIR/outputs/pred_output_$SEED.csv" \
	--training-id "$TRAINING_ID"
