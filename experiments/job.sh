#!/bin/bash

#SBATCH --job-name=com_CNN_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# Safety options, making sure the script exits at command fails and undefined variables
set -euo pipefail

PROJECT_ROOT="$SLURM_SUBMIT_DIR"
EXPERIMENT_DIR="$PROJECT_ROOT/experiments"
DATA_PATH="/ceph/project/p5_intelligent_seeds/data"
MODULE="experiments.experiment"
CONTAINER="/ceph/project/p5_intelligent_seeds/project_container.sif"

SEED="$1"
TRAINING_ID="$2"
EXP_NO="$3"

echo "seed: $SEED"
echo "node: $(hostname)"
echo "gpu_index: $SLURM_JOB_GPUS"
echo "container: $CONTAINER"

cd "$PROJECT_ROOT"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

# If Singularity fails, it gives it 10 tries of 200 seconds to fix itself. Semi works for now - should be refactored
# Uses singularity container to run command true (if container works, it simply exits 0).
# The && break, breaks the loop if true correctly returns (no more errors)
# 2>/dev/null redirects stderr to /dev/null (virtual black hole).

for try in {1..10}; do
  singularity exec "$CONTAINER" true 2>/dev/null && break
  echo "Singularity failed (try $try), trying again in 200 seconds" >&2
  sleep 200
done

# Singularity executes the Python training model using required arguments
# --nv activates NVIDIA's GPUs. --bind "$DATA_PATH:/data" mounts the AI LAB pytorch container.
# --bind "$PROJECT_ROOT" mounts the python virtual environment for the compute node. Could be a container later on
singularity exec --nv --bind "$PROJECT_ROOT" --bind "$DATA_PATH:/data" "$CONTAINER" \
	python3 -m "$MODULE" \
	--seed "$SEED" \
	--epochs 60 \
	--training-id "$TRAINING_ID" \
	--exp "$EXP_NO" \
  ${3:+--csv-output "$4"} \
  ${4:+--dist-output "$5"}