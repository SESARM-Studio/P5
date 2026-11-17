#!/usr/bin/env bash

#SBATCH --job-name=build_project_container
#SBATCH --output=build_project_container.out
#SBATCH --error=build_project_container.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=12:00:00

export SINGULARITY_TMPDIR=$HOME/.singularity/tmp
export SINGULARITY_CACHEDIR=$HOME/.singularity/cache
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

# The path to the definition file
input_def="project_container.def"

# The resulting container image
output_sif="project_container.sif"

singularity build --fakeroot $output_sif $input_def
