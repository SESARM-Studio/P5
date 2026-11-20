from db_scripts.database_seed import get_seed
from db_scripts.database_api import initialize_training
from db_scripts.database_login import setup_supabase_client
import subprocess
import time
import argparse
from pathlib import Path
import sys

# How many jobs a user can queue at a time. The maximum is defined by AI LAB
MAX_JOBS = 8
SECONDS_BEFORE_RETRY = 30

# User id is a required argument. The UID corresponds to the run_by column in the training_run table
parser = argparse.ArgumentParser()
parser.add_argument("--uid", required=True)
parser.add_argument("--csv-output")
parser.add_argument("--dist-output")
parser.add_argument("--seed-file", default="seeds.txt")
args = parser.parse_args()

# Determines the root dir as the parrents parrent directory of the file,
# because of the structure: root/experiments/run_experiment1.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent



# Setup for local run (without DB)
if args.csv_output:
    seed_path = Path(__file__).parent / args.seed_file
    with open(seed_path, 'r') as file:
        txt_seeds = [int(line.strip()) for line in file.readlines()]
    txt_seeds.append(None)

# Setup for Supabase DB. Currently uses temporary config id, as the integration isn't yet complete
else:
    client = setup_supabase_client()
temp_config_id = 'e4fecdcf-76a0-4e29-ae21-852decdd4513'


def new_seed():
    if args.csv_output:
        return txt_seeds.pop(0)
    else:
        return get_seed(client)

def main():
    # Uses subprocess to run the slurm command squeue --me, to check users running jobs,
    # and determine how many new jobs can be started
    try:
        squeue_result = subprocess.run(
                ["squeue", "--me"],
                capture_output=True,
                text=True
        ).stdout.strip().splitlines()
    except FileNotFoundError as e:
        sys.exit("Error: squeue could not be found. Is slurm installed on the machine?")
    running_jobs = len(squeue_result[1:])
    loop_range = MAX_JOBS - running_jobs

    # Starts the maximum amount of jobs at a time, until all seeds are used from the DB.
    # Uses subprocess to run the batch script job.sh with relevant arguemnts.
    print(f"{running_jobs} running jobs. Starting {loop_range}")
    for _ in range(loop_range):
        seed = new_seed()
        if seed is None:
            exit(0)

        if args.csv_output:
            training_id = "local_csv"
        else:
            training_id = initialize_training(client, seed, temp_config_id, args.uid)

        subprocess.run(['sbatch',str(PROJECT_ROOT / 'experiments' / 'job.sh'), str(seed), str(training_id), args.csv_output or "", args.dist_output or ""],
            cwd=PROJECT_ROOT,
            check=True,
        )

while True:
    main()
    time.sleep(SECONDS_BEFORE_RETRY)

