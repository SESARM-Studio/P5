from DBScripts.database_seed import get_seed
from DBScripts.database_api import setup_supabase_client, initialize_training

import subprocess
import time
import argparse
from pathlib import Path
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("--uid", required=True)
args = parser.parse_args()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

temp_config_id = 'e4fecdcf-76a0-4e29-ae21-852decdd4513'
client = setup_supabase_client()

def main():
    squeue_result = subprocess.run(
            ["squeue", "--me"],
            capture_output=True,
            text=True
    ).stdout.strip().splitlines()
    running_jobs = len(squeue_result[1:])
    loop_range = 8 - running_jobs

    print(f"{running_jobs} running jobs. Starting {loop_range}")

    for _ in range(loop_range):
        seed = get_seed(client)
        if seed is None:
            exit(0)

        training_id = initialize_training(client, seed, temp_config_id, args.uid)
        subprocess.run(['sbatch','/ceph/project/p5_intelligent_seeds/P5/experiment0/job.sh', str(seed), str(training_id)],
            cwd=PROJECT_ROOT,
            check=True,
        )

while True:
    main()
    time.sleep(30)
