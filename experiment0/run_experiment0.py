from DBScripts.database_seed import get_seed
from DBScripts.database_api import setup_supabase_client, initialize_training

import subprocess
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--uid", required=True)
args = parser.parse_args()

temp_config_id = 'e4fecdcf-76a0-4e29-ae21-852decdd4513'
client = setup_supabase_client()

def main():
    squeue_result = subprocess.run(
            ["squeue", "--me"],
            capture_output=True,
            text=True
    ).stdout.strip().splitlines()
    running_jobs = squeue_result[1:]

    for _ in range(8-running_jobs):
        seed = get_seed(client)
        if seed is None:
            exit(0)

        training_id = initialize_training(client, seed, temp_config_id, args.uid)
        subprocess.check_call(['sbatch','/ceph/project/p5_intelligent_seeds/P5/experiment0/job.sh', str(seed), str(training_id)])

while True:
    main()
    time.sleep(60)