from DBScripts.testDatabase.database_seed import *
import subprocess
import time

def main():
    squeue_result = subprocess.run(
            ["squeue", "--me"],
            capture_output=True,
            text=True
    ).stdout.strip().splitlines()
    running_jobs = squeue_result[1:]

    for _ in range(8-running_jobs):
        seed = get_seed()
        if seed is None:
            exit(0)

        subprocess.check_call(['/ceph/project/p5_intelligent_seeds/P5/experiment0/job.sh', seed])

while True:
    main()
    time.sleep(60)