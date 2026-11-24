# P5

**VIGTIGT! Før pip skal der altid skrives uv**

## Setup:
### 1. Download uv:
**Mac / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
**Windwos (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
``
### 2. Restart terminal - check uv is installed:
``uv --version``

### 3. Create virtual environment (from root dir):

```bash
uv venv
```

### 4. Activate virtual environment (from root dir):
**Mac / Linux**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```
### 5. Install packages

```bash
uv pip install -r requirements.txt
```

## How to train single model on AI LAB:
```bash
cd /ceph/project/p5_intelligent_seeds/P5/experiments
```
**Change parameters in job.sh** 
```bash
sbatch job.sh <seed> <number of epoch> <training id> <experiment number> <optional: CSV output> <optional: distribution output>
```

## How to run a experiment on AI LAB:
```bash
screen -S exp
cd /ceph/project/p5_intelligent_seeds/P5/
.venv-login/bin/python -m experiments.run_experiment --uid <Your name> --exp-no <Experiment number>
```
To run the experiment local, disconnected from the database, use the argument ```--csv-output <path_to_file>```
and specify an output location, and ````--seed-file <path_to_file.txt>```` to specify which seeds to run, seperated by newlines
(default is seeds.txt in same directory as run_experiment.py)

***Afterwards: Click Ctrl + A + D***
Should return something like: [detached from 2341069.exp0]
