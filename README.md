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
```
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

## How to train single model in experiment0 on AI LAB:
```bash
cd /ceph/project/p5_intelligent_seeds/P5/experiment0
```
**Change parameters in job.sh** 
```bash
sbatch job.sh
```

## How to run Experiment 0 on AI LAB:
```bash
screen -S exp0
cd /ceph/project/p5_intelligent_seeds/P5/
.venv-login/bin/python -m experiment0.run_experiment0 --uid <Your name>
```
***Afterwards: Click Ctrl + A + D***
Should return something like: [detached from 2341069.exp0]
