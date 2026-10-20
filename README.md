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