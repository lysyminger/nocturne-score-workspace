$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

& (Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe") (Join-Path $TrainingRoot "scripts\verify_paddle_gpu.py")
& (Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe") (Join-Path $TrainingRoot "scripts\validate_install.py")
