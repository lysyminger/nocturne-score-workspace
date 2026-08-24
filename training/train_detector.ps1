param(
    [int]$Epochs = 40,
    [int]$BatchSize = 2
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$DetectionRoot = Join-Path $TrainingRoot "vendor\PaddleDetection"
$DatasetRoot = Join-Path $TrainingRoot "data\symbols"
$ConfigPath = Join-Path $TrainingRoot "configs\symbol_detector.yml"

if (-not (Test-Path $Python)) {
    throw "Training environment is missing. Run training\install.ps1 first."
}
if (-not (Test-Path $DetectionRoot)) {
    throw "PaddleDetection source is missing. Run training\install.ps1 first."
}

& $Python (Join-Path $TrainingRoot "scripts\validate_coco.py") --dataset $DatasetRoot
Push-Location $DetectionRoot
try {
    & $Python tools\train.py -c $ConfigPath --eval --amp -o "epoch=$Epochs" "TrainReader.batch_size=$BatchSize"
    if ($LASTEXITCODE -ne 0) { throw "PaddleDetection training failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
