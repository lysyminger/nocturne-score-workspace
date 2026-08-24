param(
    [int]$Epochs = 18,
    [int]$BatchSize = 4
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$DetectionRoot = Join-Path $TrainingRoot "vendor\PaddleDetection"
$ConfigPath = Join-Path $TrainingRoot "configs\pdf_fret_location.yml"

& $Python (Join-Path $TrainingRoot "scripts\build_location_annotations.py") --dataset (Join-Path $TrainingRoot "data\pdf-frets")
Push-Location $DetectionRoot
try {
    & $Python tools\train.py -c $ConfigPath --eval --amp -o `
        "epoch=$Epochs" `
        "TrainReader.batch_size=$BatchSize" `
        "pretrain_weights=../../runs/tab-event-detector/best_model"
    if ($LASTEXITCODE -ne 0) { throw "Fret location training failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
