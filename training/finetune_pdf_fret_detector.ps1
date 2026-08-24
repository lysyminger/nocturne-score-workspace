param(
    [int]$Epochs = 18,
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$DetectionRoot = Join-Path $TrainingRoot "vendor\PaddleDetection"
$DatasetRoot = Join-Path $TrainingRoot "data\pdf-frets"
$Classes = Join-Path $TrainingRoot "configs\tab_event_classes.yaml"
$ConfigPath = Join-Path $TrainingRoot "configs\pdf_fret_finetune.yml"

& $Python (Join-Path $TrainingRoot "scripts\validate_coco.py") --dataset $DatasetRoot --classes $Classes
Push-Location $DetectionRoot
try {
    & $Python tools\train.py -c $ConfigPath --eval --amp -o `
        "epoch=$Epochs" `
        "TrainReader.batch_size=$BatchSize" `
        "pretrain_weights=../../runs/tab-event-detector/best_model"
    if ($LASTEXITCODE -ne 0) { throw "PDF fret detector fine-tuning failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
