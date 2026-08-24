param(
    [int]$Epochs = 40,
    [int]$BatchSize = 2,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$DetectionRoot = Join-Path $TrainingRoot "vendor\PaddleDetection"
$DatasetRoot = Join-Path $TrainingRoot "data\tab-events"
$Classes = Join-Path $TrainingRoot "configs\tab_event_classes.yaml"
$ConfigPath = Join-Path $TrainingRoot "configs\tab_event_detector.yml"

& $Python (Join-Path $TrainingRoot "scripts\validate_coco.py") --dataset $DatasetRoot --classes $Classes
Push-Location $DetectionRoot
try {
    $Overrides = @("epoch=$Epochs", "TrainReader.batch_size=$BatchSize")
    if ($Resume) {
        $Overrides += "pretrain_weights=../../runs/tab-event-detector/best_model"
    }
    & $Python tools\train.py -c $ConfigPath --eval --amp -o $Overrides
    if ($LASTEXITCODE -ne 0) { throw "TAB event detector training failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
