$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$DetectionRoot = Join-Path $TrainingRoot "vendor\PaddleDetection"
$ConfigPath = Join-Path $TrainingRoot "configs\pdf_fret_finetune.yml"
$Output = Join-Path $TrainingRoot "models\tab-event-inference"

Push-Location $DetectionRoot
try {
    & $Python tools\export_model.py -c $ConfigPath --output_dir $Output -o `
        "weights=../../runs/pdf-fret-detector/best_model"
    if ($LASTEXITCODE -ne 0) { throw "TAB event detector export failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
