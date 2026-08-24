param(
    [ValidateSet("val", "test")]
    [string]$Split = "test"
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$DetectionRoot = Join-Path $TrainingRoot "vendor\PaddleDetection"
$ConfigPath = Join-Path $TrainingRoot "configs\pdf_fret_finetune.yml"
$Checkpoint = "../../runs/pdf-fret-detector/best_model"

Push-Location $DetectionRoot
try {
    & $Python tools\eval.py -c $ConfigPath -o `
        "weights=$Checkpoint" `
        "EvalDataset.image_dir=images/$Split" `
        "EvalDataset.anno_path=annotations/$Split.json"
    if ($LASTEXITCODE -ne 0) { throw "PDF fret detector evaluation failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
