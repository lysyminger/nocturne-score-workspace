$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$PaddleOcr = Join-Path $TrainingRoot "vendor\PaddleOCR"
$Config = Join-Path $TrainingRoot "configs\fret_ocr.yml"
$Checkpoint = Join-Path $TrainingRoot "runs\fret-ocr\best_accuracy"
$Output = Join-Path $TrainingRoot "models\fret-ocr-inference"

if (-not (Test-Path -LiteralPath ($Checkpoint + ".pdparams"))) {
    throw "找不到最佳模型: $Checkpoint.pdparams"
}

Push-Location $PaddleOcr
try {
    & $Python tools\export_model.py -c $Config -o `
        "Global.pretrained_model=$Checkpoint" `
        "Global.save_inference_dir=$Output"
    if ($LASTEXITCODE -ne 0) {
        throw "PaddleOCR 导出失败，退出码 $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
