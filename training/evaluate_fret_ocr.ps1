param(
    [ValidateSet("val", "test")]
    [string]$Split = "test"
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$PaddleOcr = Join-Path $TrainingRoot "vendor\PaddleOCR"
$Config = Join-Path $TrainingRoot "configs\fret_ocr.yml"
$Checkpoint = Join-Path $TrainingRoot "runs\fret-ocr\best_accuracy"
$LabelFile = "../../data/fret-ocr/$Split.txt"

if (-not (Test-Path -LiteralPath ($Checkpoint + ".pdparams"))) {
    throw "找不到最佳模型: $Checkpoint.pdparams"
}

Push-Location $PaddleOcr
try {
    & $Python tools\eval.py -c $Config -o `
        "Global.checkpoints=$Checkpoint" `
        "Eval.dataset.label_file_list=[`"$LabelFile`"]"
    if ($LASTEXITCODE -ne 0) {
        throw "PaddleOCR 评估失败，退出码 $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
