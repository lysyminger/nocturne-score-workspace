param(
    [int]$Epochs = 8,
    [int]$BatchSize = 256,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $TrainingRoot
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$PaddleOcr = Join-Path $TrainingRoot "vendor\PaddleOCR"
$Config = Join-Path $TrainingRoot "configs\fret_ocr.yml"
$TrainLabels = Join-Path $TrainingRoot "data\fret-ocr\train.txt"
$LatestCheckpoint = Join-Path $TrainingRoot "runs\fret-ocr\latest"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "训练环境不存在，请先运行 training\install.ps1"
}
if (-not (Test-Path -LiteralPath $TrainLabels)) {
    throw "品位数字数据集不存在，请先运行 build_gp_corpus.mjs 和 extract_fret_crops.py"
}
Push-Location $PaddleOcr
try {
    $Overrides = @(
        "Global.epoch_num=$Epochs",
        "Train.loader.batch_size_per_card=$BatchSize"
    )
    if ($Resume) {
        if (-not (Test-Path -LiteralPath ($LatestCheckpoint + ".pdparams"))) {
            throw "找不到续训检查点: $LatestCheckpoint.pdparams"
        }
        $Overrides += "Global.checkpoints=$LatestCheckpoint"
    }
    & $Python tools\train.py -c $Config -o $Overrides
    if ($LASTEXITCODE -ne 0) {
        throw "PaddleOCR 训练失败，退出码 $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
