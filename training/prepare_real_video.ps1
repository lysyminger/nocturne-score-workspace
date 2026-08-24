param(
    [string]$Source = "training/data/real-video",
    [string]$Output = "training/data/real-video-token-review",
    [ValidateRange(1, 100)]
    [int]$Stride = 4,
    [ValidateSet("gpu", "cpu")]
    [string]$Device = "gpu",
    [int]$MaxFramesPerRun = 0
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $TrainingRoot
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$Script = Join-Path $TrainingRoot "scripts\prepare_real_video_corpus.py"
$Model = Join-Path $TrainingRoot "models\fret-ocr-inference"

Push-Location $ProjectRoot
try {
    & $Python $Script `
        --source $Source `
        --output $Output `
        --model $Model `
        --stride $Stride `
        --device $Device `
        --max-frames-per-run $MaxFramesPerRun
    if ($LASTEXITCODE -ne 0) {
        throw "真实视频训练集准备失败，退出码 $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
