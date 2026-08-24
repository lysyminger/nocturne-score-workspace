$ErrorActionPreference = "Stop"

$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonLauncher = "py"
$PaddlePython = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$PaddleSource = Join-Path $TrainingRoot "vendor\PaddleOCR"
$DetectionSource = Join-Path $TrainingRoot "vendor\PaddleDetection"

if (-not (Test-Path $PaddlePython)) {
    & $PythonLauncher -3.11 -m venv (Join-Path $TrainingRoot ".venv-paddle")
}
& $PaddlePython -m pip install --upgrade pip
& $PaddlePython -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
& $PaddlePython -m pip install paddleocr==3.7.0

if (-not (Test-Path $PaddleSource)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PaddleSource) | Out-Null
    git clone --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git $PaddleSource
}
git -C $PaddleSource fetch --depth 1 origin tag v3.7.0
git -C $PaddleSource checkout --detach v3.7.0
& $PaddlePython -m pip install -r (Join-Path $PaddleSource "requirements.txt")

if (-not (Test-Path $DetectionSource)) {
    git clone --depth 1 --branch v2.9.0 https://github.com/PaddlePaddle/PaddleDetection.git $DetectionSource
}
git -C $DetectionSource fetch --depth 1 origin tag v2.9.0
git -C $DetectionSource checkout --detach v2.9.0
& $PaddlePython -m pip install -r (Join-Path $DetectionSource "requirements.txt")
& $PaddlePython -m pip install --force-reinstall --no-deps `
    opencv-python==4.10.0.84 `
    opencv-python-headless==4.10.0.84 `
    opencv-contrib-python==4.10.0.84
& $PaddlePython -m pip check

Write-Host "Training environment installed. Run training\verify.ps1 next."
