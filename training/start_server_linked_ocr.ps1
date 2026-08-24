param(
    [string]$ServerHost = "192.168.2.131",
    [string]$ServerUser = "lysyminger",
    [ValidateRange(1024, 65535)]
    [int]$Port = 8892,
    [ValidateSet("gpu", "cpu")]
    [string]$Device = "gpu"
)

$ErrorActionPreference = "Stop"
$TrainingRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $TrainingRoot
$Python = Join-Path $TrainingRoot ".venv-paddle\Scripts\python.exe"
$Service = Join-Path $TrainingRoot "scripts\serve_fret_ocr.py"
$Model = Join-Path $TrainingRoot "models\fret-ocr-inference"
$RunDirectory = Join-Path $TrainingRoot "runs\server-linked-ocr"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "找不到训练环境 Python：$Python"
}
if (-not (Test-Path -LiteralPath (Join-Path $Model "inference.json") -PathType Leaf)) {
    throw "找不到已导出的品位 OCR 模型：$Model"
}

New-Item -ItemType Directory -Force -Path $RunDirectory | Out-Null
$ServiceLog = Join-Path $RunDirectory "service.log"
$TunnelLog = Join-Path $RunDirectory "tunnel.log"

try {
    Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 | Out-Null
}
catch {
    Start-Process -FilePath $Python -ArgumentList @(
        $Service,
        "--model", $Model,
        "--host", "127.0.0.1",
        "--port", "$Port",
        "--device", $Device
    ) -WindowStyle Hidden -RedirectStandardOutput $ServiceLog -RedirectStandardError "$ServiceLog.err"
    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 60; $Attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2 | Out-Null
            $Ready = $true
            break
        }
        catch { }
    }
    if (-not $Ready) {
        throw "本地 OCR 服务未能在 30 秒内启动，请查看 $ServiceLog.err"
    }
}

$ExistingTunnel = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "ssh.exe" -and $_.CommandLine -like "*127.0.0.1:$Port`:127.0.0.1:$Port*"
}
if (-not $ExistingTunnel) {
    Start-Process -FilePath "ssh" -ArgumentList @(
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-N",
        "-R", "127.0.0.1:$Port`:127.0.0.1:$Port",
        "$ServerUser@$ServerHost"
    ) -WindowStyle Hidden -RedirectStandardOutput $TunnelLog -RedirectStandardError "$TunnelLog.err"
    Start-Sleep -Seconds 2
}

ssh "$ServerUser@$ServerHost" "curl -fsS http://127.0.0.1:$Port/health"
