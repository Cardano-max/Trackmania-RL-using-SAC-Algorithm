# Requires: Windows, TrackMania 2020 installed and configured per tmrl docs, Python 3.10, tmrl package

param(
  [string]$TmrlDataPath = "$env:USERPROFILE\TmrlData",
  [string]$ServerIp = "127.0.0.1",
  [int]$ServerPort = 5555
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $TmrlDataPath)) { New-Item -ItemType Directory -Force -Path $TmrlDataPath | Out-Null }
if (-not (Test-Path "$TmrlDataPath\config")) { New-Item -ItemType Directory -Force -Path "$TmrlDataPath\config" | Out-Null }

$configPath = Join-Path $TmrlDataPath "config\config.json"
if (-not (Test-Path $configPath)) {
  $cfg = @{ 
    PASSWORD = "change-me";
    TLS = $false;
    TLS_CREDENTIALS_DIRECTORY = "";
    SERVER = @{ IP = $ServerIp; PORT = $ServerPort };
  } | ConvertTo-Json -Depth 5
  $cfg | Set-Content -Path $configPath -Encoding UTF8
}

$env:TMRL_DATA_PATH = $TmrlDataPath

python - << 'PY'
import os
from tmrl.rollout import rollout_worker

os.environ.setdefault('TMRL_DATA_PATH', os.path.expanduser('~/TmrlData'))
rollout_worker.main()
PY

