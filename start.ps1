param(
    [switch]$DockerBackend
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Require-Command([string]$name, [string]$hint) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "$name not found. $hint"
    }
}

Require-Command "docker" "Install Docker Desktop and start it first."

# Fail fast when the Docker CLI is installed but the Desktop engine is not
# running. Without this check `docker compose` can block for a long time on a
# missing named pipe and the rest of the development stack is never started.
$dockerInfo = docker info --format '{{.ServerVersion}}' 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Docker Desktop engine is not running. Start Docker Desktop, wait until it is ready, then run .\start.ps1 again. Details: $dockerInfo"
}

if (-not (Test-Path "backend/.env")) {
    throw "backend/.env is missing. Copy backend/.env.example to backend/.env and fill in your relay settings first."
}

Write-Host "[1/3] Starting MySQL, Redis and MinIO..." -ForegroundColor Cyan
docker compose up -d mysql redis minio
if ($LASTEXITCODE -ne 0) { throw "Docker infrastructure failed to start." }

if ($DockerBackend) {
    Write-Host "[2/3] Starting backend in Docker..." -ForegroundColor Cyan
    docker compose up -d --build backend
    if ($LASTEXITCODE -ne 0) { throw "Docker backend failed to start. Try without -DockerBackend to use the local Python environment." }
} else {
    # Avoid a port conflict if the Docker backend was started previously.
    docker compose stop backend *> $null
    Write-Host "[2/3] Opening backend development window..." -ForegroundColor Cyan
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $projectRoot "scripts/run-backend.ps1")
    )
}

Write-Host "[3/3] Opening frontend development window..." -ForegroundColor Cyan
Start-Process powershell.exe -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $projectRoot "scripts/run-frontend.ps1")
)

Write-Host "" 
Write-Host "Started. Open http://localhost:5173" -ForegroundColor Green
Write-Host "API health: http://localhost:8000/api/v1/health/ready"
Write-Host "Run .\stop.ps1 when you are done."
