$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "docker not found."
}

$dockerInfo = docker info --format '{{.ServerVersion}}' 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker Desktop engine is not running; there are no Docker services to stop." -ForegroundColor Yellow
    exit 0
}

docker compose stop backend mysql redis minio
if ($LASTEXITCODE -ne 0) { throw "Failed to stop Docker services." }
Write-Host "Docker services stopped. Local backend/frontend windows can be closed separately." -ForegroundColor Green
