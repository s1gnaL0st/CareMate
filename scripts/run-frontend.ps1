$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$frontendRoot = Join-Path $projectRoot "frontend"
Set-Location $frontendRoot
$env:npm_config_cache = Join-Path $frontendRoot ".npm-cache"

if (-not (Get-Command npm.cmd -ErrorAction SilentlyContinue)) {
    throw "npm not found. Install Node.js first, then run .\start.ps1 again."
}

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies (first run only)..." -ForegroundColor Cyan
    npm.cmd install
    if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
}

Write-Host "Starting Vite at http://localhost:5173 ..." -ForegroundColor Green
npm.cmd run dev -- --host 127.0.0.1
