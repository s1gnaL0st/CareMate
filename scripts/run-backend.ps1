$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$backendRoot = Join-Path $projectRoot "backend"
Set-Location $backendRoot

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv not found. Install uv first, then run .\start.ps1 again."
}

# The Compose containers are exposed on localhost ports for local development.
$env:DATABASE_URL = "mysql+asyncmy://health:health@127.0.0.1:3307/health_assistant"
$env:REDIS_URL = "redis://127.0.0.1:6380/0"
$env:CORS_ORIGINS = "http://localhost:5173,http://localhost:5174"

uv run python -c "from agents.llm import resolve_model_settings; s=resolve_model_settings(); print('LLM:', s.provider, '| model:', s.model, '| base_url:', s.base_url)"
if ($LASTEXITCODE -ne 0) { throw "LLM configuration is invalid. Check backend/.env (API key, model and base URL)." }

Write-Host "Running database migrations..." -ForegroundColor Cyan
$migrationSucceeded = $false
for ($attempt = 1; $attempt -le 30; $attempt++) {
    uv run alembic upgrade head
    if ($LASTEXITCODE -eq 0) {
        $migrationSucceeded = $true
        break
    }
    Write-Host "Database is not ready yet (attempt $attempt/30); retrying..." -ForegroundColor Yellow
    Start-Sleep -Seconds 2
}
if (-not $migrationSucceeded) { throw "Database migration failed after waiting. Is Docker Desktop running?" }

Write-Host "Starting FastAPI at http://localhost:8000 ..." -ForegroundColor Green
uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000
