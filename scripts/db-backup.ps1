param(
    [string]$OutputDir = "./backups",
    [string]$Service = "mysql",
    [string]$Database = "health_assistant",
    [string]$User = "root"
)

$ErrorActionPreference = "Stop"
$resolved = [System.IO.Path]::GetFullPath($OutputDir)
New-Item -ItemType Directory -Force -Path $resolved | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$target = Join-Path $resolved "$Database-$stamp.sql"

Write-Host "Creating MySQL backup: $target"
docker compose exec -T $Service mysqldump --single-transaction --routines --triggers --hex-blob -u $User -p$env:MYSQL_ROOT_PASSWORD $Database | Out-File -FilePath $target -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "mysqldump failed with exit code $LASTEXITCODE" }
Write-Host "Backup complete."
