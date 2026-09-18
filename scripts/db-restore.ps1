param(
    [Parameter(Mandatory=$true)][string]$InputFile,
    [string]$Service = "mysql",
    [string]$Database = "health_assistant",
    [string]$User = "root"
)

$ErrorActionPreference = "Stop"
$resolved = [System.IO.Path]::GetFullPath($InputFile)
if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { throw "Backup file not found: $resolved" }
Write-Host "Restoring $resolved into '$Database'. Existing rows may be overwritten."
Get-Content -LiteralPath $resolved -Raw | docker compose exec -T $Service mysql -u $User -p$env:MYSQL_ROOT_PASSWORD $Database
if ($LASTEXITCODE -ne 0) { throw "mysql restore failed with exit code $LASTEXITCODE" }
Write-Host "Restore complete."
