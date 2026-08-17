param(
    [string]$OutputPath = ".\backup\mysql\scms.sql"
)

$ErrorActionPreference = "Stop"

foreach ($name in @("MYSQL_HOST", "MYSQL_USERNAME", "MYSQL_PASSWORD")) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name))) {
        throw "Environment variable $name is required"
    }
}

$database = if ($env:MYSQL_DATABASE) { $env:MYSQL_DATABASE } else { "scms" }
$port = if ($env:MYSQL_PORT) { $env:MYSQL_PORT } else { "3306" }
$parent = Split-Path -Parent $OutputPath
if ($parent) {
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
}

$defaultsFile = Join-Path $env:TEMP ("mysql-client-" + [guid]::NewGuid() + ".cnf")
try {
    @"
[client]
user=$env:MYSQL_USERNAME
password=$env:MYSQL_PASSWORD
host=$env:MYSQL_HOST
port=$port
"@ | Set-Content -LiteralPath $defaultsFile -Encoding utf8

    & mysqldump `
        "--defaults-extra-file=$defaultsFile" `
        --single-transaction --routines --triggers --events `
        --set-gtid-purged=OFF --default-character-set=utf8mb4 `
        --databases $database `
        --result-file=$OutputPath
    if ($LASTEXITCODE -ne 0) {
        throw "mysqldump failed with exit code $LASTEXITCODE"
    }
    Write-Host "MySQL export written to $OutputPath"
}
finally {
    Remove-Item -LiteralPath $defaultsFile -Force -ErrorAction SilentlyContinue
}
