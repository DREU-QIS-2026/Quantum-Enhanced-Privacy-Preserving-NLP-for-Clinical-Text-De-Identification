# ==========================================
# QuantumHPC Release Builder
# ==========================================

Write-Host ""
Write-Host "Building QuantumHPC Release..." -ForegroundColor Cyan

# Current project folder
$Project = Get-Location

# Release folder
$Release = Join-Path $Project "Release"

# Remove old Release folder if it exists
if (Test-Path $Release) {
    Remove-Item $Release -Recurse -Force
}

# Create Release folder
New-Item -ItemType Directory -Path $Release | Out-Null

Write-Host "Copying files..." -ForegroundColor Yellow

# Copy everything except excluded folders/files
Get-ChildItem $Project -Force | Where-Object {
    $_.Name -notin @(
        ".venv",
        ".venv-gpu",
        ".venv-qsim",
        "__pycache__",
        ".git",
        ".vscode",
        "Release",
        "Progress.txt"
    )
} | ForEach-Object {

    Copy-Item $_.FullName -Destination $Release -Recurse -Force

    Write-Host "  Copied: $($_.Name)"
}

# Remove all __pycache__ folders from the Release copy
Get-ChildItem $Release -Directory -Recurse |
Where-Object { $_.Name -eq "__pycache__" } |
Remove-Item -Recurse -Force

# Remove local NinaPro dataset
$NinaPro = Join-Path $Release "data\ninapro"

if (Test-Path $NinaPro) {
    Remove-Item $NinaPro -Recurse -Force
    Write-Host "  Removed: data\ninapro"
}

Write-Host ""
Write-Host "Creating ZIP..." -ForegroundColor Yellow

Compress-Archive `
    -Path "$Release\*" `
    -DestinationPath "$Project\QuantumHPC.zip" `
    -Force

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Write-Host ""
Write-Host "Created:"
Write-Host "  QuantumHPC.zip"
Write-Host ""