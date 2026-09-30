# ==========================================
# QuantumHPC Cleanup Snapshot Builder
# ==========================================

Write-Host ""
Write-Host "Building QuantumHPC Cleanup Snapshot..." -ForegroundColor Cyan

# Current project folder
$Project = Get-Location

# Output names
$Release = Join-Path $Project "Release"
$Zip = Join-Path $Project "QuantumHPC_cleanup_snapshot.zip"

# ------------------------------------------------------------
# Remove previous generated output
# ------------------------------------------------------------

if (Test-Path $Release) {
    Remove-Item $Release -Recurse -Force
}

if (Test-Path $Zip) {
    Remove-Item $Zip -Force
}

# ------------------------------------------------------------
# Create snapshot folder
# ------------------------------------------------------------

New-Item -ItemType Directory -Path $Release | Out-Null

Write-Host "Copying files..." -ForegroundColor Yellow

# Copy project while excluding machine-specific / unnecessary items
Get-ChildItem $Project -Force | Where-Object {
    $_.Name -notin @(
        ".venv",
        ".venv-gpu",
        ".venv-qsim",
        "__pycache__",
        ".git",
        ".vscode",
        ".pytest_cache",
        "Release",
        "Progress.txt",
        "QuantumHPC.zip",
        "QuantumHPC_cleanup_snapshot.zip"
    )
} | ForEach-Object {

    Copy-Item $_.FullName -Destination $Release -Recurse -Force

    Write-Host "  Copied: $($_.Name)"
}

# ------------------------------------------------------------
# Remove Python cache directories anywhere in snapshot
# ------------------------------------------------------------

Get-ChildItem $Release -Directory -Recurse -Force |
Where-Object { $_.Name -eq "__pycache__" } |
Remove-Item -Recurse -Force

# ------------------------------------------------------------
# Remove local raw NinaPro data
# ------------------------------------------------------------

$NinaPro = Join-Path $Release "data\ninapro"

if (Test-Path $NinaPro) {
    Remove-Item $NinaPro -Recurse -Force
    Write-Host "  Removed: data\ninapro"
}

# ------------------------------------------------------------
# Record Git information without copying the .git directory
# ------------------------------------------------------------

$SnapshotInfo = Join-Path $Release "LOCAL_SNAPSHOT_INFO.txt"

"QuantumHPC Local Cleanup Snapshot" | Out-File $SnapshotInfo
"Created: $(Get-Date)" | Out-File $SnapshotInfo -Append
"" | Out-File $SnapshotInfo -Append

"=== Git Branch ===" | Out-File $SnapshotInfo -Append
git branch --show-current 2>&1 | Out-File $SnapshotInfo -Append

"" | Out-File $SnapshotInfo -Append
"=== Latest Commit ===" | Out-File $SnapshotInfo -Append
git log -1 --oneline 2>&1 | Out-File $SnapshotInfo -Append

"" | Out-File $SnapshotInfo -Append
"=== Git Status ===" | Out-File $SnapshotInfo -Append
git status --short 2>&1 | Out-File $SnapshotInfo -Append

"" | Out-File $SnapshotInfo -Append
"=== Git Remotes ===" | Out-File $SnapshotInfo -Append
git remote -v 2>&1 | Out-File $SnapshotInfo -Append

# ------------------------------------------------------------
# Create ZIP
# ------------------------------------------------------------

Write-Host ""
Write-Host "Creating ZIP..." -ForegroundColor Yellow

Compress-Archive `
    -Path "$Release\*" `
    -DestinationPath $Zip `
    -Force

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
Write-Host ""
Write-Host "Created:"
Write-Host "  $Zip"
Write-Host ""