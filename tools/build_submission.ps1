<#
Builds the official <team_name>_submission.zip described in SUBMISSION.md.

Since the Sep 2026 restructure, the repo root itself already matches the required layout
(output/, code/business_entity_resolution/, Documentation_template.md), so this script just
zips those three things straight from the root -- no staging copy, nothing else included.

    .\tools\build_submission.ps1
    .\tools\build_submission.ps1 -SkipValidate

Never run it while a pipeline stage is running: the validator needs ~8 GB of RAM.
#>
param(
    [string]$Python     = ".\.venv\Scripts\python.exe",
    [string]$TestDir    = ".\student_resource\dataset\test",
    [switch]$SkipValidate
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$matching = ".\output\matching_results.tsv"
$candDst  = ".\output\candidate_pairs.tsv"
foreach ($required in @($matching, $candDst, ".\code\business_entity_resolution\src",
                         ".\code\business_entity_resolution\README.md",
                         ".\code\business_entity_resolution\requirements.txt",
                         ".\Documentation_template.md")) {
    if (-not (Test-Path $required)) { throw "missing required submission path: $required" }
}

$repoName = Split-Path -Leaf (Get-Location)
$safe = ($repoName -replace '[^\w\-]', '_')
New-Item -ItemType Directory -Force .\submission | Out-Null
$zip = ".\submission\${safe}_submission.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }

# Not Compress-Archive: on Windows PowerShell 5.1 it writes backslash entry names, which
# Linux unzip tools extract as flat files instead of folders.
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$root = (Resolve-Path ".").Path.TrimEnd('\') + '\'
$archive = [IO.Compression.ZipFile]::Open((Join-Path $PWD $zip), 'Create')
try {
    $items = Get-ChildItem .\output -Recurse -File
    $items += Get-ChildItem .\code -Recurse -File | Where-Object { $_.FullName -notmatch '\\__pycache__\\' }
    $items += Get-Item .\Documentation_template.md
    foreach ($f in $items) {
        $name = $f.FullName.Substring($root.Length).Replace('\', '/')
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $f.FullName, $name, 'Optimal')
    }
} finally { $archive.Dispose() }
Write-Host "built $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB)"

# validate
if (-not $SkipValidate) {
    & $Python .\tools\validate_submission.py --matching $matching --candidate $candDst --test-dir $TestDir
    if ($LASTEXITCODE -ne 0) { throw "validator FAILED - do not upload $zip" }
}
