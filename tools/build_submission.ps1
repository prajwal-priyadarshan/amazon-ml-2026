<#
Builds the official <team_name>_submission.zip described in SUBMISSION.md, in one go:

  1. copies candidate_pairs.tsv from the final run (output9\) next to the final matches;
  2. writes Documentation_template.md from docs\methodology.md with team name/members filled in;
  3. stages the exact folder layout the organizers ask for and zips it into submission\;
  4. runs the submission validator on the result.

    .\tools\build_submission.ps1 -TeamName "MyTeam" -Members "Alice, Bob"
    .\tools\build_submission.ps1 -TeamName "MyTeam" -Members "Alice, Bob" -SkipValidate

Never run it while a pipeline stage is running: the validator needs ~8 GB of RAM.
#>
param(
    [Parameter(Mandatory = $true)][string]$TeamName,
    [Parameter(Mandatory = $true)][string]$Members,
    [string]$Candidates = ".\output9\candidate_pairs.tsv",
    [string]$Python     = ".\.venv\Scripts\python.exe",
    [string]$TestDir    = ".\student_resource\dataset\test",
    [switch]$SkipValidate
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

$final    = ".\results\final_submission"
$matching = "$final\matching_results.tsv"
$candDst  = "$final\candidate_pairs.tsv"

# 1. candidate_pairs.tsv (gitignored, lives only on the machine that ran the final pass)
if (-not (Test-Path $candDst)) {
    if (-not (Test-Path $Candidates)) { throw "candidate_pairs.tsv not found at $Candidates - pass -Candidates <path>" }
    Copy-Item $Candidates $candDst
}

# 2. Documentation_template.md with the placeholders filled in
$doc = Get-Content .\docs\methodology.md -Raw -Encoding UTF8
$doc = $doc.Replace("[Your Team Name]", $TeamName).Replace("[List all team members]", $Members)
[IO.File]::WriteAllText((Join-Path $PWD "Documentation_template.md"), $doc, (New-Object Text.UTF8Encoding $false))

# 3. stage the required layout and zip it
$stage = ".\_submission"
$code  = "$stage\code\business_entity_resolution"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Force "$stage\output", $code | Out-Null

Copy-Item $matching                   "$stage\output\"
Copy-Item $candDst                    "$stage\output\"
Copy-Item .\src                       "$code\src" -Recurse
Get-ChildItem "$code\src" -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Copy-Item .\requirements-lock.txt     "$code\requirements.txt"
Copy-Item .\docs\runbook.md           "$code\README.md"
Copy-Item .\Documentation_template.md "$stage\"

$safe = ($TeamName -replace '[^\w\-]', '_')
New-Item -ItemType Directory -Force .\submission | Out-Null
$zip = ".\submission\${safe}_submission.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
# Not Compress-Archive: on Windows PowerShell 5.1 it writes backslash entry names, which
# Linux unzip tools extract as flat files instead of folders.
Add-Type -AssemblyName System.IO.Compression, System.IO.Compression.FileSystem
$root = (Resolve-Path $stage).Path.TrimEnd('\') + '\'
$archive = [IO.Compression.ZipFile]::Open((Join-Path $PWD $zip), 'Create')
try {
    foreach ($f in Get-ChildItem $stage -Recurse -File) {
        $name = $f.FullName.Substring($root.Length).Replace('\', '/')
        [void][IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $f.FullName, $name, 'Optimal')
    }
} finally { $archive.Dispose() }
Remove-Item $stage -Recurse -Force
Write-Host "built $zip ($([math]::Round((Get-Item $zip).Length / 1MB, 1)) MB)"

# 4. validate
if (-not $SkipValidate) {
    & $Python .\tools\validate_submission.py --matching $matching --candidate $candDst --test-dir $TestDir
    if ($LASTEXITCODE -ne 0) { throw "validator FAILED - do not upload $zip" }
}
