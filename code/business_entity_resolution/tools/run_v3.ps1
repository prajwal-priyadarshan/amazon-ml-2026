<#
Runs the v3 pipeline stage by stage, stopping at the first failure.

    .\tools\run_v3.ps1                  # tabular stages only (no PyTorch needed)
    .\tools\run_v3.ps1 -WithNeural      # adds embed / dense / train-bi / train-ce / ce
    .\tools\run_v3.ps1 -From prune      # resume at a stage (everything is resumable anyway)

Every stage skips shards it has already written, so re-running after a crash is safe and
costs only the shard that died. Long passes belong in a plain PowerShell window: Claude
Code's background-shell reaper has killed multi-hour runs on this machine.
#>
param(
    [string]$DataDir  = "..\..\student_resource\dataset",
    [string]$WorkDir  = "..\..\work3",
    [string]$Out      = "..\..\output3",
    [string]$Python   = ".\.venv\Scripts\python.exe",
    [switch]$WithNeural,
    [switch]$Expand,
    [string]$From     = "",
    [int]$Jobs        = 8
)

$ErrorActionPreference = "Stop"

# (stage, split, extra args, neural?)
$plan = @(
    @("prep",        "all",     @("--jobs", "$Jobs"),                     $false),
    @("lexical",     "fit",     @(),                                      $false),
    @("lexical",     "holdout", @(),                                      $false),
    @("lexical",     "test",    @(),                                      $false),
    @("embed",       "fit",     @(),                                      $true),
    @("embed",       "holdout", @(),                                      $true),
    @("embed",       "test",    @(),                                      $true),
    @("dense",       "fit",     @(),                                      $true),
    @("dense",       "holdout", @(),                                      $true),
    @("dense",       "test",    @(),                                      $true),
    @("union",       "fit",     @(),                                      $false),
    @("union",       "holdout", @(),                                      $false),
    @("union",       "test",    @(),                                      $false),
    @("diag",        "holdout", @(),                                      $false),
    @("train-prune", "fit",     @(),                                      $false),
    @("prune",       "fit",     @(),                                      $false),
    @("prune",       "holdout", @(),                                      $false),
    @("prune",       "test",    @(),                                      $false),
    @("train-ce",    "fit",     @(),                                      $true),
    @("ce",          "fit",     @(),                                      $true),
    @("ce",          "holdout", @(),                                      $true),
    @("ce",          "test",    @(),                                      $true),
    @("train-stack", "fit",     @(),                                      $false),
    @("stack",       "holdout", @(),                                      $false),
    @("stack",       "test",    @(),                                      $false),
    @("tune",        "holdout", @(),                                      $false),
    @("score",       "holdout", @(),                                      $false),
    @("errors",      "holdout", @(),                                      $false),
    @("resolve",     "test",    @("--out", $Out),                         $false)
)

$started = ($From -eq "")
foreach ($step in $plan) {
    $stage = $step[0]; $split = $step[1]; $extra = $step[2]; $neural = $step[3]
    if (-not $started) {
        if ($stage -eq $From) { $started = $true } else { continue }
    }
    if ($neural -and -not $WithNeural) {
        Write-Host "-- skipping $stage ($split): neural stage, pass -WithNeural to include it"
        continue
    }
    $argv = @("-m", "src.v3.run", $stage, "--split", $split,
              "--data-dir", $DataDir, "--work-dir", $WorkDir) + $extra
    if ($Expand -and $stage -eq "prune") { $argv += "--expand" }
    Write-Host ""
    Write-Host "=== $stage ($split) ===" -ForegroundColor Cyan
    & $Python $argv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at stage '$stage' (split $split), exit $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host ""
Write-Host "=== validate ===" -ForegroundColor Cyan
& $Python "..\..\student_resource\utils\validate_submission.py" `
    --matching (Join-Path $Out "matching_results.tsv") `
    --candidate (Join-Path $Out "candidate_pairs.tsv") `
    --test-dir (Join-Path $DataDir "test")
