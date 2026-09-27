<#
The neural pass on top of a finished tabular v3 run: fine-tune the retrieval encoder, embed,
dense-search, re-union, re-prune with --expand, train + run the cross-encoder judge, restack,
tune, score, resolve.

    .\tools\run_v3_neural.ps1                    # everything, from the top
    .\tools\run_v3_neural.ps1 -From embed        # resume at a stage (all stages are resumable)

Assumes prep and lexical (fit / holdout / test) already exist in $WorkDir. Every stage that
must be recomputed because retrieval changed is run with --force; the previous tabular-only
results are backed up in work3_lexonly_backup / output3_lexonly_backup.

Run it in a plain PowerShell window. Stops at the first failing stage; prints a stage banner
that the log parser in work3_run.log also understands.
#>
param(
    [string]$DataDir = "..\..\student_resource\dataset",
    [string]$WorkDir = "..\..\work3",
    [string]$Out     = "..\..\output4",
    [string]$Python  = ".\.venv\Scripts\python.exe",
    [string]$Tag     = "e5s-ft",
    [string]$From    = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"

# (label, stage, split, extra args)
$plan = @(
    @("train-bi",     "train-bi",    "fit",     @()),
    @("embed fit",    "embed",       "fit",     @("--emb-tag", $Tag)),
    @("embed holdout","embed",       "holdout", @("--emb-tag", $Tag)),
    @("embed test",   "embed",       "test",    @("--emb-tag", $Tag)),
    @("dense fit",    "dense",       "fit",     @("--emb-tag", $Tag, "--force")),
    @("dense holdout","dense",       "holdout", @("--emb-tag", $Tag, "--force")),
    @("dense test",   "dense",       "test",    @("--emb-tag", $Tag, "--force")),
    @("union fit",    "union",       "fit",     @("--emb-tag", $Tag, "--force")),
    @("union holdout","union",       "holdout", @("--emb-tag", $Tag, "--force")),
    @("union test",   "union",       "test",    @("--emb-tag", $Tag, "--force")),
    @("diag",         "diag",        "holdout", @()),
    @("train-prune",  "train-prune", "fit",     @()),
    @("prune fit",    "prune",       "fit",     @("--expand", "--force")),
    @("prune holdout","prune",       "holdout", @("--expand", "--force")),
    @("prune test",   "prune",       "test",    @("--expand", "--force")),
    @("train-ce",     "train-ce",    "fit",     @()),
    @("ce fit",       "ce",          "fit",     @("--force")),
    @("ce holdout",   "ce",          "holdout", @("--force")),
    @("ce test",      "ce",          "test",    @("--force")),
    @("train-stack",  "train-stack", "fit",     @()),
    @("stack holdout","stack",       "holdout", @("--force")),
    @("stack test",   "stack",       "test",    @("--force")),
    @("tune",         "tune",        "holdout", @()),
    @("score",        "score",       "holdout", @()),
    @("errors",       "errors",      "holdout", @()),
    @("resolve",      "resolve",     "test",    @("--out", $Out))
)

$started = ($From -eq "")
foreach ($step in $plan) {
    $label = $step[0]; $stage = $step[1]; $split = $step[2]; $extra = $step[3]
    if (-not $started) {
        if ($label -eq $From -or $stage -eq $From) { $started = $true } else { continue }
    }
    $argv = @("-m", "src.v3.run", $stage, "--split", $split,
              "--data-dir", $DataDir, "--work-dir", $WorkDir) + $extra
    Write-Host ""
    Write-Host "=== $label ===  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    & $Python $argv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED at '$label', exit $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
Write-Host "neural pass complete $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
