<#
The final layer on top of a finished neural pass (tools\run_v3_neural.ps1):

  1. two extra never-trained-on S1 samples (hold2, hold3) pushed through the same
     retrieve -> prune -> judge -> stack chain, so the refine model has clean labels;
  2. a second judge epoch (continued from models\ce) scored on every split;
  3. the refine model (src\v3\refine.py): cluster / rarity / competition features plus the
     stacker's design matrix, cross-fitted on holdout+hold2+hold3, 3 seed bags;
  4. resolve with expected-F0.5 into $Out.

    .\tools\run_v3_refine.ps1
    .\tools\run_v3_refine.ps1 -From "refine-feats test"

Every stage is resumable. Run it in a plain PowerShell window, and never run the submission
validator at the same time: it needs ~8 GB of RAM on the full candidate file.
#>
param(
    [string]$DataDir = "..\..\student_resource\dataset",
    [string]$WorkDir = "..\..\work3",
    [string]$Out     = "..\..\output9",
    [string]$Python  = ".\.venv\Scripts\python.exe",
    [string]$Tag     = "e5s-ft",
    [string]$From    = ""
)

$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$R = "scored_r10"

$plan = @()
# the stacker's design matrix for the splits that already exist
foreach ($s in @("holdout", "test")) { $plan += ,@("stack $s", "stack", $s, @("--force", "--save-feats")) }
# fresh clean splits
foreach ($s in @("hold2", "hold3")) {
    $plan += ,@("prep $s",    "prep",    $s, @("--jobs", "6"))
    $plan += ,@("lexical $s", "lexical", $s, @())
    $plan += ,@("embed $s",   "embed",   $s, @("--emb-tag", $Tag))
    $plan += ,@("dense $s",   "dense",   $s, @("--emb-tag", $Tag))
    $plan += ,@("union $s",   "union",   $s, @("--emb-tag", $Tag))
    $plan += ,@("prune $s",   "prune",   $s, @("--expand"))
    $plan += ,@("ce $s",      "ce",      $s, @())
    $plan += ,@("stack $s",   "stack",   $s, @("--save-feats"))
}
# second judge epoch
$plan += ,@("train-ce2", "train-ce", "fit", @("--ce-out", "ce2", "--ce-init", "ce", "--seed", "1",
                                             "--ce-lr", "1e-5", "--ce-max-rows", "700000"))
foreach ($s in @("holdout", "hold2", "hold3", "test")) {
    $plan += ,@("ce2 $s", "ce", $s, @("--ce-out", "ce2"))
}
# refine
foreach ($s in @("holdout", "hold2", "hold3", "test")) {
    $plan += ,@("refine-feats $s", "refine-feats", $s, @("--refine-out", $R, "--use-ce2"))
}
$plan += ,@("train-refine", "train-refine", "holdout", @("--refine-out", $R, "--refine-splits", "holdout,hold2,hold3",
                                                         "--refine-leaves", "127", "--refine-lr", "0.025", "--refine-bags", "3"))
$plan += ,@("tune",    "tune",    "holdout", @("--scored-dir", $R))
$plan += ,@("score",   "score",   "holdout", @("--scored-dir", $R))
$plan += ,@("refine test", "refine", "test", @("--refine-out", $R))
$plan += ,@("resolve", "resolve", "test",    @("--scored-dir", $R, "--out", $Out))

$started = ($From -eq "")
foreach ($step in $plan) {
    $label = $step[0]; $stage = $step[1]; $split = $step[2]; $extra = $step[3]
    if (-not $started) {
        if ($label -eq $From) { $started = $true } else { continue }
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
Write-Host "refine pass complete $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Green
