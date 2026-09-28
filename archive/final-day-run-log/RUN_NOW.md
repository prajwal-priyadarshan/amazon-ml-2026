# RUN NOW — copy-paste commands

Open a **new PowerShell window**. Not through Claude Code: its background-shell reaper has
killed long runs on this machine three times.

Full detail lives in [code/business_entity_resolution/README_v3.md](code/business_entity_resolution/README_v3.md).
This file is just the commands.

---

## 1. Go to the project and keep the laptop awake

```powershell
cd "C:\Users\prajw\OneDrive - Amrita Vishwa Vidyapeetham\Desktop\Hackathons\amazon-ml\code\business_entity_resolution"
powercfg /change standby-timeout-ac 0
```

## 2. Optional 2-minute confidence check

Same code, 4,000-entity slice of the real data. Ends with the official validator.

```powershell
.\.venv\Scripts\python.exe tools\make_mini_dataset.py --data-dir ..\..\student_resource\dataset --out ..\..\work_mini\data
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_v3.ps1 -DataDir ..\..\work_mini\data -WorkDir ..\..\work_mini\work -Out ..\..\work_mini\out -Expand
```

Ignore its F0.5: the mini pool is 30k records instead of 10.3M, so the number reads far too
high. It only proves the code runs.

## 3. The full tabular run (~6 h, no PyTorch needed)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_v3.ps1 -Expand *>&1 | Tee-Object -FilePath ..\..\work3_run.log
```

Writes `..\..\output3\matching_results.tsv` and `..\..\output3\candidate_pairs.tsv`, then
validates them. Needs about 20 GB free.

**`output2\matching_results.tsv` (the honest 0.9547 file) is never touched — it stays
submittable the whole time.**

### If it dies

Re-run the exact same command. Every stage skips shards it already wrote, so a crash costs
one shard, not the pass.

## 4. Fastest path to the number that decides everything else

Rather than waiting ~3 h for `diag` inside the full run, do the hold-out first (~1 h):

```powershell
$py = ".\.venv\Scripts\python.exe"; $D = "..\..\student_resource\dataset"; $W = "..\..\work3"
& $py -m src.v3.run prep    --split all     --data-dir $D --work-dir $W --jobs 8
& $py -m src.v3.run lexical --split holdout --data-dir $D --work-dir $W
& $py -m src.v3.run union   --split holdout --data-dir $D --work-dir $W
& $py -m src.v3.run diag    --split holdout --data-dir $D --work-dir $W
```

Then run step 3 — it reuses everything above.

### Reading `diag`

It prints `PAIR RECALL` and `ORACLE macro F0.5 CEILING`. Later, `score` prints
`HOLDOUT macro F0.5`. Compare them:

| reading | meaning | pass 2 should be |
|---|---|---|
| hold-out F0.5 close to the ceiling | retrieval is the bottleneck | `embed` / `dense` (step 6) |
| hold-out F0.5 well below the ceiling | ranking is the bottleneck | the cross-encoder (step 7) |

v2's measured triple, for comparison: recall 0.9531, ceiling 0.9903, actual 0.9547 — i.e.
~3.6 points sat in ranking.

## 5. Validate and submit

```powershell
.\.venv\Scripts\python.exe ..\..\student_resource\utils\validate_submission.py `
    --matching ..\..\output3\matching_results.tsv `
    --candidate ..\..\output3\candidate_pairs.tsv `
    --test-dir ..\..\student_resource\dataset\test
```

`PASS` means safe to upload. Upload `matching_results.tsv`; keep `candidate_pairs.tsv` for
the final zip.

---

## 6. Pass 2 — dense retrieval (+~5 h, needs PyTorch)

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch
.\.venv\Scripts\python.exe -m pip install -r requirements-v3.txt
```

```powershell
$py = ".\.venv\Scripts\python.exe"; $D = "..\..\student_resource\dataset"; $W = "..\..\work3"
foreach ($s in "fit","holdout","test") {
    & $py -m src.v3.run embed --split $s --data-dir $D --work-dir $W
    & $py -m src.v3.run dense --split $s --data-dir $D --work-dir $W
    & $py -m src.v3.run union --split $s --data-dir $D --work-dir $W --force
}
& $py -m src.v3.run diag        --split holdout --data-dir $D --work-dir $W
& $py -m src.v3.run train-prune --data-dir $D --work-dir $W
foreach ($s in "fit","holdout","test") { & $py -m src.v3.run prune --split $s --data-dir $D --work-dir $W --expand --force }
& $py -m src.v3.run train-stack --data-dir $D --work-dir $W
foreach ($s in "holdout","test") { & $py -m src.v3.run stack --split $s --data-dir $D --work-dir $W --force }
& $py -m src.v3.run tune    --split holdout --data-dir $D --work-dir $W
& $py -m src.v3.run score   --split holdout --data-dir $D --work-dir $W
& $py -m src.v3.run resolve --split test --data-dir $D --work-dir $W --out ..\..\output3
```

Resubmit only if `score` beats what pass 1 reported. Embeddings need ~17 GB of disk.

## 7. Pass 3 — cross-encoder judge (+~5 h)

Only start this with a clear 8-hour window left. A half-finished pass gives you nothing.
This code path has not been run yet.

```powershell
$py = ".\.venv\Scripts\python.exe"; $D = "..\..\student_resource\dataset"; $W = "..\..\work3"
& $py -m src.v3.run train-ce --data-dir $D --work-dir $W
foreach ($s in "fit","holdout","test") { & $py -m src.v3.run ce --split $s --data-dir $D --work-dir $W }
& $py -m src.v3.run train-stack --data-dir $D --work-dir $W
foreach ($s in "holdout","test") { & $py -m src.v3.run stack --split $s --data-dir $D --work-dir $W --force }
& $py -m src.v3.run tune    --split holdout --data-dir $D --work-dir $W
& $py -m src.v3.run score   --split holdout --data-dir $D --work-dir $W
& $py -m src.v3.run resolve --split test --data-dir $D --work-dir $W --out ..\..\output3
```

---

## Running short on time

Levers in the order worth pulling, added to any `run_v3.ps1` or stage command:

| flag | effect |
|---|---|
| `--keep-union 40` | biggest tabular saving — the feature pass over ~208M pairs dominates |
| `--fit-frac 0.06` | halves the training pass (265k → 132k S1 entities) |
| `--holdout-n 80000` | faster hold-out scoring |
| `--ce-max-len 96` | roughly halves cross-encoder cost |
| `--ce-lo 0.05 --ce-hi 0.90` | narrows the judged band |

## Diagnostics worth running once the run finishes

```powershell
& $py -m src.v3.run errors --split holdout --data-dir $D --work-dir $W
```

Buckets every missed true pair into never-retrieved / pruned-away / scored-too-low, broken
down by country, source, native script, missing address and cluster size — and counts false
positives. That tells you which single thing to fix next.

## Data this runs on

24,229,173 rows normalised once. Then:

| split | S1 entities | matched against |
|---|---|---|
| `fit` (trains the models) | 264,818 | full 10,320,219-record train pool |
| `holdout` (honest score) | 150,000, never trained on | the same full pool |
| `test` (submission) | 1,732,544 | 9,969,589 records |

Full pool, never subsampled — a subsampled negative pool reads about 3 points high, which is
how v1's "0.9668" turned out to be 0.9388.
