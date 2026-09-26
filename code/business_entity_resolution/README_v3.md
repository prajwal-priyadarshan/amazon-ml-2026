# v3 pipeline — runbook

Implements [docs/plan_v3_architecture.md](../../docs/plan_v3_architecture.md): retrieve →
rank → resolve, with a fine-tuned bi-encoder retrieval view, a gated cross-encoder judge, a
sibling graph, and per-S1 expected-F0.5 selection. v2 (`src/pipeline.py`, `work2/`,
`output2/`) is untouched and still submittable — run v3 into a separate work directory so
the 0.9547 file stays intact as the safety net.

All commands are run from `code/business_entity_resolution/`.

## Install

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# neural stages only (about 2.5 GB of wheels):
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 torch
.\.venv\Scripts\python.exe -m pip install -r requirements-v3.txt
```

The tabular stages need nothing but `requirements.txt`. If `transformers` 5.x errors on
import, pin `transformers==4.57.6`.

## Smoke test first (2 minutes)

Every stage on a 4,000-entity slice of the real data, including France:

```powershell
.\.venv\Scripts\python.exe tools\make_mini_dataset.py --data-dir ..\..\student_resource\dataset --out ..\..\work_mini\data
.\tools\run_v3.ps1 -DataDir ..\..\work_mini\data -WorkDir ..\..\work_mini\work -Out ..\..\work_mini\out
```

This is a correctness check, not a measurement: the mini pool is 30k records instead of
10.3M, so its recall and F0.5 read far too high. Only the full-pool numbers mean anything.

## Full run

```powershell
.\tools\run_v3.ps1                      # tabular stages end to end
.\tools\run_v3.ps1 -WithNeural          # adds embed / dense / train-bi / train-ce / ce
```

Or stage by stage, which is how it is meant to be driven, because each stage has a gate:

```powershell
$py = ".\.venv\Scripts\python.exe"
$D  = "..\..\student_resource\dataset"
$W  = "..\..\work3"

& $py -m src.v3.run prep        --split all   --data-dir $D --work-dir $W --jobs 8
& $py -m src.v3.run lexical     --split fit   --data-dir $D --work-dir $W
& $py -m src.v3.run lexical     --split holdout --data-dir $D --work-dir $W
& $py -m src.v3.run union       --split holdout --data-dir $D --work-dir $W
& $py -m src.v3.run diag        --split holdout --data-dir $D --work-dir $W   # <-- READ THIS
```

### The gate that decides everything else

`diag` prints candidate recall per view and the oracle ceiling `1.25r / (0.25 + r)`.

* **actual score close to the ceiling** → retrieval is the bottleneck. Work on views:
  `embed`/`dense`, `train-bi`, `--expand`.
* **actual score well below the ceiling** → ranking is the bottleneck. Work on the judge:
  `train-ce`, `ce`, and the decision rule.

v2 measured recall 0.9531, ceiling 0.9903, actual 0.9547 — about 3.6 points sat in ranking,
which is why the cross-encoder exists in v3 and why retrieval alone was not the answer.

### Stage order

| stage | what it does | gate |
|---|---|---|
| `prep` | normalise once, mine the segmentation vocabulary, shard by (country, source) | shards exist for fit/holdout/test |
| `lexical` | five sparse views, sharded top-k | ~40 candidates per S1 per source |
| `embed` | bi-encoder → float16 memmaps | 11.7M rows written (resumable) |
| `dense` | exact GPU top-k | recall rises in `diag` |
| `union` | fuse, add exact dense cosine, group features | |
| `diag` | per-view recall + oracle ceiling | pair recall ≥ 0.99 |
| `train-bi` | InfoNCE fine-tune (see below) | India recall ≥ 0.97 |
| `train-prune` | L2 pruner | |
| `prune` | keep top 8 per S1 per source (`--expand` adds view F) | recall after prune ≥ 0.994 |
| `train-ce` | fine-tune the judge | |
| `ce` | gated inference on the uncertain band | |
| `train-stack` | stacker + per-country isotonic | |
| `stack` | calibrated probabilities | |
| `tune` | pick threshold/margin vs expected-F0.5 on hold-out | improves on the previous stage |
| `score` | hold-out F0.5, by country and cluster size | |
| `errors` | miss taxonomy: retrieval / pruning / ranking / false positives | |
| `resolve` | write the two TSVs | validator says PASS |
| `pseudo` | export France pseudo-labels, then `train-ce --pseudo` | France stats match train priors |

Everything is resumable: a stage skips any shard whose output already exists. Re-run the
same command after a crash. Pass `--force` to recompute.

### Fine-tuning the retrieval encoder

`train-bi` is not in `run_v3.ps1`, because it is a second loop rather than a step: it needs
candidates to mine hard negatives from, and its output invalidates every embedding written
with the previous weights. Embedding files are keyed by `--emb-tag`, so give the fine-tuned
pass its own tag instead of deleting anything:

```powershell
& $py -m src.v3.run train-bi --data-dir $D --work-dir $W            # writes models/bi
& $py -m src.v3.run embed  --split test --data-dir $D --work-dir $W --emb-tag e5s-ft
& $py -m src.v3.run dense  --split test --data-dir $D --work-dir $W --emb-tag e5s-ft --force
& $py -m src.v3.run union  --split test --data-dir $D --work-dir $W --emb-tag e5s-ft --force
```

`embed` picks up `models/bi` automatically once it exists. Do the same for `holdout`, then
re-run `diag` and keep the fine-tune only if recall actually went up.

### Validate before submitting

```powershell
.\.venv\Scripts\python.exe ..\..\student_resource\utils\validate_submission.py `
    --matching ..\..\output3\matching_results.tsv `
    --candidate ..\..\output3\candidate_pairs.tsv `
    --test-dir ..\..\student_resource\dataset\test
```

## Rough timings on a 16 GB / RTX 4060 laptop

Estimates, not measurements — benchmark the first shard before committing to a full pass.

| stage | estimate |
|---|---|
| `prep` (all splits) | 20–35 min |
| `lexical` (test, all countries) | 2–4 h |
| `embed` (22.4M records — both pools, e5-small, seq 64) | 2–4 h |
| `dense` (exact search, all splits) | 40–80 min |
| `train-bi` (600k anchors) | 1–1.5 h |
| `train-prune` / `train-stack` | 5–20 min each |
| `prune` (test, features over ~208M pairs) | 1.5–2.5 h |
| `train-ce` (1.5M pairs, seq 128) | 2–3 h |
| `ce` (gated, ~7M pairs of 27.7M survivors) | 1.5–3 h |
| `stack` + `resolve` | 30–60 min |

Full tabular path end to end: about **6 hours**. Adding `embed`/`dense` and refitting: about
**+5 hours**. Adding the cross-encoder on top: about **+5 hours**. Levers, in the order worth
pulling: `--keep-union 40` (the feature pass over 208M pairs dominates the tabular cost),
`--fit-frac 0.06`, `--holdout-n 80000`, `--ce-max-len 96`, `--ce-lo 0.05 --ce-hi 0.90`.

If time runs out, cut in this order: `pseudo`, `--expand`, then the judge (`train-ce`/`ce`).
Never cut `prep` → `union` → `diag`: retrieval recall is the ceiling on everything after it.

## Memory

Target peak is ≤ 9 GB so Windows never pages. What keeps it there:

* one (country, source) block in memory at a time, never a global pair table;
* sparse top-k with `prefer="threads"` — scipy releases the GIL, so one copy of the index is
  shared instead of one per process (this was the v1 out-of-memory bug);
* document-side shards with a frozen vocabulary, so merging per-shard top-k stays exact;
* float32/int32 design matrices, LightGBM at `max_bin=63`;
* embeddings on disk as float16 memmaps, never in RAM.

Knobs if a block still does not fit: `--doc-shard 250000`, `--q-chunk 256`, `--threads 2`,
`--keep-union 40`, `--feat-chunk 500000`.

Claude Code's background-shell reaper has killed long runs on this machine three times. Run
full passes in a plain PowerShell window, or start Claude Code with
`CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1`.

## Models and licences

| role | model | licence | params |
|---|---|---|---|
| bi-encoder (default) | `intfloat/multilingual-e5-small` | MIT | 118M |
| bi-encoder (upgrade) | `intfloat/multilingual-e5-base` | MIT | 278M |
| cross-encoder judge | `FacebookAI/xlm-roberta-base` | MIT | 279M |
| tabular pruner + stacker | LightGBM | MIT | — |

All MIT, all far under the 8B limit. No external data: every alias table is mined from the
provided files, and the France region↔department signal comes from pseudo-labelled test
clusters, not a gazetteer.

The plan names e5-**base** as the primary; the default here is e5-**small** because it
embeds 11.7M records in 1–2 h instead of 4–6 h, and the records are short enough (a name
plus an address, well under 64 tokens) that the gap is smallest in exactly this regime.
Switch with `--bi-model intfloat/multilingual-e5-base --emb-tag e5b` once the small model's
recall is measured.

## Glued-name segmentation

`prep` mines a word list from the Source-1 names in the provided files (nothing external)
and uses it to split glued name tokens: `akshayagases` → `akshaya gases`,
`wilfordhancock` → `wilford hancock`, `bordeauxparentssarl` → `bordeaux parents sarl`. Real
true pairs in the training data are exactly this shape — a domain with its spaces removed
against the spaced reference name — and they defeat the word view, the exact-name key and
the phonetic skeleton simultaneously.

It adds `name_seg` and `seg_key` at prep time, points the word retrieval view at `name_seg`,
adds an exact `seg_key` join, and adds two pair features (`seg_tset`, `seg_key_eq`).
Splitting is conservative: every part must be a known word of at least three characters, at
most four parts, and the split has to beat leaving the token alone. On the mini set it fires
on 0.85% of rows and every split inspected was correct.

**Its effect on the score is not yet measured at full scale.** `--no-segment` turns it off
for an A/B. Do not try to judge it on the mini set: the mini hold-out contains no France
rows (France is test-only, and French glued names are where this fires hardest) and it fits
the stacker on a few hundred rows, so two extra features move it by a point in either
direction on noise alone.

## What differs from the plan, and why

* **No Hungarian solver.** Ownership is one-sided — each S2/S3 record has at most one owner,
  while an S1 entity may own up to 11 — so maximising Σ(p − threshold) decomposes per record
  and the optimum *is* the argmax. A matching solver would only matter if the per-S1 cap
  bound, and with a measured maximum cluster of 11 it effectively never does. The cap is
  still enforced and logged when it fires. See `src/v3/resolve.py`.
* **No exact per-view lexical cosine for every union pair.** Filling those needs a second
  full transform pass over the corpus. Each pair instead carries its per-view rank and its
  score where that view retrieved it, plus one exact bi-encoder cosine computed from the
  memmap for *all* union pairs. See `src/v3/lexical.py`.
* **Sibling groups are within-source.** Cross-source groups would need both source shards of
  a country resident at once. See `src/v3/sibling.py`.
* **The neural stages read raw text; the sparse and fuzzy stages read the folded text.**
  `unidecode` turns `राम मार्केटिंग` into `raam maarkettinng`: right for character n-grams,
  destructive for a multilingual encoder. Both views are stored at `prep` time.
