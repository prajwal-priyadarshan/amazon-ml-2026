# Kept results

Three milestones out of the ~24 submission variants actually generated during the
competition, chosen to show the score progression rather than every intermediate experiment
(hybrid ensembles, per-country ratio sweeps, loosened-threshold variants, etc. — all real,
none of them kept here). Each folder holds `matching_results.tsv`, the file format the
[official validator](../tools/validate_submission.py) checks against the test source files.

| folder | pipeline | held-out macro F0.5 | portal score |
|---|---|---|---|
| [`baseline_v2/`](baseline_v2/matching_results.tsv) | v2: multi-view lexical retrieval, two-stage GBDT, isotonic calibration | 0.9547 | ~0.90s (v1 lineage) |
| [`neural_v3/`](neural_v3/matching_results.tsv) | v3: + fine-tuned multilingual bi-encoder retrieval, XLM-R cross-encoder judge | 0.9776 | 0.9776 |
| [`final_submission/`](final_submission/matching_results.tsv) | v3 + cluster-aware refine model, trained on held-out splits | 0.9862 / 0.9886 (open/closed) | **0.985018 (rank #568)** |

`final_submission/` is the exact file uploaded to the leaderboard for the final score. See
the top-level [README](../README.md) for the full story and [`docs/methodology.md`](../docs/methodology.md)
for the write-up submitted for judging.

`candidate_pairs.tsv` (the blocking output each of these was resolved from) isn't tracked —
it runs 2.5+ GB per variant, well over what's sensible to keep in git. Regenerate it with
[`docs/runbook.md`](../docs/runbook.md) if you need it.
