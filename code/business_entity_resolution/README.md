# Business Entity Resolution: phase 1 baseline

Pipeline: normalise -> multi-view retrieval -> pair features -> GBDT -> isotonic calibration ->
one-owner assignment -> threshold or expected-F0.5 selection.

All paths below are relative to this folder. Generated files go to `../../work` and `../../output`,
never into `student_resource/`.

## Setup
```bash
pip install -r requirements.txt
```
(On macOS LightGBM needs `brew install libomp`; without it the code falls back to scikit-learn's
HistGradientBoosting automatically.)

## Run (from `code/business_entity_resolution/`)
```bash
# 1. optional: learn address aliases from train ground truth (NY<->new york, romanised native states, ...)
python -m src.learn_aliases --data-dir ../../student_resource/dataset --out ../../work/aliases.json
#    open the JSON and delete any wrong entries before using it

# 2. preprocess once: normalise train + test, cache as parquet in ../../work/prep (inspect_*.tsv = cleaned samples per country)
python -m src.pipeline preprocess --data-dir ../../student_resource/dataset --work-dir ../../work \
    --aliases ../../work/aliases.json --jobs 16
#    train/predict load this cache automatically; pass the same --aliases (or none) to all three commands

# 3. train on a 25% cluster-consistent sample of train, tune decision rule, report held-out F0.5
python -m src.pipeline train --data-dir ../../student_resource/dataset --work-dir ../../work \
    --aliases ../../work/aliases.json --frac 0.25 --jobs 16

# 4. predict the full test set and write both TSVs
python -m src.pipeline predict --data-dir ../../student_resource/dataset --work-dir ../../work \
    --out ../../output --jobs 16

# 5. validate
python3 ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```
Use `--limit-s1 5000` on either command for a quick dry run.

## Modules
| File | Role |
|---|---|
| `src/textnorm.py` | Name/address normalisation (latin fold, legal-suffix split, leetspeak/URL/junk cleanup, abbreviation + learned aliases) |
| `src/retrieval.py` | Address-word TF-IDF, name and address letter-chunk TF-IDF, exact name key and reverse (S2/S3 -> S1) retrieval, rank fusion |
| `src/features.py` | rapidfuzz, number (incl. digit edit distance), truncation/extra-word, legal-suffix and competition features; stage-2 features built from stage-1 probabilities |
| `src/decision.py` | one-owner rule, global threshold, Monte-Carlo expected-F0.5 selection |
| `src/metric.py` | exact macro F0.5 |
| `src/learn_aliases.py` | data-driven alias mining |
| `src/pipeline.py` | `train` / `predict` CLI |

## Model
Stage 1 GBDT on pair features -> stage 2 GBDT that re-scores each pair using its competitors' stage-1 probabilities
(rank within S1, margin over the runner-up S1 for the same record, counts of confident candidates) -> isotonic
calibration -> one-owner rule -> threshold or expected-F0.5. `train` reports stage-1 and stage-2 held-out F0.5 and keeps the better one.
Speed knobs: `--k-rev 0 --k-addrc 0` turn off the two extra retrieval views; `--k`, `--keep` set the shortlist size.
`preprocess` also builds `work/vocab.json` (word counts from clean S1 names) used to split glued names such as `suzygillenpeak.com`.

## Known limitations (planned next)
- Native-script names are only handled through romanisation; the multilingual encoder + cross-encoder is the next step.
- The training universe is a subsample, so hard negatives are sparser than in the full data; expect the real
  leaderboard score to be below the held-out number.
- No France-specific adaptation yet (pseudo-labelling of test clusters is planned).
