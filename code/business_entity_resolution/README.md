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

# 2. train on a 25% cluster-consistent sample of train, tune decision rule, report held-out F0.5
python -m src.pipeline train --data-dir ../../student_resource/dataset --work-dir ../../work \
    --aliases ../../work/aliases.json --frac 0.25 --jobs 16

# 3. predict the full test set and write both TSVs
python -m src.pipeline predict --data-dir ../../student_resource/dataset --work-dir ../../work \
    --out ../../output --jobs 16

# 4. validate
python3 ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```
Use `--limit-s1 5000` on either command for a quick dry run.

## Modules
| File | Role |
|---|---|
| `src/textnorm.py` | Name/address normalisation (latin fold, legal-suffix split, leetspeak/URL/junk cleanup, abbreviation + learned aliases) |
| `src/retrieval.py` | Address TF-IDF, name char-n-gram TF-IDF and exact name-key views, rank fusion |
| `src/features.py` | rapidfuzz, number, legal-suffix and group-competition features |
| `src/decision.py` | one-owner rule, global threshold, Monte-Carlo expected-F0.5 selection |
| `src/metric.py` | exact macro F0.5 |
| `src/learn_aliases.py` | data-driven alias mining |
| `src/pipeline.py` | `train` / `predict` CLI |

## Known limitations (planned next)
- The competition features only see S1 entities that retrieved a record (no reverse S2/S3 -> S1 retrieval yet).
- Native-script names are only handled through romanisation; the multilingual encoder + cross-encoder is the next step.
- The training universe is a subsample, so hard negatives are sparser than in the full data; expect the real
  leaderboard score to be below the held-out number.
- No France-specific adaptation yet (pseudo-labelling of test clusters is planned).
