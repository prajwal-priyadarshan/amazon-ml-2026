#!/bin/bash
set -e
cd "$(dirname "$0")/../code/business_entity_resolution"
PY=.venv/Scripts/python.exe
D=../../student_resource/dataset
W=../../work_v2
$PY -m src.pipeline train --data-dir $D --work-dir $W --aliases ../../work/aliases.json --frac 0.5 --jobs 16 \
  --pseudo-country France --pseudo-model ../../work/model.joblib > $W/train.log 2>&1
$PY -m src.pipeline predict --data-dir $D --work-dir $W --out ../../output_v2 --jobs 16 > $W/predict.log 2>&1
$PY ../../student_resource/utils/validate_submission.py --matching ../../output_v2/matching_results.tsv \
  --candidate ../../output_v2/candidate_pairs.tsv --test-dir $D/test --check-ids > $W/validate.log 2>&1
$PY -m src.eval_holdout --data-dir $D --work-dir $W --frac 0.5 --seed 0 --limit-s1 250000 --jobs 16 > $W/eval_holdout.log 2>&1
echo ALL_DONE > $W/done.flag
