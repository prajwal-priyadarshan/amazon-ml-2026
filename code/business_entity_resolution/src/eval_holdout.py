"""Score the trained model against the 75% of train S1 entities NOT used for training/tuning,
using the FULL (unsubsampled) S2/S3 universe as the candidate pool -- a much larger and more
realistic test of generalisation than the internal held-out split in `pipeline.py train`, which
only holds out 10% of the already-25%-subsampled universe.

  python -m src.eval_holdout --data-dir ../../student_resource/dataset --work-dir ../../work \
      --jobs 4 --frac 0.25 --seed 0
"""
import argparse
import json

import joblib
import numpy as np

from . import textnorm
from .data import read_ground_truth, read_source
from .decision import to_lists
from .metric import f05_single, macro_f05
from .pipeline import concat_sources, decide, log, prepare, score_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--frac", type=float, default=0.25, help="must match the frac used at train time")
    ap.add_argument("--seed", type=int, default=0, help="must match the seed used at train time")
    ap.add_argument("--limit-s1", type=int, default=0)
    a = ap.parse_args()

    with open(f"{a.work_dir}/config.json") as f:
        cfg = json.load(f)
    if cfg.get("aliases"):
        textnorm.load_aliases(cfg["aliases"])
    bundle = joblib.load(f"{a.work_dir}/model.joblib")

    tr = f"{a.data_dir}/train"
    s1_all = read_source(f"{tr}/train_source1.tsv")
    d = concat_sources(read_source(f"{tr}/train_source2.tsv"), read_source(f"{tr}/train_source3.tsv"))
    gt = read_ground_truth(f"{tr}/train_ground_truth.tsv")

    trained_on = s1_all.sample(frac=a.frac, random_state=a.seed).index
    holdout = s1_all.drop(trained_on).reset_index(drop=True)
    if a.limit_s1:
        holdout = holdout.sample(n=min(a.limit_s1, len(holdout)), random_state=1).reset_index(drop=True)
    log(f"holdout: {len(holdout):,} S1 entities (never used for train/val/cal/tune/test), "
        f"full S2/S3 universe: {len(d):,} records")

    s1, d = prepare(holdout, a.jobs), prepare(d, a.jobs)
    pairs = score_all(s1, d, bundle, a.jobs, cfg.get("k", 30), cfg.get("keep", 30))
    s1_ids, d_ids = s1.entity_id.values, d.entity_id.values

    matches = decide(pairs, cfg["mode"], cfg["thr"], s1_ids, d_ids, cfg["margin"])
    truth = {e: set(gt[e]) for e in s1_ids}
    score = macro_f05(matches, truth, list(s1_ids))
    n_matched = sum(1 for v in matches.values() if v)
    log(f"HOLDOUT (unseen S1, full negative universe): macro F0.5={score:.4f}  "
        f"{n_matched:,}/{len(s1_ids):,} predicted >=1 match  "
        f"(true prior: {sum(1 for e in s1_ids if gt[e]) / len(s1_ids):.4f})")

    # Breakdowns: by country, and by number of true matches incl. singletons (factors 3 and 4).
    country = s1.country.values
    ntrue = np.array([len(gt[e]) for e in s1_ids])
    per = np.array([f05_single(set(matches.get(e, ())), truth[e]) for e in s1_ids])
    for c in sorted(set(country)):
        m = country == c
        log(f"  [{c}] F0.5={per[m].mean():.4f}  (n={m.sum():,})")
    for k in range(0, 7):
        m = (ntrue == k) if k < 6 else (ntrue >= 6)
        if m.any():
            log(f"  #true={k}{'+' if k == 6 else ''}: F0.5={per[m].mean():.4f}  (n={m.sum():,})")


if __name__ == "__main__":
    main()
