"""Baseline pipeline: multi-view retrieval -> pair features -> GBDT -> calibration -> one-owner + decision.

  python -m src.pipeline train   --data-dir ../../student_resource/dataset --work-dir ../../work [--frac 0.25]
  python -m src.pipeline predict --data-dir ../../student_resource/dataset --work-dir ../../work --out ../../output
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import textnorm
from .data import read_ground_truth, read_source, write_id_lists
from .decision import one_owner, select_expected_f05, select_threshold, to_lists
from .features import FEATURE_COLS, compute_features
from .metric import macro_f05
from .retrieval import gen_candidates


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(df, n_jobs):
    return textnorm.normalize_frame(df, n_jobs=n_jobs)


def concat_sources(s2, s3):
    s2 = s2.assign(src=2)
    s3 = s3.assign(src=3)
    return pd.concat([s2, s3], ignore_index=True)


def build_pairs(s1, dall, n_jobs, k=20, keep=15):
    """Candidates + features, country by country (country is an open set of labels)."""
    out = []
    for c, s1c in s1.groupby("country"):
        parts = []
        for src in (2, 3):
            dc = dall[(dall.country == c) & (dall.src == src)]
            if dc.empty:
                continue
            cand = gen_candidates(s1c.reset_index(drop=True), dc.reset_index(drop=True),
                                  k_addr=k, k_name=k, keep=keep, n_jobs=n_jobs)
            if cand.empty:
                continue
            cand["s1_pos"] = s1c.index.values[cand["s1_pos"].values]
            cand["d_pos"] = dc.index.values[cand["d_pos"].values]
            parts.append(cand)
        if not parts:
            continue
        cand = pd.concat(parts, ignore_index=True)
        log(f"country={c}: {len(s1c):,} S1, {len(cand):,} candidate pairs")
        out.append(compute_features(cand, s1, dall))
    return pd.concat(out, ignore_index=True)


def make_model():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=127, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, min_child_samples=50, n_jobs=-1,
                                  verbose=-1), True
    except Exception:  # missing libomp etc.: fall back so the pipeline still runs
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63), False


def decide(df, mode, thr, s1_ids, dall_ids, margin=0.0, miss_rate=0.0):
    own = one_owner(df, margin=margin)
    sel = select_expected_f05(own, miss_rate=miss_rate) if mode == "expected" else select_threshold(own, thr)
    return to_lists(sel, s1_ids, dall_ids)


def cmd_train(a):
    if a.aliases:
        os.environ["ER_ALIASES"] = a.aliases
        log(f"aliases loaded: {textnorm.load_aliases(a.aliases)}")
    tr = f"{a.data_dir}/train"
    s1 = read_source(f"{tr}/train_source1.tsv")
    d = concat_sources(read_source(f"{tr}/train_source2.tsv"), read_source(f"{tr}/train_source3.tsv"))
    gt = read_ground_truth(f"{tr}/train_ground_truth.tsv")
    owner = {m: s for s, ms in gt.items() for m in ms}

    rng = np.random.default_rng(a.seed)
    s1 = s1.sample(frac=a.frac, random_state=a.seed).reset_index(drop=True)
    if a.limit_s1:
        s1 = s1.iloc[:a.limit_s1]
    matched_here = set(m for s in s1.entity_id for m in gt[s])
    is_matched_any = d.entity_id.isin(owner.keys())
    keep = d.entity_id.isin(matched_here) | (~is_matched_any & (rng.random(len(d)) < a.frac))
    d = d[keep].reset_index(drop=True)
    log(f"universe: {len(s1):,} S1, {len(d):,} S2/S3 records")

    s1, d = prepare(s1, a.jobs), prepare(d, a.jobs)
    pairs = build_pairs(s1, d, a.jobs)
    s1_ids, d_ids = s1.entity_id.values, d.entity_id.values
    pairs["y"] = (pd.Series(d_ids[pairs.d_pos.values]).map(owner).values == s1_ids[pairs.s1_pos.values]).astype("int8")
    total_true = sum(len(gt[s]) for s in s1_ids)
    log(f"pairs={len(pairs):,}  positives={pairs.y.sum():,}  blocking recall={pairs.y.sum() / max(total_true, 1):.4f}")

    grp = rng.choice(5, size=len(s1), p=[0.6, 0.1, 0.1, 0.1, 0.1])  # train/val/cal/tune/test by S1
    g = grp[pairs.s1_pos.values]
    model, is_lgb = make_model()
    Xtr, ytr = pairs.loc[g == 0, FEATURE_COLS], pairs.loc[g == 0, "y"]
    if is_lgb:
        import lightgbm as lgb
        model.fit(Xtr, ytr, eval_set=[(pairs.loc[g == 1, FEATURE_COLS], pairs.loc[g == 1, "y"])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(Xtr, ytr)
    raw = model.predict_proba(pairs[FEATURE_COLS])[:, 1]
    cal = IsotonicRegression(out_of_bounds="clip").fit(raw[g == 2], pairs.y.values[g == 2])
    pairs["prob"] = cal.predict(raw)

    def score(group, mode, thr=0.0, margin=0.0):
        ids = np.flatnonzero(grp == group)
        sub = pairs[np.isin(pairs.s1_pos.values, ids)]
        pred = decide(sub, mode, thr, s1_ids, d_ids, margin)
        ents = [s1_ids[i] for i in ids]
        return macro_f05(pred, {e: set(gt[e]) for e in ents}, ents)

    best = ("threshold", 0.5, 0.0, -1.0)
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        for m in (0.0, 0.1, 0.2):
            sc = score(3, "threshold", thr, m)
            log(f"tune thr={thr} margin={m}: F0.5={sc:.4f}")
            if sc > best[3]:
                best = ("threshold", thr, m, sc)
    sc_exp = score(3, "expected")
    log(f"tune expected-F0.5: {sc_exp:.4f}")
    if sc_exp > best[3]:
        best = ("expected", 0.0, 0.0, sc_exp)
    final = score(4, best[0], best[1], best[2])
    log(f"BEST mode={best[0]} thr={best[1]} margin={best[2]}  tune={best[3]:.4f}  held-out test F0.5={final:.4f}")

    os.makedirs(a.work_dir, exist_ok=True)
    joblib.dump({"model": model, "cal": cal}, f"{a.work_dir}/model.joblib")
    with open(f"{a.work_dir}/config.json", "w") as f:
        json.dump({"mode": best[0], "thr": best[1], "margin": best[2], "heldout_f05": final,
                   "aliases": a.aliases}, f, indent=1)
    imp = getattr(model, "feature_importances_", None)
    if imp is not None:
        log("top features: " + ", ".join(f"{n}:{int(v)}" for n, v in
                                          sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])[:10]))


def cmd_predict(a):
    with open(f"{a.work_dir}/config.json") as f:
        cfg = json.load(f)
    if cfg.get("aliases"):
        os.environ["ER_ALIASES"] = cfg["aliases"]
        textnorm.load_aliases(cfg["aliases"])
    bundle = joblib.load(f"{a.work_dir}/model.joblib")
    te = f"{a.data_dir}/test"
    s1 = read_source(f"{te}/test_source1.tsv")
    d = concat_sources(read_source(f"{te}/test_source2.tsv"), read_source(f"{te}/test_source3.tsv"))
    if a.limit_s1:
        s1 = s1.iloc[:a.limit_s1]
    s1, d = prepare(s1, a.jobs), prepare(d, a.jobs)
    log(f"test: {len(s1):,} S1, {len(d):,} S2/S3; countries={sorted(s1.country.unique())}")
    pairs = build_pairs(s1, d, a.jobs)
    raw = bundle["model"].predict_proba(pairs[FEATURE_COLS])[:, 1]
    pairs["prob"] = bundle["cal"].predict(raw)
    s1_ids, d_ids = s1.entity_id.values, d.entity_id.values

    cands = to_lists(pairs, s1_ids, d_ids)
    matches = decide(pairs, cfg["mode"], cfg["thr"], s1_ids, d_ids, cfg["margin"])
    write_id_lists(f"{a.out}/candidate_pairs.tsv", s1_ids, cands, ["source1_entity_id", "candidate_entity_ids"])
    write_id_lists(f"{a.out}/matching_results.tsv", s1_ids, matches, ["source1_entity_id", "matched_entity_ids"])
    n = sum(1 for v in matches.values() if v)
    log(f"wrote {a.out}: {n:,}/{len(s1_ids):,} S1 with >=1 match (train prior ~94%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit-s1", type=int, default=0)
    a = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
