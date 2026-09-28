"""Two-stage entity resolution pipeline.

  multi-view retrieval -> pair features -> stage-1 GBDT -> competition features
  -> stage-2 GBDT -> isotonic calibration -> one-owner -> threshold/margin decision

  python -m src.pipeline train   --data-dir student_resource/dataset --work-dir work2
  python -m src.pipeline predict --data-dir student_resource/dataset --work-dir work2 --out output2

Memory notes: every stage that touches the full candidate set is chunked and keeps only
lean columns, because the full test set is ~11.7M rows on a 16GB machine.
"""
import argparse
import gc
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
from .features import (FEATURE_COLS, GROUP_COLS, RETRIEVAL_COLS, STAGE2_COLS,
                       add_group_features, add_stage2_features, compute_features)
from .metric import macro_f05
from .retrieval import gen_candidates

LEAN = ["s1_pos", "d_pos"]
PRUNE = 0.02  # stage-1 probability below which a pair can never survive the final threshold


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(df, n_jobs):
    return textnorm.normalize_frame(df, n_jobs=n_jobs)


def concat_sources(s2, s3):
    s2 = s2.assign(src=2)
    s3 = s3.assign(src=3)
    return pd.concat([s2, s3], ignore_index=True)


def block_candidates(s1c, dall, c, src, n_jobs, k=30, keep=60):
    """Lean candidate frame for one (country, source) block, with retrieval-side group
    features. Splitting by source halves peak memory, and because S2 and S3 occupy disjoint
    row ranges of ``dall`` every per-record grouping stays exact within the block.
    Positions are global (index into s1 / dall)."""
    dc = dall[(dall.country == c) & (dall.src == src)]
    if dc.empty:
        return pd.DataFrame()
    cand = gen_candidates(s1c.reset_index(drop=True), dc.reset_index(drop=True),
                          k_addr=k, k_name=k, keep=keep, n_jobs=n_jobs)
    if cand.empty:
        return pd.DataFrame()
    cand["s1_pos"] = s1c.index.values[cand["s1_pos"].values].astype(np.int32)
    cand["d_pos"] = dc.index.values[cand["d_pos"].values].astype(np.int32)
    cand["src"] = np.int8(src)
    cand = add_group_features(cand)
    for col in ("rrf", "sim", "rec_best", "rec_n", "rec_margin", "s1_rank", "s1_best", "s1_gap"):
        cand[col] = cand[col].astype(np.float32)
    return cand


def iter_blocks(s1, dall, n_jobs, k, keep):
    for c, s1c in s1.groupby("country"):
        for src in (2, 3):
            cand = block_candidates(s1c, dall, c, src, n_jobs, k, keep)
            if not cand.empty:
                yield c, src, len(s1c), cand


def feature_matrix(cand, s1, dall, cols, chunk=1_500_000):
    """float32 design matrix for ``cols`` (avoids pandas' float64 upcast, which doubles peak RSS)."""
    pair = compute_features(cand, s1, dall, chunk=chunk)
    # `src` is produced by both sides; take it from the pair features only, or the concat
    # yields duplicate labels and out[cols] silently changes shape.
    have = [c for c in cols if c in cand.columns and c not in pair.columns]
    out = pd.concat([cand[have].reset_index(drop=True), pair.reset_index(drop=True)], axis=1)
    return out[cols].astype(np.float32).values


def write_candidate_lists(path, s1_ids, d_ids, pairs, header):
    """Write per-S1 id lists for ~200M pairs.

    A Python loop over 1.7M rows joining ids one row at a time measured at 3 MB/min, which
    is hours for this file. Grouping with pandas keeps the aggregation in native code and
    the final serialisation in ``to_csv``, which is a different order of magnitude."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame({"s1_pos": pairs["s1_pos"].to_numpy(),
                       "d_id": pd.Categorical.from_codes(pairs["d_pos"].to_numpy(),
                                                         categories=pd.Index(d_ids)).astype(str)})
    joined = df.groupby("s1_pos", sort=True)["d_id"].agg(",".join)
    out = pd.DataFrame({header[0]: s1_ids,
                        header[1]: joined.reindex(np.arange(len(s1_ids))).fillna("").to_numpy()})
    out.to_csv(path, sep="\t", index=False, header=True, lineterminator="\n")


def score_candidates(cand, s1, dall, bundle, chunk=1_500_000):
    """Stage-1 score -> prune hopeless pairs -> competition features -> stage-2 score.
    Returns the lean frame with calibrated ``prob`` for the surviving pairs."""
    n0 = len(cand)
    p1 = np.empty(n0, np.float32)
    for s in range(0, n0, chunk):
        blk = cand.iloc[s:s + chunk]
        X = feature_matrix(blk, s1, dall, FEATURE_COLS, chunk)
        p1[s:s + chunk] = bundle["m1"].predict_proba(X)[:, 1].astype(np.float32)
        del X
    cand["prob1"] = p1
    del p1
    gc.collect()

    keep = cand["prob1"].values >= PRUNE
    cand = add_stage2_features(cand[keep].reset_index(drop=True))
    log(f"    stage-1 done: {n0:,} -> {len(cand):,} survive prune")

    p2 = np.empty(len(cand), np.float32)
    for s in range(0, len(cand), chunk):
        blk = cand.iloc[s:s + chunk]
        X = feature_matrix(blk, s1, dall, STAGE2_COLS, chunk)
        p2[s:s + chunk] = bundle["m2"].predict_proba(X)[:, 1].astype(np.float32)
        del X
    out = cand[LEAN].copy()
    out["prob"] = bundle["cal"].predict(p2).astype(np.float32)
    del cand, p2
    gc.collect()
    return out


def score_all(s1, dall, bundle, n_jobs, k=30, keep=60, cand_sink=None, ckpt=None):
    """Per (country, source): build candidates, score, keep only ids + probability.

    With ``ckpt`` each block is written to disk as soon as it is scored and reloaded on a
    later run, so an interrupted job resumes at the next block instead of recomputing
    everything. Blocking + scoring one block costs ~20 minutes, the whole pass ~80."""
    out = []
    if ckpt:
        os.makedirs(ckpt, exist_ok=True)
    for c, s1c in s1.groupby("country"):
        for src in (2, 3):
            tag = f"{c}_{src}"
            sp = f"{ckpt}/scored_{tag}.parquet" if ckpt else None
            cp = f"{ckpt}/cands_{tag}.parquet" if ckpt else None
            if sp and os.path.exists(sp) and os.path.exists(cp):
                out.append(pd.read_parquet(sp))
                if cand_sink is not None:
                    cand_sink.append(pd.read_parquet(cp))
                log(f"country={c} src={src}: loaded from checkpoint")
                continue
            cand = block_candidates(s1c, dall, c, src, n_jobs, k, keep)
            if cand.empty:
                continue
            log(f"country={c} src={src}: {len(s1c):,} S1, {len(cand):,} candidate pairs")
            lean = cand[LEAN]
            if cp:
                lean.to_parquet(cp, index=False)
            if cand_sink is not None:
                cand_sink.append(lean)
            scored = score_candidates(cand, s1, dall, bundle)
            if sp:
                scored.to_parquet(sp, index=False)
            out.append(scored)
            del cand, scored
            gc.collect()
            log(f"country={c} src={src}: scored")
    return pd.concat(out, ignore_index=True)


def make_model(n_estimators=1200):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=0.05, num_leaves=127,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  min_child_samples=50, n_jobs=-1, verbose=-1), True
    except Exception:  # missing libomp etc.: fall back so the pipeline still runs
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, max_leaf_nodes=63), False


def fit(X, y, Xv, yv):
    model, is_lgb = make_model()
    if is_lgb:
        import lightgbm as lgb
        model.fit(X, y, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model.fit(X, y)
    return model


def decide(df, mode, thr, s1_ids, dall_ids, margin=0.0, miss_rate=0.0):
    own = one_owner(df, margin=margin)
    sel = select_expected_f05(own, miss_rate=miss_rate) if mode == "expected" else select_threshold(own, thr)
    return to_lists(sel, s1_ids, dall_ids)


def cmd_train(a):
    if a.aliases:
        os.environ["ER_ALIASES"] = a.aliases
        log(f"aliases loaded: {textnorm.load_aliases(a.aliases)}")
    tr = f"{a.data_dir}/train"
    s1_all = read_source(f"{tr}/train_source1.tsv")
    d = concat_sources(read_source(f"{tr}/train_source2.tsv"), read_source(f"{tr}/train_source3.tsv"))
    gt = read_ground_truth(f"{tr}/train_ground_truth.tsv")
    owner = {m: s for s, ms in gt.items() for m in ms}

    rng = np.random.default_rng(a.seed)
    s1 = s1_all.sample(frac=a.frac, random_state=a.seed)
    if a.fit_s1 and a.fit_s1 < len(s1):
        s1 = s1.iloc[:a.fit_s1]
    s1 = s1.reset_index(drop=True)
    # The full S2/S3 pool is kept as the negative universe. Subsampling it (the old default)
    # thinned out hard negatives and was the main reason held-out F0.5 read ~3 points high.
    log(f"universe: {len(s1):,} S1 (frac={a.frac}), {len(d):,} S2/S3 records (full pool)")

    s1, d = prepare(s1, a.jobs), prepare(d, a.jobs)
    s1_ids, d_ids = s1.entity_id.values, d.entity_id.values
    grp = rng.choice(5, size=len(s1), p=[0.6, 0.1, 0.1, 0.1, 0.1])  # train/val/cal/tune/test by S1

    lean = []
    for c, src, n_s1, blk in iter_blocks(s1, d, a.jobs, a.k, a.keep):
        log(f"country={c} src={src}: {n_s1:,} S1, {len(blk):,} candidate pairs")
        lean.append(blk)
    cand = pd.concat(lean, ignore_index=True)
    del lean
    gc.collect()
    cand["y"] = (pd.Series(d_ids[cand.d_pos.values]).map(owner).values == s1_ids[cand.s1_pos.values]).astype("int8")
    total_true = sum(len(gt[s]) for s in s1_ids)
    recall = cand.y.sum() / max(total_true, 1)
    log(f"pairs={len(cand):,}  positives={cand.y.sum():,}  blocking recall={recall:.4f}")
    log(f"  oracle macro F0.5 ceiling at this recall ~= {1.25 * recall / (0.25 + recall):.4f}")

    g = grp[cand.s1_pos.values]
    y = cand.y.values
    # Keep every positive; sample negatives. Negatives are plentiful and highly redundant,
    # and this is what makes a 50% universe fit in memory at 60 candidates per entity.
    sub = (y == 1) | (rng.random(len(cand)) < a.neg_rate)
    log(f"fit rows: {sub.sum():,} ({(y[sub] == 1).sum():,} pos / {(y[sub] == 0).sum():,} neg)")

    idx_tr = np.flatnonzero(sub & (g == 0))
    idx_va = np.flatnonzero(sub & (g == 1))
    X1 = feature_matrix(cand.iloc[idx_tr], s1, d, FEATURE_COLS)
    Xv1 = feature_matrix(cand.iloc[idx_va], s1, d, FEATURE_COLS)
    m1 = fit(X1, y[idx_tr], Xv1, y[idx_va])
    del X1, Xv1
    gc.collect()
    log("stage-1 fitted")

    p1 = np.empty(len(cand), np.float32)
    for s in range(0, len(cand), 1_500_000):
        blk = cand.iloc[s:s + 1_500_000]
        Xb = feature_matrix(blk, s1, d, FEATURE_COLS)
        p1[s:s + 1_500_000] = m1.predict_proba(Xb)[:, 1].astype(np.float32)
        del Xb
    cand["prob1"] = p1
    del p1
    gc.collect()

    keep = cand["prob1"].values >= PRUNE
    cand, g, y, sub = cand[keep].reset_index(drop=True), g[keep], y[keep], sub[keep]
    add_stage2_features(cand)
    kept_recall = y.sum() / max(total_true, 1)
    log(f"after prune: {len(cand):,} pairs, recall={kept_recall:.4f}")

    idx_tr = np.flatnonzero(sub & (g == 0))
    idx_va = np.flatnonzero(sub & (g == 1))
    X2 = feature_matrix(cand.iloc[idx_tr], s1, d, STAGE2_COLS)
    Xv2 = feature_matrix(cand.iloc[idx_va], s1, d, STAGE2_COLS)
    m2 = fit(X2, y[idx_tr], Xv2, y[idx_va])
    del X2, Xv2
    gc.collect()
    log("stage-2 fitted")

    p2 = np.empty(len(cand), np.float32)
    for s in range(0, len(cand), 1_500_000):
        blk = cand.iloc[s:s + 1_500_000]
        Xb = feature_matrix(blk, s1, d, STAGE2_COLS)
        p2[s:s + 1_500_000] = m2.predict_proba(Xb)[:, 1].astype(np.float32)
        del Xb
    cal = IsotonicRegression(out_of_bounds="clip").fit(p2[g == 2], y[g == 2])
    pairs = cand[LEAN].copy()
    pairs["prob"] = cal.predict(p2).astype(np.float32)
    del cand, p2
    gc.collect()

    def score(group, mode, thr=0.0, margin=0.0):
        ids = np.flatnonzero(grp == group)
        sub_p = pairs[np.isin(pairs.s1_pos.values, ids)]
        pred = decide(sub_p, mode, thr, s1_ids, d_ids, margin)
        ents = [s1_ids[i] for i in ids]
        return macro_f05(pred, {e: set(gt[e]) for e in ents}, ents)

    best = ("threshold", 0.5, 0.0, -1.0)
    for thr in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
        for m in (0.0, 0.1, 0.2, 0.3):
            sc = score(3, "threshold", thr, m)
            if sc > best[3]:
                best = ("threshold", thr, m, sc)
                log(f"tune thr={thr} margin={m}: F0.5={sc:.4f}  <- best")
    sc_exp = score(3, "expected")
    log(f"tune expected-F0.5: {sc_exp:.4f}")
    if sc_exp > best[3]:
        best = ("expected", 0.0, 0.0, sc_exp)
    final = score(4, best[0], best[1], best[2])
    log(f"BEST mode={best[0]} thr={best[1]} margin={best[2]}  tune={best[3]:.4f}  held-out test F0.5={final:.4f}")

    os.makedirs(a.work_dir, exist_ok=True)
    joblib.dump({"m1": m1, "m2": m2, "cal": cal}, f"{a.work_dir}/model.joblib")
    with open(f"{a.work_dir}/config.json", "w") as f:
        json.dump({"mode": best[0], "thr": best[1], "margin": best[2], "heldout_f05": final,
                   "blocking_recall": float(recall), "aliases": a.aliases,
                   "k": a.k, "keep": a.keep}, f, indent=1)
    imp = getattr(m2, "feature_importances_", None)
    if imp is not None:
        log("top stage-2 features: " + ", ".join(f"{n}:{int(v)}" for n, v in
                                                 sorted(zip(STAGE2_COLS, imp), key=lambda x: -x[1])[:12]))


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

    sink = []
    pairs = score_all(s1, d, bundle, a.jobs, cfg.get("k", 30), cfg.get("keep", 60),
                      cand_sink=sink, ckpt=a.ckpt or f"{a.work_dir}/blocks")
    s1_ids, d_ids = s1.entity_id.values, d.entity_id.values
    log(f"all blocks done: {len(pairs):,} scored pairs; writing output")

    # Scored file first: it is the one the leaderboard reads, and it is far smaller.
    matches = decide(pairs, cfg["mode"], cfg["thr"], s1_ids, d_ids, cfg["margin"])
    write_id_lists(f"{a.out}/matching_results.tsv", s1_ids, matches, ["source1_entity_id", "matched_entity_ids"])
    n = sum(1 for v in matches.values() if v)
    log(f"wrote matching_results.tsv: {n:,}/{len(s1_ids):,} S1 with >=1 match (train prior ~94%)")
    del pairs, matches
    gc.collect()

    write_candidate_lists(f"{a.out}/candidate_pairs.tsv", s1_ids, d_ids,
                          pd.concat(sink, ignore_index=True),
                          ["source1_entity_id", "candidate_entity_ids"])
    del sink
    gc.collect()
    log(f"wrote {a.out}: done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out", default="output")
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--fit-s1", type=int, default=200_000,
                    help="cap on S1 entities used to build training rows (memory bound)")
    ap.add_argument("--neg-rate", type=float, default=0.30, help="negative sampling rate")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--keep", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit-s1", type=int, default=0)
    ap.add_argument("--ckpt", default=None, help="block checkpoint dir (default <work-dir>/blocks)")
    a = ap.parse_args()
    {"train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
