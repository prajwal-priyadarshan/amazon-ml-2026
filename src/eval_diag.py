"""Blocking diagnostics on unseen S1 entities against the FULL S2/S3 pool.

Candidate recall is the hard ceiling on the final score, so this measures it directly and
reports the oracle macro F0.5 a perfect classifier could reach on these candidates. No
feature computation or model scoring, so it is cheap enough to iterate on retrieval.

  python -m src.eval_diag --data-dir student_resource/dataset --limit-s1 150000 --jobs 4
"""
import argparse

import numpy as np
import pandas as pd

from . import textnorm
from .data import read_ground_truth, read_source
from .pipeline import concat_sources, iter_blocks, log, prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--frac", type=float, default=0.5, help="fraction that was used for training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-s1", type=int, default=150_000)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--keep", type=int, default=30)
    a = ap.parse_args()
    if a.aliases:
        textnorm.load_aliases(a.aliases)

    tr = f"{a.data_dir}/train"
    s1_all = read_source(f"{tr}/train_source1.tsv")
    d = concat_sources(read_source(f"{tr}/train_source2.tsv"), read_source(f"{tr}/train_source3.tsv"))
    gt = read_ground_truth(f"{tr}/train_ground_truth.tsv")
    holdout = s1_all.drop(s1_all.sample(frac=a.frac, random_state=a.seed).index)
    holdout = holdout.sample(n=min(a.limit_s1, len(holdout)), random_state=1).reset_index(drop=True)
    log(f"diag on {len(holdout):,} unseen S1 vs full pool of {len(d):,} S2/S3 records")

    s1, d = prepare(holdout, a.jobs), prepare(d, a.jobs)
    s1_ids, d_ids = s1.entity_id.values, d.entity_id.values
    owner = {m: s for s in s1_ids for m in gt[s]}
    n = len(s1_ids)

    parts = []
    for c, src, n_s1, blk in iter_blocks(s1, d, a.jobs, a.k, a.keep):
        log(f"country={c} src={src}: {n_s1:,} S1, {len(blk):,} candidate pairs")
        parts.append(blk[["s1_pos", "d_pos"]])
    cand = pd.concat(parts, ignore_index=True)
    del parts

    y = (pd.Series(d_ids[cand.d_pos.values]).map(owner).values == s1_ids[cand.s1_pos.values])
    ntrue = np.array([len(gt[e]) for e in s1_ids], dtype=np.float64)
    found = np.bincount(cand.s1_pos.values, weights=y.astype(np.float64), minlength=n)
    country = s1.country.values

    log(f"pairs={len(cand):,}  cands/S1={len(cand) / n:.1f}  true pairs={int(ntrue.sum()):,}")
    log(f"CANDIDATE RECALL = {found.sum() / ntrue.sum():.4f}")
    for c in sorted(set(country)):
        m = country == c
        log(f"  recall[{c}] = {found[m].sum() / max(ntrue[m].sum(), 1):.4f}")

    has = ntrue > 0
    r = found[has] / ntrue[has]
    log(f"  entities fully recovered: {(r >= 1).mean():.4f}   entities with zero hits: {(r == 0).mean():.4f}")
    for k in range(1, 7):
        m = (ntrue == k) if k < 6 else (ntrue >= 6)
        if m.any():
            log(f"  recall by #true={k}{'+' if k == 6 else ''}: {(found[m] / ntrue[m]).mean():.4f}  (n={int(m.sum()):,})")

    # Oracle: perfect precision, recall limited to what blocking retrieved. Singletons score 1.
    rr = np.where(ntrue > 0, found / np.maximum(ntrue, 1), 0.0)
    oracle = np.where(ntrue == 0, 1.0, np.where(rr > 0, 1.25 * rr / (0.25 + rr), 0.0))
    log(f"ORACLE macro F0.5 ceiling = {oracle.mean():.4f}")
    log("DONE")


if __name__ == "__main__":
    main()
