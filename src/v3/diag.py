"""Diagnostics: per-view recall with the oracle ceiling, and the miss taxonomy.

This is step 1 of the plan, and it runs before any modelling decision, because the two
numbers it produces answer different questions and only one of them is actionable at a time:

* **candidate recall** and the **oracle ceiling** ``1.25r / (0.25 + r)`` bound what a perfect
  classifier could score on the candidates retrieval produced. If the live score is near
  that ceiling, every hour spent on the classifier is wasted and the work is in retrieval.
* the **miss taxonomy** splits the remaining loss into never-shortlisted, shortlisted but
  dropped, and false positives. Those three have disjoint fixes.

Recall is measured per view, not just for the union, so a view that costs an hour and adds
nothing can be deleted rather than kept out of caution.
"""
import numpy as np
import pandas as pd

from .lexical import VIEWS
from .paths import log


def true_pairs(s1, d, src):
    """(s1_pos, d_pos, src) for every ground-truth pair reachable in this block."""
    pos_of = pd.Series(np.arange(len(s1), dtype=np.int32), index=s1["entity_id"].to_numpy())
    owner = d["owner"].to_numpy()
    has = owner != ""
    sp = pos_of.reindex(owner[has]).to_numpy()
    keep = ~pd.isna(sp)
    out = pd.DataFrame({
        "s1_pos": sp[keep].astype(np.int32),
        "d_pos": np.flatnonzero(has)[keep].astype(np.int32),
    })
    out["src"] = np.int8(src)
    return out


def _mark(truth, cand, cols):
    """Left-join the candidate flags onto the truth table."""
    got = cand[["s1_pos", "d_pos"] + cols].copy()
    return truth.merge(got, on=["s1_pos", "d_pos"], how="left")


def recall_report(blocks, log_fn=log):
    """blocks: iterable of (country, src, n_s1, truth, cand). Prints recall per view."""
    view_names = [n for n, _, _ in VIEWS]
    agg = {}
    per_entity = {}
    for country, src, n_s1, truth, cand in blocks:
        cols = [f"rank_{n}" for n in view_names] + ["in_key", "in_skel"]
        cols = [c for c in cols if c in cand.columns]
        if "rank_dense" in cand.columns:
            cols.append("rank_dense")
        m = _mark(truth, cand, cols)
        found = m["s1_pos"].notna() & m[cols].notna().any(axis=1) if cols else m["s1_pos"].notna()
        hit = {}
        for n in view_names + (["dense"] if "rank_dense" in cols else []):
            c = f"rank_{n}"
            hit[n] = (m[c].notna() & (m[c] < 1e6)).to_numpy() if c in m.columns else np.zeros(len(m), bool)
        for n in ("in_key", "in_skel"):
            hit[n] = (m[n].fillna(0) > 0).to_numpy() if n in m.columns else np.zeros(len(m), bool)
        union = np.zeros(len(m), bool)
        for v in hit.values():
            union |= v
        a = agg.setdefault(country, {"n_true": 0, "union": 0, **{k: 0 for k in hit}})
        a["n_true"] += len(m)
        a["union"] += int(union.sum())
        for k, v in hit.items():
            a[k] += int(v.sum())
        pe = per_entity.setdefault(country, {"found": np.zeros(n_s1), "n": np.zeros(n_s1)})
        np.add.at(pe["found"], m["s1_pos"].to_numpy(np.int64), union.astype(np.float64))
        np.add.at(pe["n"], m["s1_pos"].to_numpy(np.int64), 1.0)
        log_fn(f"  {country} src={src}: {len(m):,} true pairs, union recall "
               f"{union.mean() if len(m) else 0:.4f}")

    total_true = sum(a["n_true"] for a in agg.values())
    total_hit = sum(a["union"] for a in agg.values())
    log_fn(f"PAIR RECALL (all countries) = {total_hit / max(total_true, 1):.4f}")
    for country, a in sorted(agg.items()):
        log_fn(f"  [{country}] union={a['union'] / max(a['n_true'], 1):.4f}  "
               + "  ".join(f"{k}={a[k] / max(a['n_true'], 1):.3f}"
                           for k in a if k not in ("n_true", "union")))
    ceilings = []
    for country, pe in sorted(per_entity.items()):
        n = pe["n"]
        r = np.where(n > 0, pe["found"] / np.maximum(n, 1), 0.0)
        oracle = np.where(n == 0, 1.0, np.where(r > 0, 1.25 * r / (0.25 + r), 0.0))
        ceilings.append(oracle)
        log_fn(f"  [{country}] oracle F0.5 ceiling = {oracle.mean():.4f}  "
               f"(entities fully recovered {float((r[n > 0] >= 1).mean() if (n > 0).any() else 0):.4f}, "
               f"zero hits {float((r[n > 0] == 0).mean() if (n > 0).any() else 0):.4f})")
    if ceilings:
        allc = np.concatenate(ceilings)
        log_fn(f"ORACLE macro F0.5 CEILING = {allc.mean():.4f}")
    return total_hit / max(total_true, 1)


def miss_taxonomy(rows, log_fn=log):
    """rows: DataFrame with country, src, stage columns for every true pair, plus false
    positives counted separately. Prints the three buckets and their breakdowns."""
    n = len(rows)
    if not n:
        return
    buckets = rows["bucket"].value_counts()
    log_fn("MISS TAXONOMY over " + f"{n:,} true pairs")
    for b, c in buckets.items():
        log_fn(f"  {b}: {c:,} ({c / n:.4f})")
    for key in ("country", "src", "b_native", "b_missing", "n_true_bin"):
        if key not in rows.columns:
            continue
        tab = rows.groupby(key)["bucket"].value_counts(normalize=True).unstack(fill_value=0.0)
        log_fn(f"  by {key}:")
        for idx, r in tab.iterrows():
            log_fn("    " + f"{idx}: " + "  ".join(f"{c}={v:.3f}" for c, v in r.items()))


def bin_n_true(n):
    return np.where(n >= 6, "6+", n.astype(str))
