"""L4 resolve: global one-owner assignment, then per-S1 expected-F0.5 selection.

**Why there is no Hungarian solver here.** The v3 plan proposed one, and on re-reading the
constraint it buys nothing. Ownership is one-sided: each S2/S3 record has at most one owner
(measured: 0 violations in 7,638,365 matched ids), while an S1 entity may own up to 11
records. Maximising the total of (probability - threshold) under a constraint that only
binds on the record side therefore *decomposes per record* -- each record independently
takes its best owner, which is exactly the argmax below. A matching solver would only be
needed if the per-S1 cap bound, and with a measured maximum cluster of 11 and a mean of
3.46 it effectively never does. The cap is still enforced, greedily by probability, and
logged when it fires so the assumption stays visible rather than assumed.

Selection is then per S1 and per country: expected F0.5 over calibrated probabilities,
including the empty list. That last part is what scores the 5.6% singletons -- under F0.5 a
correct empty list is worth a full 1.0, and a single false positive on a singleton takes it
to 0.0, so "predict nothing" has to be a candidate answer, not a fallback.

``miss_rate`` feeds the selector the probability mass of true matches retrieval never
retrieved. Without it the selector believes the candidate list is exhaustive and picks lists
that are systematically too short.
"""
import gc
import os

import numpy as np
import pandas as pd

from ..decision import select_expected_f05, select_threshold
from .paths import log

OWNER_KEY = ["src", "d_pos"]


def record_key(df):
    """One int64 key for (source, record position): d_pos is a position inside one source's
    shard, so positions collide across sources within a country."""
    return (df["src"].to_numpy(np.int64) << np.int64(32)) | df["d_pos"].to_numpy(np.int64)


def one_owner(pairs, margin=0.0, prob_col="prob"):
    """Keep, for each record, only its best-scoring S1 owner.

    ``margin`` drops records whose best two owners are within ``margin`` of each other: a
    contested record is a likely false merge, and under F0.5 a false positive costs twice
    what a miss does.
    """
    df = pairs.assign(_rk=record_key(pairs))
    df = df.sort_values(["_rk", prob_col], ascending=[True, False])
    first = ~df.duplicated("_rk")
    out = df[first]
    if margin > 0:
        second = df[~first].drop_duplicates("_rk").set_index("_rk")[prob_col]
        gap = out["_rk"].map(second).fillna(0.0).to_numpy(np.float32)
        out = out[(out[prob_col].to_numpy(np.float32) - gap) >= margin]
    return out.drop(columns=["_rk"])


def cap_per_s1(df, cap=11, prob_col="prob"):
    """No S1 may own more than the largest cluster ever observed in train."""
    sizes = df.groupby("s1_pos", sort=False)[prob_col].transform("size")
    if (sizes > cap).any():
        n = int((sizes > cap).sum())
        log(f"  per-S1 cap fired on {n:,} rows (cap={cap})")
        df = df.sort_values(["s1_pos", prob_col], ascending=[True, False])
        df = df.groupby("s1_pos", sort=False).head(cap)
    return df


def select(df, mode, thr, margin=0.0, miss_rate=0.0, cap=11):
    own = one_owner(df, margin=margin)
    own = cap_per_s1(own, cap=cap)
    if mode == "expected":
        return select_expected_f05(own, miss_rate=miss_rate)
    return select_threshold(own, thr)


def to_lists(sel, s1_ids, ids_by_src):
    """{s1 entity id: [record entity ids]} for one country."""
    out = {}
    sp = sel["s1_pos"].to_numpy()
    dp = sel["d_pos"].to_numpy()
    sc = sel["src"].to_numpy()
    for src in np.unique(sc):
        m = sc == src
        names = ids_by_src[int(src)][dp[m]]
        for s, nm in zip(sp[m], names):
            out.setdefault(s1_ids[s], []).append(nm)
    return out


def write_matching(path, s1_ids_all, matches):
    """One row per test S1 entity, in file order; empty list means "no match"."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1_ids_all:
            f.write(s)
            f.write("\t")
            got = matches.get(s)
            if got:
                f.write(",".join(got))
            f.write("\n")


def write_candidate_part(path, n_s1, s1_pos, d_pos, d_ids, block=200_000):
    """One line per S1 position (blank where there are no candidates), in position order.

    Written per (country, source) and merged later. Building the whole file's strings at
    once would hold hundreds of millions of ids in memory; blocking by S1 position keeps
    that bounded while the joins stay in pandas' native code, which is what makes this
    minutes instead of hours.
    """
    order = np.argsort(s1_pos, kind="stable")
    s1_pos = s1_pos[order]
    d_pos = d_pos[order]
    with open(path, "w", encoding="utf-8", newline="") as f:
        for lo in range(0, n_s1, block):
            hi = min(lo + block, n_s1)
            a, b = np.searchsorted(s1_pos, [lo, hi])
            lines = [""] * (hi - lo)
            if b > a:
                sub = pd.DataFrame({"s": s1_pos[a:b], "i": d_ids[d_pos[a:b]]})
                joined = sub.groupby("s", sort=True)["i"].agg(",".join)
                for s, v in joined.items():
                    lines[s - lo] = v
            f.write("\n".join(lines))
            f.write("\n")
            del lines
            gc.collect()


def merge_candidate_parts(out_path, per_country, append):
    """Stitch the per-(country, source) part files into candidate_pairs.tsv by streaming."""
    mode = "a" if append else "w"
    with open(out_path, mode, encoding="utf-8", newline="") as out:
        if not append:
            out.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_ids, parts in per_country:
            handles = [open(p, encoding="utf-8") for p in parts]
            try:
                for s in s1_ids:
                    vals = [h.readline().rstrip("\n") for h in handles]
                    out.write(s)
                    out.write("\t")
                    out.write(",".join(v for v in vals if v))
                    out.write("\n")
            finally:
                for h in handles:
                    h.close()
