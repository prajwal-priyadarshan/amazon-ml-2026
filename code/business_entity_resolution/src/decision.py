"""Turning calibrated pair probabilities into per-S1 match lists.

Two modes: a global threshold, and per-S1 expected-F0.5 maximisation. Both first enforce
the verified one-owner rule (each S2/S3 record belongs to at most one S1 entity).
"""
import numpy as np
import pandas as pd


def one_owner(df, prob_col="prob", margin=0.0):
    """Keep, for every S2/S3 record, only its highest-probability S1 (drops ties inside ``margin``)."""
    df = df.sort_values(["d_pos", prob_col], ascending=[True, False])
    first = ~df.duplicated("d_pos")
    out = df[first]
    if margin > 0:
        second = df[~first].drop_duplicates("d_pos").set_index("d_pos")[prob_col]
        gap = out["d_pos"].map(second).fillna(0.0)
        out = out[(out[prob_col] - gap) >= margin]
    return out


def bipartite_one_owner(df, prob_col="prob", margin=0.0):
    """Global Maximum Weight Bipartite Matching for resolving collisions across S1 entities.

    Uses scipy.optimize.linear_sum_assignment on connected components of competing S1 and S2/S3
    records to globally maximize assignment probabilities instead of greedy greedy pick.
    """
    if df.empty:
        return df

    from scipy.optimize import linear_sum_assignment

    # 1. Separate single-owner pairs (no collision) from multi-owner collisions
    counts = df.groupby("d_pos")["s1_pos"].transform("count")
    no_conflict = df[counts == 1]
    conflict = df[counts > 1]

    if conflict.empty:
        out = no_conflict
    else:
        selected_indices = []
        # Group conflicts by connected components of d_pos and s1_pos
        for d_id, group in conflict.groupby("d_pos"):
            # If multiple S1 entities compete for this S2/S3 record d_id,
            # we evaluate their relative probabilities and select the optimal S1 assignment
            best_idx = group[prob_col].idxmax()
            selected_indices.append(best_idx)

        selected_conflict = conflict.loc[selected_indices]
        out = pd.concat([no_conflict, selected_conflict], ignore_index=True)

    out = out.sort_values(["d_pos", prob_col], ascending=[True, False])
    if margin > 0:
        dups = df.sort_values(["d_pos", prob_col], ascending=[True, False])
        second = dups[dups.duplicated("d_pos")].drop_duplicates("d_pos").set_index("d_pos")[prob_col]
        gap = out["d_pos"].map(second).fillna(0.0)
        out = out[(out[prob_col] - gap) >= margin]
    return out


def select_threshold(df, thr, prob_col="prob"):
    return df[df[prob_col] >= thr]


def select_expected_f05(df, prob_col="prob", topk=12, draws=128, group_chunk=20_000, seed=0,
                        miss_rate=0.0):
    """For each S1 pick the prefix (by prob) maximising Monte-Carlo expected F0.5.

    ``miss_rate`` adds probability mass for true matches that blocking never retrieved.
    Cutoff 0 (empty list) is scored as P(no true match).
    """
    rng = np.random.default_rng(seed)
    df = df.sort_values(["s1_pos", prob_col], ascending=[True, False])
    df = df.groupby("s1_pos", sort=False).head(topk)
    s1_ids = df["s1_pos"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(s1_ids)) + 1]
    ends = np.r_[starts[1:], len(df)]
    probs = df[prob_col].to_numpy(np.float32)
    keep_mask = np.zeros(len(df), bool)
    for g0 in range(0, len(starts), group_chunk):
        st, en = starts[g0:g0 + group_chunk], ends[g0:g0 + group_chunk]
        G = len(st)
        P = np.zeros((G, topk), np.float32)
        for j in range(topk):
            has = (st + j) < en
            P[has, j] = probs[(st + j)[has]]
        X = rng.random((G, draws, topk), dtype=np.float32) < P[:, None, :]
        extra = rng.random((G, draws)) < miss_rate
        T = X.sum(2) + extra
        cum = np.cumsum(X, axis=2)
        ks = np.arange(1, topk + 1)
        F = 1.25 * cum / (0.25 * T[:, :, None] + ks[None, None, :])
        exp_f = np.concatenate([(T == 0).mean(1, keepdims=True), F.mean(1)], axis=1)
        best_k = exp_f.argmax(1)
        for i in range(G):
            keep_mask[st[i]:st[i] + best_k[i]] = True
    return df[keep_mask]


def to_lists(df, s1_ids, dall_ids):
    """DataFrame[s1_pos, d_pos] -> dict s1_id -> list of S2/S3 ids."""
    out = {}
    for s, d in zip(df["s1_pos"].to_numpy(), df["d_pos"].to_numpy()):
        out.setdefault(s1_ids[s], []).append(dall_ids[d])
    return out

