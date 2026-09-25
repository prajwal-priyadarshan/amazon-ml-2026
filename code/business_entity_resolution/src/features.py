"""Pair features for the stage-1 GBDT. Deliberately country/language agnostic (no country feature)."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz import process

FUZZ = {
    "name_ratio": ("name_lat", fuzz.ratio),
    "name_tset": ("name_lat", fuzz.token_set_ratio),
    "name_tsort": ("name_lat", fuzz.token_sort_ratio),
    "name_part": ("name_lat", fuzz.partial_ratio),
    "core_tset": ("name_core", fuzz.token_set_ratio),
    "name_jw": ("name_lat", JaroWinkler.similarity),
    "addr_tset": ("addr_norm", fuzz.token_set_ratio),
    "addr_tsort": ("addr_norm", fuzz.token_sort_ratio),
    "addr_part": ("addr_norm", fuzz.partial_ratio),
}


def _pairwise(scorer, a, b):
    try:
        return process.cpdist(a, b, scorer=scorer, dtype=np.float32, workers=-1).astype(np.float32)
    except AttributeError:
        return np.array([scorer(x, y) for x, y in zip(a, b)], dtype=np.float32)


def _set_feats(na, nb, ca, cb):
    """Number-token and core-name-token features per pair (python loop; pairs are short)."""
    out = np.zeros((len(na), 9), np.float32)
    for i, (a, b, x, y) in enumerate(zip(na, nb, ca, cb)):
        sa, sb = set(a.split()), set(b.split())
        ta, tb = set(x.split()), set(y.split())
        common = len(ta & tb)
        out[i, 5] = float(bool(tb) and tb < ta)          # b is a truncation of a
        out[i, 6] = float(bool(ta) and ta < tb)          # b has extra words
        out[i, 7] = common / max(len(ta | tb), 1)
        out[i, 8] = common
        if not sa or not sb:
            out[i, :5] = (0, 0, -1, 0, -1)
            continue
        inter = len(sa & sb)
        big_a = [x_ for x_ in sa if len(x_) >= 2]
        big_b = [y_ for y_ in sb if len(y_) >= 2]
        diff = min((abs(int(p) - int(q)) for p in big_a for q in big_b), default=-1)
        lev = min((Levenshtein.distance(p, q) for p in big_a for q in big_b), default=-1)
        sub = any(p != q and (p.endswith(q) or q.endswith(p)) for p in sa for q in sb if len(p) > 1 and len(q) > 1)
        out[i, :5] = (inter, inter / len(sa | sb), diff, float(sub), lev)
    return out


def compute_features(cand, s1, dall, chunk=2_000_000):
    """cand: s1_pos, d_pos (+ retrieval columns). dall: concatenated S2+S3 frame with ``src`` column."""
    parts = []
    for s in range(0, len(cand), chunk):
        c = cand.iloc[s:s + chunk]
        a = s1.iloc[c.s1_pos.values]
        b = dall.iloc[c.d_pos.values]
        f = {}
        for name, (col, scorer) in FUZZ.items():
            f[name] = _pairwise(scorer, a[col].tolist(), b[col].tolist())
        nf = _set_feats(a["nums"].tolist(), b["nums"].tolist(), a["name_core"].tolist(), b["name_core"].tolist())
        for j, n in enumerate(("num_inter", "num_jac", "num_mindiff", "num_sub", "num_lev",
                               "core_sub", "core_sup", "core_jac", "core_common")):
            f[n] = nf[:, j]
        f["legal_eq"] = (a["legal"].values == b["legal"].values).astype("int8")
        f["key_eq"] = (a["name_key"].values == b["name_key"].values).astype("int8")
        f["len_ratio"] = (a["name_lat"].str.len().values + 1) / (b["name_lat"].str.len().values + 1)
        f["ntok_a"] = a["name_core"].str.count(" ").values + 1
        f["ntok_b"] = b["name_core"].str.count(" ").values + 1
        f["a_missing"] = a["addr_missing"].values
        f["b_missing"] = b["addr_missing"].values
        f["b_native"] = b["native"].values
        f["a_native"] = a["native"].values
        f["src"] = b["src"].values
        parts.append(pd.DataFrame(f, index=c.index))
    feats = pd.concat(parts)
    out = pd.concat([cand, feats], axis=1)
    return add_group_features(out)


def add_group_features(df):
    """Competition features among the S1 rows that retrieved the same S2/S3 record."""
    df["sim"] = df["cos_addr"] + df["cos_name"]
    o = df.sort_values(["d_pos", "sim"], ascending=[True, False])
    r = o.groupby("d_pos").cumcount()
    best = o["sim"].where(r == 0).groupby(o["d_pos"]).transform("max")
    second = o["sim"].where(r == 1).groupby(o["d_pos"]).transform("max").fillna(0.0)
    df["rec_best"] = best.reindex(df.index)
    df["rec_n"] = o.groupby("d_pos")["sim"].transform("size").reindex(df.index)
    margin = pd.Series(np.where(r == 0, o["sim"] - second, o["sim"] - best), index=o.index)
    df["rec_margin"] = margin.reindex(df.index)
    df["s1_rank"] = df.groupby(["s1_pos", "src"])["sim"].rank(ascending=False, method="first")
    df["s1_best"] = df.groupby(["s1_pos", "src"])["sim"].transform("max")
    df["s1_gap"] = df["sim"] - df["s1_best"]
    return df


FEATURE_COLS = (
    ["rank_addr", "rank_name", "rank_addrc", "rank_rev", "in_key", "cos_addr", "cos_name", "cos_addrc", "rrf"]
    + list(FUZZ)
    + ["num_inter", "num_jac", "num_mindiff", "num_sub", "num_lev", "core_sub", "core_sup", "core_jac",
       "core_common", "legal_eq", "key_eq", "len_ratio", "ntok_a", "ntok_b",
       "a_missing", "b_missing", "a_native", "b_native", "src",
       "rec_best", "rec_n", "rec_margin", "s1_rank", "s1_best", "s1_gap"]
)

STAGE2_COLS = ["p1", "s1_rank_p", "s1_best_p", "s1_sum_p", "s1_n_hi", "rec_n_hi", "rec_best_p",
               "rec_margin_p", "s1_gap_p"]
FEATURE_COLS2 = FEATURE_COLS + STAGE2_COLS


def add_stage2_features(df, p1):
    """Competition features computed from first-stage probabilities (out-of-sample for stage-2 training rows)."""
    df["p1"] = p1.astype(np.float32)
    df["s1_rank_p"] = df.groupby(["s1_pos", "src"])["p1"].rank(ascending=False, method="first")
    df["s1_best_p"] = df.groupby("s1_pos")["p1"].transform("max")
    df["s1_sum_p"] = df.groupby("s1_pos")["p1"].transform("sum")
    df["s1_gap_p"] = df["p1"] - df.groupby(["s1_pos", "src"])["p1"].transform("max")
    hi = (df["p1"] > 0.5).astype("int8")
    df["s1_n_hi"] = hi.groupby(df["s1_pos"]).transform("sum")
    df["rec_n_hi"] = hi.groupby(df["d_pos"]).transform("sum")
    o = df.sort_values(["d_pos", "p1"], ascending=[True, False])
    r = o.groupby("d_pos").cumcount()
    best = o["p1"].where(r == 0).groupby(o["d_pos"]).transform("max")
    second = o["p1"].where(r == 1).groupby(o["d_pos"]).transform("max").fillna(0.0)
    df["rec_best_p"] = best.reindex(df.index)
    df["rec_margin_p"] = pd.Series(np.where(r == 0, o["p1"] - second, o["p1"] - best), index=o.index).reindex(df.index)
    return df
