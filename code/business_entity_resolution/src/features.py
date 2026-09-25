"""Pair features for the stage-1 GBDT. Deliberately country/language agnostic (no country feature)."""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
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


def _num_feats(na, nb):
    out = np.zeros((len(na), 4), np.float32)
    for i, (a, b) in enumerate(zip(na, nb)):
        sa, sb = set(a.split()), set(b.split())
        if not sa or not sb:
            out[i] = (0, 0, -1, 0)
            continue
        inter = len(sa & sb)
        big_a = [int(x) for x in sa if len(x) >= 2]
        big_b = [int(x) for x in sb if len(x) >= 2]
        diff = min((abs(x - y) for x in big_a for y in big_b), default=-1)
        sub = any(x != y and (x.endswith(y) or y.endswith(x)) for x in sa for y in sb if len(x) > 1 and len(y) > 1)
        out[i] = (inter, inter / len(sa | sb), diff, float(sub))
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
        nf = _num_feats(a["nums"].tolist(), b["nums"].tolist())
        for j, n in enumerate(("num_inter", "num_jac", "num_mindiff", "num_sub")):
            f[n] = nf[:, j]
        f["legal_eq"] = (a["legal"].values == b["legal"].values).astype("int8")
        f["key_eq"] = (a["name_key"].values == b["name_key"].values).astype("int8")
        f["len_ratio"] = (a["name_lat"].str.len().values + 1) / (b["name_lat"].str.len().values + 1)
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
    ["rank_addr", "rank_name", "in_key", "cos_addr", "cos_name", "rrf"] + list(FUZZ)
    + ["num_inter", "num_jac", "num_mindiff", "num_sub", "legal_eq", "key_eq", "len_ratio",
       "a_missing", "b_missing", "a_native", "b_native", "src",
       "rec_best", "rec_n", "rec_margin", "s1_rank", "s1_best", "s1_gap"]
)
