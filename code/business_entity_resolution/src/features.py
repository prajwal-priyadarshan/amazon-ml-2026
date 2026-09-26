"""Pair features for the GBDT. Deliberately country/language agnostic (no country feature,
no state or postcode logic), so the model transfers to France, which never appears in train."""
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
    "core_ratio": ("name_core", fuzz.ratio),
    "skel_tset": ("name_skel", fuzz.token_set_ratio),
    "name_jw": ("name_lat", JaroWinkler.similarity),
    "addr_tset": ("addr_norm", fuzz.token_set_ratio),
    "addr_tsort": ("addr_norm", fuzz.token_sort_ratio),
    "addr_part": ("addr_norm", fuzz.partial_ratio),
}

LOOP_COLS = ("tok_inter", "tok_jac", "tok_contain", "tok_extra_a", "tok_extra_b",
             "adr_inter", "adr_jac", "adr_contain", "adr_extra",
             "num_inter", "num_jac", "num_mindiff", "num_sub")


def _pairwise(scorer, a, b):
    try:
        return process.cpdist(a, b, scorer=scorer, dtype=np.float32, workers=-1).astype(np.float32)
    except AttributeError:
        return np.array([scorer(x, y) for x, y in zip(a, b)], dtype=np.float32)


def _loop_feats(core_a, core_b, adr_a, adr_b, num_a, num_b):
    """Token-set evidence in one pass: overlap, containment (the truncation signal) and
    extra-word counts for names and addresses, plus street/house number agreement."""
    out = np.zeros((len(core_a), len(LOOP_COLS)), np.float32)
    for i in range(len(core_a)):
        sa, sb = set(core_a[i].split()), set(core_b[i].split())
        if sa and sb:
            inter = len(sa & sb)
            out[i, 0] = inter
            out[i, 1] = inter / len(sa | sb)
            out[i, 2] = inter / min(len(sa), len(sb))
            out[i, 3] = len(sa - sb)
            out[i, 4] = len(sb - sa)
        else:
            out[i, 2] = -1.0
        aa, ab = set(adr_a[i].split()), set(adr_b[i].split())
        if aa and ab:
            inter = len(aa & ab)
            out[i, 5] = inter
            out[i, 6] = inter / len(aa | ab)
            out[i, 7] = inter / min(len(aa), len(ab))
            out[i, 8] = len(aa ^ ab)
        else:
            out[i, 7] = -1.0
        na, nb = set(num_a[i].split()), set(num_b[i].split())
        if na and nb:
            inter = len(na & nb)
            out[i, 9] = inter
            out[i, 10] = inter / len(na | nb)
            big_a = [int(x) for x in na if len(x) >= 2]
            big_b = [int(x) for x in nb if len(x) >= 2]
            out[i, 11] = min((abs(x - y) for x in big_a for y in big_b), default=-1)
            out[i, 12] = float(any(x != y and (x.endswith(y) or y.endswith(x))
                                   for x in na for y in nb if len(x) > 1 and len(y) > 1))
        else:
            out[i, 11] = -1.0
    return out


def compute_features(cand, s1, dall, chunk=2_000_000):
    """cand: s1_pos, d_pos (+ retrieval columns). dall: concatenated S2+S3 frame with ``src``.
    Group features are added separately, on the lean frame, so this expensive per-pair pass
    can be chunked without breaking group statistics."""
    parts = []
    for s in range(0, len(cand), chunk):
        c = cand.iloc[s:s + chunk]
        a = s1.iloc[c.s1_pos.values]
        b = dall.iloc[c.d_pos.values]
        f = {}
        for name, (col, scorer) in FUZZ.items():
            f[name] = _pairwise(scorer, a[col].tolist(), b[col].tolist())
        lf = _loop_feats(a["name_core"].tolist(), b["name_core"].tolist(),
                         a["addr_norm"].tolist(), b["addr_norm"].tolist(),
                         a["nums"].tolist(), b["nums"].tolist())
        for j, n in enumerate(LOOP_COLS):
            f[n] = lf[:, j]
        la, lb = a["legal"].values, b["legal"].values
        f["legal_eq"] = (la == lb).astype("int8")
        # Two different explicit legal forms are evidence *against* a match (Foo Inc vs Foo LLC).
        f["legal_conflict"] = ((la != lb) & (la != "") & (lb != "")).astype("int8")
        f["key_eq"] = (a["name_key"].values == b["name_key"].values).astype("int8")
        f["skel_eq"] = (a["name_skel"].values == b["name_skel"].values).astype("int8")
        alen = a["name_lat"].str.len().values.astype(np.float32)
        blen = b["name_lat"].str.len().values.astype(np.float32)
        f["len_ratio"] = (alen + 1) / (blen + 1)
        f["len_absdiff"] = np.abs(alen - blen)
        f["a_missing"] = a["addr_missing"].values
        f["b_missing"] = b["addr_missing"].values
        f["b_native"] = b["native"].values
        f["a_native"] = a["native"].values
        f["src"] = b["src"].values
        parts.append(pd.DataFrame(f, index=c.index))
    return pd.concat(parts)


def add_group_features(df):
    """Retrieval-side competition among the S1 rows that retrieved the same S2/S3 record.
    Runs on the lean candidate frame (no fuzzy columns needed)."""
    emb_sim = df["cos_emb"] if "cos_emb" in df.columns else 0.0
    df["sim"] = df["cos_addr"] + df["cos_name"] + df["cos_nword"] + df["cos_skel"] + emb_sim
    o = df.sort_values(["d_pos", "sim"], ascending=[True, False])
    r = o.groupby("d_pos").cumcount()
    best = o["sim"].where(r == 0).groupby(o["d_pos"]).transform("max")
    second = o["sim"].where(r == 1).groupby(o["d_pos"]).transform("max").fillna(0.0)
    df["rec_best"] = best.reindex(df.index)
    df["rec_n"] = o.groupby("d_pos")["sim"].transform("size").reindex(df.index)
    margin = pd.Series(np.where(r == 0, o["sim"] - second, o["sim"] - best), index=o.index)
    df["rec_margin"] = margin.reindex(df.index)
    g = df.groupby(["s1_pos", "src"])["sim"]
    df["s1_rank"] = g.rank(ascending=False, method="first")
    df["s1_best"] = g.transform("max")
    df["s1_gap"] = df["sim"] - df["s1_best"]
    return df


def add_stage2_features(df, prob_col="prob1"):
    """Re-describe every pair by how its stage-1 probability compares with the other
    candidates competing for the same S1 and for the same S2/S3 record. This is what lets
    the model conclude 'every candidate for this entity is weak, so predict nothing',
    instead of judging each pair in isolation."""
    p = df[prob_col]
    df["p_strong"] = (p >= 0.5).astype(np.float32)
    g1 = df.groupby(["s1_pos", "src"])[prob_col]
    df["p_rank"] = g1.rank(ascending=False, method="first").astype(np.float32)
    df["p_best"] = g1.transform("max").astype(np.float32)
    df["p_gap"] = (p - df["p_best"]).astype(np.float32)
    df["p_sum"] = g1.transform("sum").astype(np.float32)
    df["p_cnt"] = df.groupby(["s1_pos", "src"])["p_strong"].transform("sum").astype(np.float32)
    ga = df.groupby("s1_pos")[prob_col]
    df["pa_best"] = ga.transform("max").astype(np.float32)
    df["pa_sum"] = ga.transform("sum").astype(np.float32)
    df["pa_cnt"] = df.groupby("s1_pos")["p_strong"].transform("sum").astype(np.float32)
    df["pa_gap"] = (p - df["pa_best"]).astype(np.float32)
    gd = df.groupby("d_pos")[prob_col]
    df["d_best"] = gd.transform("max").astype(np.float32)
    df["d_gap"] = (p - df["d_best"]).astype(np.float32)
    df["d_n"] = gd.transform("size").astype(np.float32)
    df["d_cnt"] = df.groupby("d_pos")["p_strong"].transform("sum").astype(np.float32)
    return df


RETRIEVAL_COLS = ["rank_addr", "rank_name", "rank_nword", "rank_skel", "rank_emb", "in_key", "in_skel",
                  "cos_addr", "cos_name", "cos_nword", "cos_skel", "cos_emb", "rrf"]
GROUP_COLS = ["sim", "rec_best", "rec_n", "rec_margin", "s1_rank", "s1_best", "s1_gap"]
PAIR_COLS = (list(FUZZ) + list(LOOP_COLS)
             + ["legal_eq", "legal_conflict", "key_eq", "skel_eq", "len_ratio", "len_absdiff",
                "a_missing", "b_missing", "a_native", "b_native", "src"])

FEATURE_COLS = RETRIEVAL_COLS + PAIR_COLS + GROUP_COLS
STAGE2_EXTRA = ["prob1", "p_rank", "p_best", "p_gap", "p_sum", "p_cnt",
                "pa_best", "pa_sum", "pa_cnt", "pa_gap", "d_best", "d_gap", "d_n", "d_cnt"]
STAGE2_COLS = FEATURE_COLS + STAGE2_EXTRA
