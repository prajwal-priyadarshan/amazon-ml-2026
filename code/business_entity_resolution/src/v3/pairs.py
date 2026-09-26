"""Candidate bookkeeping: union of views, labels, competition features, design matrices.

The v2 pair-feature block (rapidfuzz ratios, token-set containment, digit agreement,
legal-suffix conflict) is reused verbatim from ``src.features``. It is measured, it is
country-agnostic, and none of the v3 changes touch it -- v3 adds views and a judge around
it, it does not replace it.

Every column here is float32 by construction. pandas upcasts to float64 on the slightest
provocation, which doubles peak RSS on a 30M-row block for no benefit.
"""
import gc

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from ..features import (PAIR_COLS, STAGE2_EXTRA, add_stage2_features,
                        compute_features)
from .lexical import LEX_COLS

DENSE_COLS = ["cos_dense", "rank_dense"]
# Name similarity after glued-name segmentation. The v2 feature block compares `name_core`,
# where `akshayagases` and `akshaya gases` share no tokens at all; these two compare the
# segmented form, which is the only view in which that pair agrees.
SEG_COLS = ["seg_tset", "seg_key_eq"]
RETRIEVAL_COLS = LEX_COLS + DENSE_COLS
GROUP_COLS = ["sim", "rec_best", "rec_n", "rec_margin", "s1_rank", "s1_best", "s1_gap"]
SIB_COLS = ["sib_max", "sib_mean", "sib_cnt"]
CE_COLS = ["ce_logit", "ce_run"]
# Cross-source competition. Blocks are processed one (country, source) at a time, so the
# within-block stage-2 features cannot see that the same S1 already has a strong match in
# the other source -- which is exactly the evidence that a cluster is real (the measured
# shapes are dominated by 1-1, 1-2, 2-1, 2-2). These two columns carry it back in.
OTHER_COLS = ["oth_best", "oth_cnt"]

PRUNE_COLS = RETRIEVAL_COLS + PAIR_COLS + SEG_COLS + GROUP_COLS
STACK_COLS = PRUNE_COLS + STAGE2_EXTRA + SIB_COLS + CE_COLS + OTHER_COLS

LEAN = ["s1_pos", "d_pos"]


def union(lex, dense, keep):
    """Fuse the lexical union with the dense top-k.

    An outer join, not a filter: the dense view exists precisely to add the pairs no
    lexical view could reach (transliteration, initialisms, renamed businesses), and the
    lexical views exist to add the pairs where the name is a random string and only the
    address agrees. Capping happens after fusion so neither side is starved.
    """
    if dense is None or len(dense) == 0:
        cand = lex.copy()
        cand["cos_dense"] = np.float32(0.0)
        cand["rank_dense"] = np.float32(1e6)
    elif lex is None or len(lex) == 0:
        cand = dense.copy()
        for c in LEX_COLS:
            cand[c] = np.float32(1e6 if c.startswith("rank_") else 0.0)
    else:
        cand = lex.merge(dense, on=LEAN, how="outer")
        for c in LEX_COLS:
            cand[c] = cand[c].fillna(1e6 if c.startswith("rank_") else 0.0)
        cand["cos_dense"] = cand["cos_dense"].fillna(0.0)
        cand["rank_dense"] = cand["rank_dense"].fillna(1e6)

    cand["fused"] = (cand["rrf"].astype(np.float32)
                     + (1.0 / (60.0 + cand["rank_dense"].astype(np.float32))))
    if keep:
        cand = cand.sort_values(["s1_pos", "fused"], ascending=[True, False])
        cand = cand.groupby("s1_pos", sort=False).head(keep)
    cand = cand.drop(columns=["fused"]).reset_index(drop=True)
    cand["s1_pos"] = cand["s1_pos"].astype(np.int32)
    cand["d_pos"] = cand["d_pos"].astype(np.int32)
    for c in RETRIEVAL_COLS:
        cand[c] = cand[c].astype(np.float32)
    return cand


def label(cand, s1, d):
    """1 when the record's ground-truth owner is this S1 entity.

    ``owner`` was resolved once at prep time, so this is a vectorised string comparison
    instead of 7.6M dictionary lookups per block.
    """
    own = d["owner"].to_numpy()[cand["d_pos"].to_numpy()]
    mine = s1["entity_id"].to_numpy()[cand["s1_pos"].to_numpy()]
    return (own == mine).astype("int8")


def add_group_features(cand):
    """Competition on the retrieval side: who else wants this record, and how strongly.

    A record that is the best candidate of twelve different S1 entities is probably a
    generic string; a record that only one S1 retrieved at rank 1 is probably its match.
    That contrast is invisible to a per-pair model.
    """
    cand["sim"] = (cand["cos_dense"] + cand["sc_addr"] + cand["sc_name"]
                   + cand["sc_nword"] + cand["sc_skel"]).astype(np.float32)
    o = cand.sort_values(["d_pos", "sim"], ascending=[True, False])
    r = o.groupby("d_pos").cumcount()
    best = o["sim"].where(r == 0).groupby(o["d_pos"]).transform("max")
    second = o["sim"].where(r == 1).groupby(o["d_pos"]).transform("max").fillna(0.0)
    cand["rec_best"] = best.reindex(cand.index)
    cand["rec_n"] = o.groupby("d_pos")["sim"].transform("size").reindex(cand.index)
    margin = pd.Series(np.where(r == 0, o["sim"] - second, o["sim"] - best), index=o.index)
    cand["rec_margin"] = margin.reindex(cand.index)
    g = cand.groupby("s1_pos")["sim"]
    cand["s1_rank"] = g.rank(ascending=False, method="first")
    cand["s1_best"] = g.transform("max")
    cand["s1_gap"] = cand["sim"] - cand["s1_best"]
    for c in GROUP_COLS:
        cand[c] = cand[c].astype(np.float32)
    return cand


def add_stage2(cand, prob_col="prob1"):
    """v2's stage-2 competition features, computed on stage-1 probabilities."""
    return add_stage2_features(cand, prob_col=prob_col)


def fill_missing(cand, cols):
    for c in cols:
        if c not in cand.columns:
            cand[c] = np.float32(0.0)
    return cand


def seg_features(cand, s1, d):
    """seg_tset / seg_key_eq for one chunk of pairs."""
    a_seg = s1["name_seg"].to_numpy()[cand["s1_pos"].to_numpy()]
    b_seg = d["name_seg"].to_numpy()[cand["d_pos"].to_numpy()]
    a_key = s1["seg_key"].to_numpy()[cand["s1_pos"].to_numpy()]
    b_key = d["seg_key"].to_numpy()[cand["d_pos"].to_numpy()]
    try:
        tset = process.cpdist(list(a_seg), list(b_seg), scorer=fuzz.token_set_ratio,
                              dtype=np.float32, workers=-1).astype(np.float32)
    except AttributeError:  # older rapidfuzz without cpdist
        tset = np.array([fuzz.token_set_ratio(x, y) for x, y in zip(a_seg, b_seg)], np.float32)
    return {"seg_tset": tset, "seg_key_eq": (a_key == b_key).astype(np.float32)}


def matrix(cand, s1, d, cols, chunk=1_000_000):
    """float32 design matrix for ``cols`` over an arbitrary candidate frame."""
    have_pair = [c for c in cols if c in PAIR_COLS]
    have_seg = [c for c in cols if c in SEG_COLS]
    out = np.empty((len(cand), len(cols)), np.float32)
    pos = {c: i for i, c in enumerate(cols)}
    own = [c for c in cols if c not in PAIR_COLS and c not in SEG_COLS]
    if own:
        own_vals = cand[own].to_numpy(np.float32, copy=False) if len(cand) else np.zeros((0, len(own)), np.float32)
        for j, c in enumerate(own):
            out[:, pos[c]] = own_vals[:, j]
    for s in range(0, len(cand), chunk):
        e = min(s + chunk, len(cand))
        blk = cand.iloc[s:e]
        pf = compute_features(blk, s1, d, chunk=chunk)
        for c in have_pair:
            out[s:e, pos[c]] = pf[c].to_numpy(np.float32, copy=False)
        del pf
        if have_seg:
            sf = seg_features(blk, s1, d)
            for c in have_seg:
                out[s:e, pos[c]] = sf[c]
            del sf
        del blk
        gc.collect()
    return out
