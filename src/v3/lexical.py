"""L1 sparse retrieval: five complementary views, sharded so peak memory is O(shard).

Why sharded. v2 fitted one TF-IDF index over a whole (country, source) pool and handed it
to a process pool; joblib memory-maps dense arrays but pickles scipy sparse matrices whole,
so every worker got a private multi-GB copy. Three changes remove that, all measured in
docs/v2_build_plan.md 0.6:

1. **Threads, not processes.** scipy's sparse matmul releases the GIL, so one copy of the
   index is shared. Measured 3.2x faster than sequential *and* flat memory.
2. **Document-side shards with a frozen vocabulary.** The vectoriser is fitted once on a
   sample, then only ``transform`` runs per shard. Merging per-shard top-k lists is exact,
   not an approximation -- but only if the vocabulary and IDF are identical across shards,
   which is why the fit happens once and is reused.
3. **A running top-k instead of concatenating shard results.** Peak is
   n_s1 x k x 8 bytes per view, and views run one at a time.

What is deliberately *not* computed here: an exact cosine for every union pair in every
view. Filling those needs a second full transform pass over the corpus, for a feature the
GBDT can largely read off the per-view rank plus the score-where-retrieved. The one exact,
dense-valued similarity every union pair does get is the bi-encoder cosine from
``dense.py``, which is computed from the memmap for arbitrary pairs at negligible cost.
"""
import gc

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .paths import log

# (view name, text column, analyzer kind)
VIEWS = (
    ("addr", "addr_norm", "word"),
    ("name", "name_lat", "char"),
    # The word view reads the *segmented* name: it is the view that a glued name defeats
    # outright, and segmentation only ever changes tokens that were glued.
    ("nword", "name_seg", "word1"),
    ("skel", "name_skel", "char"),
)
KEY_VIEWS = (("in_key", "name_key"), ("in_skel", "name_skel"), ("in_seg", "seg_key"))
RRF_K = 60


def _vectorizer(kind, cap):
    if kind == "word":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+",
                               min_df=1, max_df=cap, sublinear_tf=True, dtype=np.float32)
    if kind == "word1":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 1), token_pattern=r"\S+",
                               min_df=1, max_df=cap, sublinear_tf=True, dtype=np.float32)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=cap,
                           sublinear_tf=True, dtype=np.float32)


def _topk_block(Q, DT, lo, hi, k, offset):
    """Top-k documents for query rows [lo, hi) against one document shard."""
    P = (Q[lo:hi] @ DT).tocsr()
    n = hi - lo
    idx = np.full((n, k), -1, np.int32)
    sc = np.zeros((n, k), np.float32)
    ip, ind, dat = P.indptr, P.indices, P.data
    for r in range(n):
        a, b = ip[r], ip[r + 1]
        if a == b:
            continue
        d = dat[a:b]
        sel = np.argpartition(-d, k - 1)[:k] if b - a > k else np.arange(b - a)
        sel = sel[np.argsort(-d[sel])]
        idx[r, :len(sel)] = ind[a:b][sel] + offset
        sc[r, :len(sel)] = d[sel]
    return lo, idx, sc


def _merge(cur_idx, cur_sc, lo, new_idx, new_sc, k):
    """Fold one shard's top-k into the running top-k for the same query rows."""
    hi = lo + len(new_idx)
    both_i = np.concatenate([cur_idx[lo:hi], new_idx], axis=1)
    both_s = np.concatenate([cur_sc[lo:hi], new_sc], axis=1)
    order = np.argsort(-both_s, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(both_i))[:, None]
    cur_idx[lo:hi] = both_i[rows, order]
    cur_sc[lo:hi] = both_s[rows, order]


def _spans(n, size):
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def _run_view(name, col, kind, s1, d, k, cfg):
    """Running top-k for one view over every document shard."""
    n_q = len(s1)
    dtext = d[col].to_numpy()
    fit_n = min(cfg.vocab_sample, len(d))
    if fit_n < len(d):
        rng = np.random.default_rng(0)
        fit_rows = dtext[rng.choice(len(d), fit_n, replace=False)]
    else:
        fit_rows = dtext
    # max_df as an integer counts documents *in the fitted sample*, so the cap has to be
    # scaled to the sample and not to the full pool -- otherwise it is looser by exactly the
    # sampling ratio, and the most common n-grams survive to blow up the score matrix.
    cap = max(50, int(cfg.max_df_frac * fit_n))
    vec = _vectorizer(kind, cap)
    vec.fit(fit_rows)
    del fit_rows
    Q = sparse.csr_matrix(vec.transform(s1[col].to_numpy()))
    log(f"    view {name}: vocab={len(vec.vocabulary_):,} Q_nnz={Q.nnz:,}")

    idx = np.full((n_q, k), -1, np.int32)
    sc = np.zeros((n_q, k), np.float32)
    for s, e in _spans(len(d), cfg.doc_shard):
        DT = sparse.csr_matrix(vec.transform(dtext[s:e]).T)
        gc.collect()
        res = Parallel(n_jobs=cfg.threads, prefer="threads")(
            delayed(_topk_block)(Q, DT, lo, hi, min(k, e - s), s) for lo, hi in _spans(n_q, cfg.q_chunk))
        for lo, ni, ns in res:
            _merge(idx, sc, lo, ni, ns, k)
        del DT, res
        gc.collect()
    del Q, vec, dtext
    gc.collect()
    return idx, sc


def _pairs_from_topk(idx, sc, name):
    n_q, k = idx.shape
    keep = idx.ravel() >= 0
    q = np.repeat(np.arange(n_q, dtype=np.int32), k)[keep]
    return pd.DataFrame({
        "s1_pos": q,
        "d_pos": idx.ravel()[keep],
        f"rank_{name}": np.tile(np.arange(1, k + 1, dtype=np.float32), n_q)[keep],
        f"sc_{name}": sc.ravel()[keep],
    })


def _key_pairs(s1, d, col, flag, cap):
    """Exact-key hash join. Bidirectional by construction, so it recovers pairs that
    neither side's top-k would surface (an exact rare name buried under 30 lookalikes)."""
    sizes = d[col].value_counts()
    ok = sizes[sizes <= cap].index
    dd = pd.DataFrame({col: d[col].to_numpy(), "d_pos": np.arange(len(d), dtype=np.int32)})
    dd = dd[dd[col].isin(ok) & (dd[col] != "")]
    ss = pd.DataFrame({col: s1[col].to_numpy(), "s1_pos": np.arange(len(s1), dtype=np.int32)})
    m = ss.merge(dd, on=col)[["s1_pos", "d_pos"]]
    m[flag] = np.int8(1)
    return m


def build(cfg, s1, d):
    """Union of the five views for one (country, source) block, capped per S1 row."""
    frames = []
    for name, col, kind in VIEWS:
        idx, sc = _run_view(name, col, kind, s1, d, cfg.k, cfg)
        frames.append(_pairs_from_topk(idx, sc, name))
        del idx, sc
        gc.collect()
    for flag, col in KEY_VIEWS:
        frames.append(_key_pairs(s1, d, col, flag, cfg.key_cap))

    cand = frames[0]
    for f in frames[1:]:
        cand = cand.merge(f, on=["s1_pos", "d_pos"], how="outer")
    del frames
    gc.collect()

    for flag, _ in KEY_VIEWS:
        cand[flag] = cand[flag].fillna(0).astype("int8")
    rrf = 0.0
    nv = 0
    for name, _, _ in VIEWS:
        r = cand[f"rank_{name}"].fillna(1e6)
        cand[f"rank_{name}"] = r.astype(np.float32)
        cand[f"sc_{name}"] = cand[f"sc_{name}"].fillna(0.0).astype(np.float32)
        rrf = rrf + 1.0 / (RRF_K + r)
        nv = nv + (r < 1e6).astype("int8")
    cand["n_views"] = nv
    cand["rrf"] = (rrf + 0.01 * cand["in_key"] + 0.01 * cand["in_skel"]
                   + 0.01 * cand["in_seg"]).astype(np.float32)

    if cfg.keep:
        cand = cand.sort_values(["s1_pos", "rrf"], ascending=[True, False])
        cand = cand.groupby("s1_pos", sort=False).head(cfg.keep)
    cand = cand.sort_values(["s1_pos", "d_pos"]).reset_index(drop=True)
    cand["s1_pos"] = cand["s1_pos"].astype(np.int32)
    cand["d_pos"] = cand["d_pos"].astype(np.int32)
    return cand


LEX_COLS = (["rank_" + n for n, _, _ in VIEWS] + ["sc_" + n for n, _, _ in VIEWS]
            + [flag for flag, _ in KEY_VIEWS] + ["n_views", "rrf"])
