"""Multi-view candidate generation: address TF-IDF, name char-n-gram TF-IDF, exact name key.

Frames passed in must have a clean 0..n-1 RangeIndex; returned positions refer to those rows.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer

RRF_K = 60


def _topk_chunk(Q, DT, s, e, k):
    P = (Q[s:e] @ DT).tocsr()
    n = e - s
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
        idx[r, :len(sel)] = ind[a:b][sel]
        sc[r, :len(sel)] = d[sel]
    return idx, sc


def topk_sparse(Q, D, k, chunk=500, n_jobs=-1):
    DT = D.T.tocsr()
    spans = [(s, min(s + chunk, Q.shape[0])) for s in range(0, Q.shape[0], chunk)]
    res = Parallel(n_jobs=n_jobs)(delayed(_topk_chunk)(Q, DT, s, e, k) for s, e in spans)
    return np.vstack([r[0] for r in res]), np.vstack([r[1] for r in res])


def pair_cos(Q, D, qi, di, chunk=1_000_000):
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        e = s + chunk
        out[s:e] = np.asarray(Q[qi[s:e]].multiply(D[di[s:e]]).sum(axis=1)).ravel()
    return out


def _vec(kind, texts_d, texts_q, cap):
    if kind == "addr":
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"\S+", min_df=1,
                            max_df=cap, sublinear_tf=True, dtype=np.float32)
    else:
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, max_df=cap,
                            sublinear_tf=True, dtype=np.float32)
    D = v.fit_transform(texts_d)
    return v.transform(texts_q), D


def _ranks_to_frame(idx, sc, name):
    nq, k = idx.shape
    q = np.repeat(np.arange(nq, dtype=np.int32), k)
    df = pd.DataFrame({"s1_pos": q, "d_pos": idx.ravel(), f"rank_{name}": np.tile(np.arange(1, k + 1), nq)})
    return df[df.d_pos >= 0]


def key_pairs(s1, d, cap=30):
    sizes = d.groupby("name_key").size()
    ok = sizes[(sizes <= cap)].index
    dd = pd.DataFrame({"name_key": d["name_key"].values, "d_pos": np.arange(len(d), dtype=np.int32)})
    dd = dd[dd.name_key.isin(ok) & (dd.name_key != "")]
    ss = pd.DataFrame({"name_key": s1["name_key"].values, "s1_pos": np.arange(len(s1), dtype=np.int32)})
    m = ss.merge(dd, on="name_key")[["s1_pos", "d_pos"]]
    m["in_key"] = 1
    return m


def gen_candidates(s1, d, k_addr=20, k_name=20, keep=15, n_jobs=-1):
    """Candidates for S1 rows against one country's slice of one source.

    Returns DataFrame[s1_pos, d_pos, rank_addr, rank_name, in_key, cos_addr, cos_name, rrf]
    limited to the ``keep`` best-fused candidates per S1 row.
    """
    if len(s1) == 0 or len(d) == 0:
        return pd.DataFrame()
    cap = max(50, int(0.0015 * len(d)))
    Qa, Da = _vec("addr", d["addr_norm"].values, s1["addr_norm"].values, cap)
    Qn, Dn = _vec("name", d["name_lat"].values, s1["name_lat"].values, cap)
    frames = []
    for name, Q, D, k in (("addr", Qa, Da, k_addr), ("name", Qn, Dn, k_name)):
        idx, sc = topk_sparse(Q, D, min(k, D.shape[0]), n_jobs=n_jobs)
        frames.append(_ranks_to_frame(idx, sc, name))
    frames.append(key_pairs(s1, d))
    cand = frames[0]
    for f in frames[1:]:
        cand = cand.merge(f, on=["s1_pos", "d_pos"], how="outer")
    cand["in_key"] = cand["in_key"].fillna(0).astype("int8")
    ra = cand["rank_addr"].fillna(1e6)
    rn = cand["rank_name"].fillna(1e6)
    cand["rrf"] = 1 / (RRF_K + ra) + 1 / (RRF_K + rn) + 0.01 * cand["in_key"]
    cand = cand.sort_values(["s1_pos", "rrf"], ascending=[True, False])
    cand = cand.groupby("s1_pos", sort=False).head(keep).reset_index(drop=True)
    qi, di = cand.s1_pos.values, cand.d_pos.values
    cand["cos_addr"] = pair_cos(Qa, Da, qi, di)
    cand["cos_name"] = pair_cos(Qn, Dn, qi, di)
    return cand
