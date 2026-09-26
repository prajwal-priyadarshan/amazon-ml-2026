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


def topk_sparse(Q, D, k, chunk=2_500, n_jobs=-1):
    DT = D.T.tocsr()
    spans = [(s, min(s + chunk, Q.shape[0])) for s in range(0, Q.shape[0], chunk)]
    res = Parallel(n_jobs=n_jobs, batch_size=1)(delayed(_topk_chunk)(Q, DT, s, e, k) for s, e in spans)
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
    elif kind == "nword":
        # Word-level name view: survives word reordering and abbreviation swaps that
        # dilute char n-grams, and its IDF weighting makes rare tokens carry the match.
        v = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), token_pattern=r"\S+", min_df=1,
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


def key_pairs(s1, d, col="name_key", flag="in_key", cap=50):
    """Exact-key join. Hash joins are inherently bidirectional, so these views also
    recover pairs that neither side's top-k would have surfaced."""
    sizes = d.groupby(col).size()
    ok = sizes[(sizes <= cap)].index
    dd = pd.DataFrame({col: d[col].values, "d_pos": np.arange(len(d), dtype=np.int32)})
    dd = dd[dd[col].isin(ok) & (dd[col] != "")]
    ss = pd.DataFrame({col: s1[col].values, "s1_pos": np.arange(len(s1), dtype=np.int32)})
    m = ss.merge(dd, on=col)[["s1_pos", "d_pos"]]
    m[flag] = 1
    return m


def _get_dense_embeddings(texts, model_name="all-MiniLM-L6-v2", batch_size=512):
    """Compute normalized dense embeddings using sentence-transformers if available."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float32)
    except Exception:
        return None


def gen_candidates(s1, d, k_addr=30, k_name=30, keep=30, n_jobs=-1, use_dense=True):
    """Candidates for S1 rows against one country's slice of one source.

    Six complementary views:
    address word TF-IDF, name char n-grams, name word TF-IDF, exact name key, exact
    phonetic skeleton, and dense sentence-transformer embeddings (if sentence_transformers is installed).
    """
    if len(s1) == 0 or len(d) == 0:
        return pd.DataFrame()
    cap = max(50, int(0.0015 * len(d)))
    Qa, Da = _vec("addr", d["addr_norm"].values, s1["addr_norm"].values, cap)
    Qn, Dn = _vec("name", d["name_lat"].values, s1["name_lat"].values, cap)
    Qw, Dw = _vec("nword", d["name_core"].values, s1["name_core"].values, cap)
    # Phonetic view: the only view that reliably survives romanised Indic text
    Qp, Dp = _vec("name", d["name_skel"].values, s1["name_skel"].values, cap)
    frames = []
    for name, Q, D, k in (("addr", Qa, Da, k_addr), ("name", Qn, Dn, k_name),
                          ("nword", Qw, Dw, k_name), ("skel", Qp, Dp, k_name)):
        idx, sc = topk_sparse(Q, D, min(k, D.shape[0]), n_jobs=n_jobs)
        frames.append(_ranks_to_frame(idx, sc, name))
    frames.append(key_pairs(s1, d, "name_key", "in_key"))
    frames.append(key_pairs(s1, d, "name_skel", "in_skel"))

    # Dense Vector Embedding Retrieval (if available)
    dense_active = False
    if use_dense and len(s1) <= 150_000 and len(d) <= 1_500_000:
        s1_texts = (s1["name_lat"] + " " + s1["addr_norm"]).tolist()
        d_texts = (d["name_lat"] + " " + d["addr_norm"]).tolist()
        emb_q = _get_dense_embeddings(s1_texts)
        emb_d = _get_dense_embeddings(d_texts)
        if emb_q is not None and emb_d is not None:
            dense_active = True
            # Compute top-k nearest neighbors per S1 query
            k_dense = min(k_name, emb_d.shape[0])
            import torch
            t_q = torch.from_numpy(emb_q)
            t_d = torch.from_numpy(emb_d).t()
            sims = torch.mm(t_q, t_d)
            vals, inds = torch.topk(sims, k=k_dense, dim=1)
            idx_emb = inds.numpy().astype(np.int32)
            sc_emb = vals.numpy().astype(np.float32)
            frames.append(_ranks_to_frame(idx_emb, sc_emb, "emb"))

    cand = frames[0]
    for f in frames[1:]:
        cand = cand.merge(f, on=["s1_pos", "d_pos"], how="outer")
    for flag in ("in_key", "in_skel"):
        cand[flag] = cand[flag].fillna(0).astype("int8")
    ra = cand["rank_addr"].fillna(1e6)
    rn = cand["rank_name"].fillna(1e6)
    rw = cand["rank_nword"].fillna(1e6)
    rp = cand["rank_skel"].fillna(1e6)
    remb = cand["rank_emb"].fillna(1e6) if "rank_emb" in cand.columns else 1e6
    cand["rrf"] = (1 / (RRF_K + ra) + 1 / (RRF_K + rn) + 1 / (RRF_K + rw) + 1 / (RRF_K + rp)
                   + (1 / (RRF_K + remb) if dense_active else 0.0)
                   + 0.01 * cand["in_key"] + 0.01 * cand["in_skel"])
    cand = cand.sort_values(["s1_pos", "rrf"], ascending=[True, False])
    if keep:  # keep=0 returns the full union, i.e. the recall ceiling of this view set
        cand = cand.groupby("s1_pos", sort=False).head(keep).reset_index(drop=True)
    else:
        cand = cand.reset_index(drop=True)
    qi, di = cand.s1_pos.values, cand.d_pos.values
    cand["cos_addr"] = pair_cos(Qa, Da, qi, di)
    cand["cos_name"] = pair_cos(Qn, Dn, qi, di)
    cand["cos_nword"] = pair_cos(Qw, Dw, qi, di)
    cand["cos_skel"] = pair_cos(Qp, Dp, qi, di)
    cand["cos_emb"] = 0.0 if not dense_active else np.zeros(len(cand), dtype=np.float32)
    for c in ("rank_addr", "rank_name", "rank_nword", "rank_skel", "rank_emb"):
        if c in cand.columns:
            cand[c] = cand[c].fillna(1e6).astype(np.float32)
        else:
            cand[c] = np.float32(1e6)
    return cand

