"""Bi-encoder retrieval: embed to disk, search on the GPU by chunked matmul.

Three deliberate choices, all forced by 8 GB of VRAM and 16 GB of RAM:

* **Embeddings live on disk** as float16 memmaps, never in RAM. 11.7M records x 384 dims
  x 2 bytes = 9.0 GB; at 768 dims it is 18 GB. Both are fine on an SSD and neither fits
  in memory alongside the rest of the pipeline.
* **Brute-force chunked matmul, not FAISS.** FAISS-GPU has no Windows build, and an exact
  search is affordable here: the whole test set is about 1.4e16 FLOPs, minutes to tens of
  minutes on a 4060. Exact search also means the measured recall is the real recall, with
  no index-quality term to debug.
* **The encoder reads the raw text, not the folded text.** ``unidecode`` turns
  "राम मार्केटिंग" into "raam maarkettinng"; a multilingual encoder relates the original
  to "Ram Marketing" and the folded form to nothing. Folding is right for n-grams and
  wrong here.

Embedding is resumable at row granularity: a ``.progress`` file records how many rows of
the memmap are valid, so an interrupted pass restarts at that row.
"""
import gc
import json
import os

import numpy as np
import pandas as pd

from .paths import log

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def torch_device(prefer_gpu=True):
    try:
        import torch
    except ImportError as e:  # pragma: no cover - environment problem, not logic
        raise SystemExit(
            "PyTorch is not installed. Install the GPU build with:\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu126 torch\n"
            "  pip install transformers\n"
            "Every non-neural stage runs without it.") from e
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_encoder(model_name, device, fp16=True):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    if fp16 and device.type == "cuda":
        model.half()
    return tok, model


def pair_text(df):
    """`name | address` from the raw fields, with the e5 query prefix.

    e5 is trained with "query: "/"passage: " prefixes; entity resolution is symmetric
    (both sides are the same kind of string), so both sides get "query: ".
    """
    name = df["business_name"].fillna("").astype(str)
    addr = df["business_address"].fillna("").astype(str)
    return ("query: " + name + " | " + addr).to_numpy()


def encode_batch(tok, model, texts, device, max_len, dim=None):
    """Mean-pooled, unit-norm embeddings for one batch.

    Shared by embedding and fine-tuning on purpose: if the pooling used at training time
    differs by even a detail from the pooling used at inference time, the fine-tune silently
    makes retrieval worse instead of better.
    """
    enc = tok(list(texts), padding=True, truncation=True, max_length=max_len,
              return_tensors="pt").to(device)
    out = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
    if dim:
        pooled = pooled[:, :dim]
    return pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-6)


def _emb_meta_path(path):
    return path + ".json"


def emb_shape(path):
    with open(_emb_meta_path(path), encoding="utf-8") as f:
        m = json.load(f)
    return m["rows"], m["dim"], m.get("valid", 0)


def open_emb(path, mode="r"):
    rows, dim, _ = emb_shape(path)
    return np.memmap(path, dtype=np.float16, mode=mode, shape=(rows, dim))


def embed(path, texts, tok, model, device, cfg):
    """Write unit-norm float16 embeddings for ``texts`` to a memmap at ``path``.

    Restartable: rows already written are skipped, so a killed pass costs one batch.
    """
    import torch

    n = len(texts)
    dim = model.config.hidden_size
    if cfg.dim and cfg.dim < dim:
        dim = cfg.dim  # Matryoshka-style truncation: cheaper store, small recall cost
    valid = 0
    if os.path.exists(path) and os.path.exists(_emb_meta_path(path)):
        rows, d0, valid = emb_shape(path)
        if rows != n or d0 != dim:
            valid = 0
    meta = {"rows": n, "dim": dim, "valid": valid}
    mm = np.memmap(path, dtype=np.float16, mode="r+" if valid else "w+", shape=(n, dim))
    # Shape metadata is written before the first batch, so the file is always describable
    # even if the pass dies during it.
    with open(_emb_meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    if valid >= n:
        log(f"  {os.path.basename(path)}: complete ({n:,} rows)")
        del mm
        return
    log(f"  {os.path.basename(path)}: {n:,} rows, dim={dim}, resuming at {valid:,}")

    bs = cfg.batch
    t0 = valid
    with torch.inference_mode():
        for s in range(valid, n, bs):
            batch = list(texts[s:s + bs])
            pooled = encode_batch(tok, model, batch, device, cfg.max_len, dim)
            mm[s:s + len(batch)] = pooled.float().cpu().numpy().astype(np.float16)
            if (s // bs) % cfg.flush_every == 0 and s > t0:
                mm.flush()
                meta["valid"] = s
                with open(_emb_meta_path(path), "w", encoding="utf-8") as f:
                    json.dump(meta, f)
                log(f"    {s:,}/{n:,}")
    mm.flush()
    del mm
    meta["valid"] = n
    with open(_emb_meta_path(path), "w", encoding="utf-8") as f:
        json.dump(meta, f)


def search(q_path, d_path, k, cfg):
    """Exact top-k by chunked fp16 matmul. Returns (s1_pos, d_pos, cos) arrays.

    The running top-k lives on the GPU for one query block at a time, so nothing of size
    n_s1 x k is ever held on the host.
    """
    import torch

    device = torch_device()
    # fp16 throughout on the GPU: the score matrix is the peak allocation, and keeping it
    # half (instead of casting to float32 for topk) is what makes q_block x d_block fit in
    # 8 GB. CPU has no fast half matmul, so it falls back to float32.
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    Q = open_emb(q_path)
    D = open_emb(d_path)
    nq, dim = Q.shape
    nd = D.shape[0]
    k = min(k, nd)
    qi_all, di_all, sc_all = [], [], []
    for qs in range(0, nq, cfg.q_block):
        qe = min(qs + cfg.q_block, nq)
        q = torch.from_numpy(np.ascontiguousarray(Q[qs:qe])).to(device=device, dtype=dtype)
        best_s = torch.full((qe - qs, k), -2.0, device=device, dtype=dtype)
        best_i = torch.full((qe - qs, k), -1, device=device, dtype=torch.int32)
        for ds in range(0, nd, cfg.d_block):
            de = min(ds + cfg.d_block, nd)
            d = torch.from_numpy(np.ascontiguousarray(D[ds:de])).to(device=device, dtype=dtype)
            sims = q @ d.T
            kk = min(k, de - ds)
            s, i = torch.topk(sims, kk, dim=1)
            del sims, d
            cat_s = torch.cat([best_s, s], 1)
            cat_i = torch.cat([best_i, (i + ds).int()], 1)
            best_s, order = torch.topk(cat_s, k, dim=1)
            best_i = torch.gather(cat_i, 1, order)
            del cat_s, cat_i, s, i, order
        m = (best_i >= 0).cpu().numpy()
        rows = np.repeat(np.arange(qs, qe, dtype=np.int32)[:, None], k, axis=1)
        qi_all.append(rows[m])
        di_all.append(best_i.cpu().numpy()[m].astype(np.int32))
        sc_all.append(best_s.float().cpu().numpy()[m].astype(np.float32))
        del q, best_s, best_i
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if (qs // cfg.q_block) % 10 == 0:
            log(f"    dense search {qe:,}/{nq:,}")
    del Q, D
    gc.collect()
    return (np.concatenate(qi_all), np.concatenate(di_all), np.concatenate(sc_all))


def cos_for_pairs(q_path, d_path, qi, di, chunk=2_000_000):
    """Exact cosine for arbitrary (s1_pos, d_pos) pairs, straight out of the memmaps.

    This is what gives every union pair one exact dense similarity, including pairs that
    only a lexical view retrieved."""
    Q = open_emb(q_path)
    D = open_emb(d_path)
    out = np.empty(len(qi), np.float32)
    for s in range(0, len(qi), chunk):
        e = min(s + chunk, len(qi))
        a = np.asarray(Q[qi[s:e]], dtype=np.float32)
        b = np.asarray(D[di[s:e]], dtype=np.float32)
        out[s:e] = np.einsum("ij,ij->i", a, b)
        del a, b
    del Q, D
    return out


def topk_frame(qi, di, sc):
    df = pd.DataFrame({"s1_pos": qi, "d_pos": di, "cos_dense": sc})
    df = df.sort_values(["s1_pos", "cos_dense"], ascending=[True, False])
    df["rank_dense"] = df.groupby("s1_pos", sort=False).cumcount().add(1).astype(np.float32)
    return df.reset_index(drop=True)
