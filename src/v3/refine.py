"""L5 refine: a second-level model over the stacker's calibrated probability.

Why a level on top instead of more stacker features. The encoder, pruner, judge and stacker
were all trained on the *fit* split, so on fit their scores are over-confident and the
stacker learned to trust them too much. The hold-out split has never been seen by any of
them. A model trained on hold-out (with S1-grouped cross-fitting for its own evaluation)
sees the stacker's probability as it really behaves on unseen entities, and can correct it.

What it adds is evidence no pairwise model has: the *cluster*. An audit of hold-out errors
shows the two dominant failure shapes:

* decoys: near-copies of a real entity with no owner ("Cebrio Value Box co, 101 LEE PL"
  against true copies at "101-D LEE PL"; "60 WATERFORD WAY" against "600-604 ..."). They
  look like the S1 record but *disagree with the entity's other confident copies*.
* records with no address ("Lumium Security", "") that are only matchable by name, which
  is safe when the name is rare and when the entity's other copies carry the same name.

So every pair is described against the entity's other confident candidates in *both*
sources (name, address and house-number agreement, consensus house number), by how rare
its name is in the country, and by the competition for the record across all S1 entities.
Everything is country-agnostic and mined from the provided files only.
"""
import gc

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .paths import log

TXT = ["name_core", "name_key", "addr_norm", "nums", "addr_missing", "legal", "business_name"]

REFINE_COLS = [
    "logit", "src",
    "a_nm", "a_ad", "a_num", "rec_miss", "s1_miss",
    "xs_rank", "xs_n50", "xs_n90", "xs_sum", "xs_other", "xs_other_src",
    "r_other", "r_n50", "r_rank",
    "k_pool_rec", "k_pool_s1", "k_addr_rec", "k_s1a_rec", "k_s1a_s1", "k_s1a_rec_other",
    "m_cnt", "m_nm_max", "m_nm_mean", "m_nm_wmax", "m_nm_eq", "m_ad_max",
    "m_num_has", "m_num_agree", "m_num_dis", "m_x_nm_max", "cons_num",
    "a_legal", "a_extra", "a_drop", "m_leg_has", "m_leg_agree", "cons_legal",
    "a_nm_idf", "a_ad_idf", "a_ad_idf_miss",
    "rec_oov", "s1_oov", "rec_alias", "s1_alias",
]

_ALIAS = r"(?i)\b(?:f/?k/?a|formerly|d/?b/?a|doing business as|trading as|t/a|a/?k/?a|also known as)\b"


def _oov(texts, vocab, min_count=3):
    """Share of name tokens that never occur in any Source-1 name: invented brand names
    ("Rizapyra", "Orbidrex") that the generator uses for rebranded copies."""
    out = np.zeros(len(texts), np.float32)
    for i, x in enumerate(texts):
        toks = str(x).split() if x else []
        if toks:
            out[i] = sum(1 for t in toks if vocab.get(t, 0) < min_count) / len(toks)
    return out


def _idf(texts):
    """Per-country token IDF over the record pool: region names, "rue", "road", a city
    shared by half the pool all get near-zero weight, rare street names get most."""
    df = pd.Series([t for s in texts for t in set(str(s).split())]).value_counts()
    n = max(len(texts), 1)
    return (np.log((n + 1) / (df + 1)) + 1.0).to_dict()


def _wjacc(a, b, idf, default):
    """IDF-weighted Jaccard of the token sets, plus the IDF mass of S1 tokens the record lacks."""
    out = np.zeros(len(a), np.float32)
    miss = np.zeros(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        xs = set(str(x).split()) if x else set()
        ys = set(str(y).split()) if y else set()
        if not xs or not ys:
            out[i] = -1
            continue
        inter = sum(idf.get(t, default) for t in xs & ys)
        uni = sum(idf.get(t, default) for t in xs | ys)
        out[i] = inter / uni if uni else 0.0
        miss[i] = sum(idf.get(t, default) for t in xs - ys)
    return out, miss


def _rescale(p, pi_from, pi_to):
    a = pi_to / pi_from
    b = (1 - pi_to) / (1 - pi_from)
    return (a * p / (a * p + b * (1 - p))).astype(np.float32)


def em_prior(p, pi_train, iters=100, tol=1e-6):
    """Saerens-Latinne-Decaestecker EM: the positive rate these probabilities imply on a
    new population whose class prior may differ from the training one."""
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    pi = pi_train
    for _ in range(iters):
        new = float(_rescale(p, pi_train, pi).mean())
        if abs(new - pi) < tol:
            break
        pi = new
    return pi


def prior_shift(p, pi_train, country=None):
    pi = em_prior(p, pi_train)
    log(f"  prior shift [{country}]: train {pi_train:.4f} -> EM {pi:.4f}")
    return _rescale(np.clip(p, 1e-6, 1 - 1e-6), pi_train, pi)


def _tset(a, b):
    if len(a) == 0:
        return np.zeros(0, np.float32)
    return process.cpdist(list(a), list(b), scorer=fuzz.token_set_ratio,
                          dtype=np.float32, workers=-1).astype(np.float32)


def _first_num(nums):
    s = pd.Series(nums, dtype=object).fillna("").astype(str).str.split().str[0]
    return pd.to_numeric(s, errors="coerce").fillna(-1).to_numpy(np.float64)


def _other_best(key, p):
    """For each row: the best probability among the *other* rows sharing ``key``."""
    df = pd.DataFrame({"k": key, "p": p, "i": np.arange(len(p))})
    df = df.sort_values(["k", "p"], ascending=[True, False])
    first = ~df.duplicated("k")
    top1 = df[first].set_index("k")["p"]
    top2 = df[~first].drop_duplicates("k").set_index("k")["p"]
    t1 = top1.reindex(key).to_numpy(np.float32)
    t2 = top2.reindex(key).fillna(0.0).to_numpy(np.float32)
    is_first = np.zeros(len(p), bool)
    is_first[df["i"].to_numpy()[first.to_numpy()]] = True
    return np.where(is_first, t2, t1).astype(np.float32)


def features(sc, s1, dmap, s1_keys=None, vocab=None, conf=0.8, max_mates=8, s1_chunk=100_000):
    """sc: one country, both sources: s1_pos, d_pos, src, prob. Returns sc + REFINE_COLS."""
    sc = sc.reset_index(drop=True)
    n = len(sc)
    sp = sc["s1_pos"].to_numpy(np.int64)
    dp = sc["d_pos"].to_numpy(np.int64)
    sr = sc["src"].to_numpy(np.int64)
    p = sc["prob"].to_numpy(np.float32)

    def dcol(c):
        out = np.empty(n, dtype=object if c != "addr_missing" else np.int8)
        for s, d in dmap.items():
            m = sr == s
            out[m] = d[c].to_numpy()[dp[m]]
        return out

    rec = {c: dcol(c) for c in TXT}
    s1c = {c: s1[c].to_numpy()[sp] for c in TXT}
    rnum = _first_num(rec["nums"])
    snum = _first_num(s1c["nums"])

    f = {}
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    f["logit"] = np.log(pc / (1 - pc)).astype(np.float32)
    f["src"] = sr.astype(np.float32)
    f["a_nm"] = _tset(s1c["name_core"], rec["name_core"])
    f["a_ad"] = _tset(s1c["addr_norm"], rec["addr_norm"])
    f["a_num"] = np.where((rnum < 0) | (snum < 0), -1, (rnum == snum).astype(np.float32)).astype(np.float32)
    f["rec_miss"] = rec["addr_missing"].astype(np.float32)
    rl = pd.Series(rec["legal"], dtype=object).fillna("").astype(str).to_numpy()
    sl = pd.Series(s1c["legal"], dtype=object).fillna("").astype(str).to_numpy()
    f["a_legal"] = np.where((rl == "") | (sl == ""), -1, (rl == sl).astype(np.float32)).astype(np.float32)
    # words the record adds to / drops from the S1 name: sibling businesses at the same
    # address differ by exactly this ("... FRANCE SAS", "... International SAS")
    ext = np.zeros(n, np.float32)
    drp = np.zeros(n, np.float32)
    for i, (x, y) in enumerate(zip(rec["name_core"], s1c["name_core"])):
        a_ = set(str(x).split()) if x else set()
        b_ = set(str(y).split()) if y else set()
        ext[i] = len(a_ - b_)
        drp[i] = len(b_ - a_)
    f["a_extra"] = ext
    f["a_drop"] = drp
    vocab = vocab or {}
    f["rec_oov"] = _oov(rec["name_core"], vocab)
    f["s1_oov"] = _oov(s1c["name_core"], vocab)
    f["rec_alias"] = pd.Series(rec["business_name"], dtype=object).fillna("").str.contains(_ALIAS).to_numpy(np.float32)
    f["s1_alias"] = pd.Series(s1c["business_name"], dtype=object).fillna("").str.contains(_ALIAS).to_numpy(np.float32)
    pool_names = np.concatenate([d["name_core"].fillna("").to_numpy() for d in dmap.values()])
    pool_addrs = np.concatenate([d["addr_norm"].fillna("").to_numpy() for d in dmap.values()])
    idf_n, idf_a = _idf(pool_names), _idf(pool_addrs)
    dn = float(np.log(len(pool_names) + 1) + 1)
    f["a_nm_idf"], _ = _wjacc(s1c["name_core"], rec["name_core"], idf_n, dn)
    f["a_ad_idf"], f["a_ad_idf_miss"] = _wjacc(s1c["addr_norm"], rec["addr_norm"], idf_a, dn)
    del pool_names, pool_addrs, idf_n, idf_a
    f["s1_miss"] = s1c["addr_missing"].astype(np.float32)

    # --- per-S1 competition across both sources --------------------------------------
    g = pd.DataFrame({"sp": sp, "p": p, "sr": sr})
    f["xs_rank"] = g.groupby("sp")["p"].rank(ascending=False, method="first").to_numpy(np.float32)
    f["xs_n50"] = g.assign(v=p >= 0.5).groupby("sp")["v"].transform("sum").to_numpy(np.float32)
    f["xs_n90"] = g.assign(v=p >= 0.9).groupby("sp")["v"].transform("sum").to_numpy(np.float32)
    f["xs_sum"] = g.groupby("sp")["p"].transform("sum").to_numpy(np.float32)
    f["xs_other"] = _other_best(sp, p)
    # best probability of this S1 in the *other* source
    other_src_best = g.groupby(["sp", "sr"])["p"].max()
    osr = np.where(sr == 2, 3, 2)
    idx = pd.MultiIndex.from_arrays([sp, osr])
    f["xs_other_src"] = other_src_best.reindex(idx).fillna(0.0).to_numpy(np.float32)

    # --- per-record competition across S1 entities -----------------------------------
    rk = (sr << 32) | dp
    f["r_other"] = _other_best(rk, p)
    r = pd.DataFrame({"rk": rk, "p": p})
    f["r_n50"] = r.assign(v=p >= 0.5).groupby("rk")["v"].transform("sum").to_numpy(np.float32)
    f["r_rank"] = r.groupby("rk")["p"].rank(ascending=False, method="first").to_numpy(np.float32)

    # --- rarity -------------------------------------------------------------------------
    s1_key_cnt = s1["name_key"].value_counts()
    pool_key_cnt = pd.concat([d["name_key"] for d in dmap.values()]).value_counts()
    pool_addr = pd.concat([d.loc[d["addr_missing"] == 0, "addr_norm"] for d in dmap.values()])
    pool_addr_cnt = pool_addr.value_counts()
    f["k_s1_rec"] = np.log1p(pd.Series(rec["name_key"]).map(s1_key_cnt).fillna(0).to_numpy(np.float32))
    f["k_pool_rec"] = np.log1p(pd.Series(rec["name_key"]).map(pool_key_cnt).fillna(0).to_numpy(np.float32))
    f["k_pool_s1"] = np.log1p(pd.Series(s1c["name_key"]).map(pool_key_cnt).fillna(0).to_numpy(np.float32))
    ka = pd.Series(rec["addr_norm"]).map(pool_addr_cnt).fillna(0).to_numpy(np.float32)
    f["k_addr_rec"] = np.where(rec["addr_missing"] == 1, -1, np.log1p(ka)).astype(np.float32)
    # potential owners: S1 entities in the whole S1 table sharing the name key
    all_cnt = (s1_keys if s1_keys is not None else s1["name_key"]).value_counts()
    c_rec = pd.Series(rec["name_key"]).map(all_cnt).fillna(0).to_numpy(np.float32)
    f["k_s1a_rec"] = np.log1p(c_rec)
    f["k_s1a_s1"] = np.log1p(pd.Series(s1c["name_key"]).map(all_cnt).fillna(0).to_numpy(np.float32))
    same = (pd.Series(rec["name_key"]).to_numpy() == pd.Series(s1c["name_key"]).to_numpy())
    f["k_s1a_rec_other"] = np.log1p(np.maximum(c_rec - same, 0)).astype(np.float32)
    del s1_key_cnt, pool_key_cnt, pool_addr, pool_addr_cnt, all_cnt

    # --- cluster mates: the entity's other confident candidates ----------------------
    for c in ("m_cnt", "m_nm_max", "m_nm_mean", "m_nm_wmax", "m_nm_eq", "m_ad_max",
              "m_num_has", "m_num_agree", "m_num_dis", "m_x_nm_max", "m_leg_has", "m_leg_agree"):
        f[c] = np.zeros(n, np.float32)
    f["m_ad_max"][:] = -1
    conf_rows = np.flatnonzero(p >= conf)
    cm = pd.DataFrame({"sp": sp[conf_rows], "mrow": conf_rows, "p": p[conf_rows]})
    cm = cm.sort_values(["sp", "p"], ascending=[True, False]).groupby("sp", sort=False).head(max_mates)
    rows = pd.DataFrame({"sp": sp, "row": np.arange(n)})
    s1_order = np.unique(sp)
    for lo in range(0, len(s1_order), s1_chunk):
        keys = s1_order[lo:lo + s1_chunk]
        a = rows[rows["sp"].isin(keys)]
        b = cm[cm["sp"].isin(keys)]
        x = a.merge(b[["sp", "mrow"]], on="sp")
        x = x[x["row"] != x["mrow"]]
        if x.empty:
            continue
        ri = x["row"].to_numpy()
        mi = x["mrow"].to_numpy()
        nm = _tset(rec["name_core"][ri], rec["name_core"][mi])
        both_addr = (rec["addr_missing"][ri] == 0) & (rec["addr_missing"][mi] == 0)
        ad = np.full(len(ri), -1.0, np.float32)
        if both_addr.any():
            ad[both_addr] = _tset(rec["addr_norm"][ri][both_addr], rec["addr_norm"][mi][both_addr])
        n_has = (rnum[ri] >= 0) & (rnum[mi] >= 0)
        n_eq = n_has & (rnum[ri] == rnum[mi])
        xsrc = sr[ri] != sr[mi]
        l_has = (rl[ri] != "") & (rl[mi] != "")
        l_eq = l_has & (rl[ri] == rl[mi])
        t = pd.DataFrame({"row": ri, "nm": nm, "w": nm * p[mi], "eq": nm >= 99.5, "ad": ad,
                          "nh": n_has, "ne": n_eq, "nd": n_has & ~n_eq,
                          "xn": np.where(xsrc, nm, 0.0), "lh": l_has, "le": l_eq})
        agg = t.groupby("row").agg(m_cnt=("nm", "size"), m_nm_max=("nm", "max"),
                                   m_nm_mean=("nm", "mean"), m_nm_wmax=("w", "max"),
                                   m_nm_eq=("eq", "max"), m_ad_max=("ad", "max"),
                                   m_num_has=("nh", "sum"), m_num_agree=("ne", "sum"),
                                   m_num_dis=("nd", "sum"), m_x_nm_max=("xn", "max"),
                                   m_leg_has=("lh", "sum"), m_leg_agree=("le", "sum"))
        idx = agg.index.to_numpy()
        for c in agg.columns:
            f[c][idx] = agg[c].to_numpy(np.float32)
        del a, b, x, t, agg
        gc.collect()
    # consensus house number: the mode over the S1 record and its confident mates
    cn = pd.concat([pd.DataFrame({"sp": np.unique(sp), "num": _first_num(s1["nums"].to_numpy()[np.unique(sp)])}),
                    pd.DataFrame({"sp": sp[cm["mrow"].to_numpy()], "num": rnum[cm["mrow"].to_numpy()]})])
    cn = cn[cn["num"] >= 0]
    mode = (cn.groupby(["sp", "num"]).size().rename("c").reset_index()
            .sort_values(["sp", "c"], ascending=[True, False]).drop_duplicates("sp")
            .set_index("sp")["num"])
    cons = mode.reindex(sp).to_numpy(np.float64)
    f["cons_num"] = np.where((rnum < 0) | np.isnan(cons), -1, (rnum == cons).astype(np.float32)).astype(np.float32)

    # consensus legal form over the S1 record and its confident mates
    u = np.unique(sp)
    cl = pd.concat([pd.DataFrame({"sp": u, "l": s1["legal"].fillna("").astype(str).to_numpy()[u]}),
                    pd.DataFrame({"sp": sp[cm["mrow"].to_numpy()], "l": rl[cm["mrow"].to_numpy()]})])
    cl = cl[cl["l"] != ""]
    lmode = (cl.groupby(["sp", "l"]).size().rename("c").reset_index()
             .sort_values(["sp", "c"], ascending=[True, False]).drop_duplicates("sp")
             .set_index("sp")["l"])
    consl = lmode.reindex(sp).to_numpy(dtype=object)
    has_c = pd.notna(consl)
    f["cons_legal"] = np.where((rl == "") | ~has_c, -1, (rl == consl).astype(np.float32)).astype(np.float32)

    for c in REFINE_COLS:
        sc[c] = f[c].astype(np.float32)
    return sc
