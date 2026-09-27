"""v3 orchestrator. One subcommand per stage, every stage resumable, every stage gated.

    python -m src.v3.run <stage> --data-dir ../../student_resource/dataset --work-dir ../../work3

Stages, in dependency order (see README_v3.md for the runbook and the gate on each one):

    prep         normalise once, shard by (country, source)
    lexical      sparse retrieval, five views
    embed        bi-encoder -> float16 memmaps on disk
    dense        exact GPU top-k search
    union        fuse lexical + dense into the candidate set
    diag         per-view recall and the oracle F0.5 ceiling      <- read this before modelling
    train-bi     fine-tune the bi-encoder (then re-run embed/dense/union)
    train-prune  fit the L2 pruner
    prune        apply it, keep the top few per S1 per source
    train-ce     fine-tune the cross-encoder judge
    ce           gated judge inference
    train-stack  fit the L3 stacker and the per-country calibrators
    stack        apply them
    tune         pick the decision rule on the hold-out
    score        hold-out macro F0.5, with breakdowns
    errors       miss taxonomy: retrieval vs ranking vs false positives
    resolve      write matching_results.tsv and candidate_pairs.tsv
    pseudo       export France pseudo-labels for a second fine-tune pass

The split axis is ``fit`` / ``holdout`` / ``test``. ``fit`` and ``holdout`` are disjoint S1
samples of the train file measured against the *full* S2/S3 pool, which is the only setup
that gives an honest number: scoring against a subsampled pool reads about three points
high, and that artefact is what made an earlier version look like 0.9668 when it was 0.9388.
"""
import argparse
import gc
import os

import joblib
import numpy as np
import pandas as pd

from ..data import read_source
from ..metric import f05_single
from . import crossenc, dense, diag, gbdt, lexical, pairs, prep, resolve, sibling, train_bi
from .paths import SOURCES, Work, done, log, pool_of

LEX_NEED = ["addr_norm", "name_lat", "name_core", "name_skel", "name_key",
            "name_seg", "seg_key"]
FEAT_NEED = ["entity_id", "name_lat", "name_core", "name_key", "name_skel", "legal",
             "native", "addr_norm", "nums", "addr_missing", "name_seg", "seg_key"]
TEXT_NEED = ["business_name", "business_address"]
PRUNE_BLOCK = 4_000_000  # rows scored per matrix in `prune`; bounds the float32 design matrix
PRUNED_COLS =pairs.LEAN + ["src"] + pairs.RETRIEVAL_COLS + pairs.GROUP_COLS + ["prob1"]


# ---------------------------------------------------------------- helpers

def _s1(w, split, country, cols):
    return prep.load_s1(w, split, country, cols)


def _d(w, split, country, src, cols, owner=False):
    cols = list(cols) + (["owner"] if owner else [])
    d = prep.load_d(w, split, country, src, cols)
    d["src"] = np.int8(src)
    return d


def _blocks(w, split):
    for country in w.countries(split):
        for src in SOURCES:
            yield country, src


def _group_of(s1_pos, n_groups=5, seed=0):
    """Deterministic per-S1 group, so train/val/cal never share an entity."""
    h = (s1_pos.astype(np.uint64) * np.uint64(2654435761) + np.uint64(seed))
    return (h % np.uint64(n_groups)).astype(np.int8)


def _emb_key_s1(split, country):
    return f"s1_{split}_{country}"


def _emb_key_d(split, country, src):
    return f"d_{pool_of(split)}_{country}_{src}"


def _emb_path(w, a, kind, key):
    return w.emb(f"{a.emb_tag}_{kind}", key)


def _bi_source(w, a):
    """The fine-tuned encoder if it exists, otherwise the pretrained checkpoint."""
    tuned = w.model("bi")
    if os.path.isdir(tuned) and os.path.exists(os.path.join(tuned, "config.json")):
        return tuned
    return a.bi_model or dense.DEFAULT_MODEL


# ---------------------------------------------------------------- stages

def stage_prep(a, w):
    prep.run(a, w)


def stage_lexical(a, w):
    for country, src in _blocks(w, a.split):
        out = w.cand("lex", a.split, country, src)
        if done(out, a.force):
            continue
        s1 = _s1(w, a.split, country, LEX_NEED)
        d = _d(w, a.split, country, src, LEX_NEED)
        log(f"lexical {a.split} {country} src={src}: {len(s1):,} S1 x {len(d):,} records")
        cand = lexical.build(a, s1, d)
        cand.to_parquet(out, index=False, compression="zstd")
        log(f"  -> {len(cand):,} pairs ({len(cand) / max(len(s1), 1):.1f}/S1)")
        del s1, d, cand
        gc.collect()


def stage_embed(a, w):
    model_src = _bi_source(w, a)
    device = dense.torch_device()
    tok, model = dense.load_encoder(model_src, device)
    log(f"embedding with {model_src} on {device.type}")
    for country in w.countries(a.split):
        path = _emb_path(w, a, "s1", _emb_key_s1(a.split, country))
        s1 = _s1(w, a.split, country, TEXT_NEED)
        dense.embed(path, dense.pair_text(s1), tok, model, device, a)
        del s1
        gc.collect()
    for country in w.countries(a.split):
        for src in SOURCES:
            path = _emb_path(w, a, "d", _emb_key_d(a.split, country, src))
            d = prep.load_d(w, a.split, country, src, TEXT_NEED)
            dense.embed(path, dense.pair_text(d), tok, model, device, a)
            del d
            gc.collect()


def stage_dense(a, w):
    for country, src in _blocks(w, a.split):
        out = w.cand("dense", a.split, country, src)
        if done(out, a.force):
            continue
        q = _emb_path(w, a, "s1", _emb_key_s1(a.split, country))
        dp = _emb_path(w, a, "d", _emb_key_d(a.split, country, src))
        if not (os.path.exists(q + ".json") and os.path.exists(dp + ".json")):
            log(f"  no embeddings for {country} src={src}; run `embed` first")
            continue
        log(f"dense search {a.split} {country} src={src}")
        qi, di, sc = dense.search(q, dp, a.k_dense, a)
        frame = dense.topk_frame(qi, di, sc)
        frame.to_parquet(out, index=False, compression="zstd")
        log(f"  -> {len(frame):,} pairs")
        del qi, di, sc, frame
        gc.collect()


def stage_union(a, w):
    for country, src in _blocks(w, a.split):
        out = w.cand("union", a.split, country, src)
        if done(out, a.force):
            continue
        lex_p = w.cand("lex", a.split, country, src)
        dn_p = w.cand("dense", a.split, country, src)
        lex = pd.read_parquet(lex_p) if os.path.exists(lex_p) else None
        dn = pd.read_parquet(dn_p) if os.path.exists(dn_p) else None
        if lex is None and dn is None:
            continue
        cand = pairs.union(lex, dn, a.keep_union)
        # Exact dense cosine for every union pair, including the ones only a lexical view
        # found. This is the one continuous similarity that is defined on all of them.
        q = _emb_path(w, a, "s1", _emb_key_s1(a.split, country))
        dp = _emb_path(w, a, "d", _emb_key_d(a.split, country, src))
        if os.path.exists(q + ".json") and os.path.exists(dp + ".json"):
            cand["cos_dense"] = dense.cos_for_pairs(q, dp, cand["s1_pos"].to_numpy(),
                                                    cand["d_pos"].to_numpy())
        cand["src"] = np.int8(src)
        cand = pairs.add_group_features(cand)
        cand.to_parquet(out, index=False, compression="zstd")
        log(f"union {a.split} {country} src={src}: {len(cand):,} pairs")
        del lex, dn, cand
        gc.collect()


def stage_diag(a, w):
    def blocks():
        for country, src in _blocks(w, a.split):
            p = w.cand("union", a.split, country, src)
            if not os.path.exists(p):
                continue
            s1 = _s1(w, a.split, country, ["entity_id"])
            d = _d(w, a.split, country, src, ["entity_id"], owner=True)
            truth = diag.true_pairs(s1, d, src)
            cand = pd.read_parquet(p)
            yield country, src, len(s1), truth, cand
            del s1, d, truth, cand
            gc.collect()

    diag.recall_report(blocks())


def _fit_rows(w, a, cols, source, sample_neg, groups=(0,), max_rows=None, extra=None):
    """Assemble a labelled design matrix from the fit split.

    ``source`` is ``"union"`` (pruner) or ``"pruned"`` (stacker). Positives are always kept
    in full and negatives are sampled: negatives are plentiful and highly redundant, and
    this is what makes the fit hold in 16 GB.
    """
    rng = np.random.default_rng(a.seed)
    Xs, ys, cs, ps = [], [], [], []
    total = 0
    for country, src in _blocks(w, "fit"):
        path = w.cand("union", "fit", country, src) if source == "union" else w.pruned("fit", country, src)
        if not os.path.exists(path):
            continue
        cand = pd.read_parquet(path)
        s1 = _s1(w, "fit", country, FEAT_NEED)
        d = _d(w, "fit", country, src, FEAT_NEED, owner=True)
        y = pairs.label(cand, s1, d)
        # Late features first, sampling second. Every stage-2, sibling and cross-source
        # column is a statistic over an S1's *whole* candidate list; computing them after
        # dropping most negatives would train the model on a distribution that never occurs
        # at inference time.
        if extra is not None:
            cand = extra(cand, s1, d, country, src)
        g = _group_of(cand["s1_pos"].to_numpy(), seed=a.seed)
        keep = np.isin(g, groups)
        take = keep & ((y == 1) | (rng.random(len(cand)) < sample_neg))
        cand = cand[take].reset_index(drop=True)
        y = y[take]
        cand = pairs.fill_missing(cand, cols)
        X = pairs.matrix(cand, s1, d, cols, chunk=a.feat_chunk)
        Xs.append(X)
        ys.append(y)
        cs.append(np.full(len(y), country, dtype=object))
        ps.append(cand["s1_pos"].to_numpy())
        total += len(y)
        log(f"  fit rows {country} src={src}: {len(y):,} ({int(y.sum()):,} pos)")
        del cand, s1, d, X
        gc.collect()
        if max_rows and total >= max_rows:
            log(f"  stopping at --max-fit-rows={max_rows:,}")
            break
    if not Xs:
        raise SystemExit("no fit rows found - run the earlier stages for --split fit first")
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(cs), np.concatenate(ps))


def stage_train_prune(a, w):
    cols = pairs.PRUNE_COLS
    X, y, country, s1_pos = _fit_rows(w, a, cols, "union", a.neg_rate,
                                      groups=(0, 1), max_rows=a.max_fit_rows)
    g = _group_of(s1_pos, seed=a.seed)
    tr, va = g == 0, g == 1
    log(f"pruner: {int(tr.sum()):,} train / {int(va.sum()):,} val rows, {len(cols)} features")
    model = gbdt.fit(X[tr], y[tr], X[va], y[va], name="pruner",
                     n_estimators=a.trees, leaves=a.leaves)
    log("  top features: " + gbdt.importances(model, cols))
    joblib.dump({"model": model, "cols": cols}, w.model("prune.joblib"))
    del X, y
    gc.collect()


def stage_prune(a, w):
    bundle = joblib.load(w.model("prune.joblib"))
    model, cols = bundle["model"], bundle["cols"]
    for country, src in _blocks(w, a.split):
        out = w.pruned(a.split, country, src)
        if done(out, a.force):
            continue
        p = w.cand("union", a.split, country, src)
        if not os.path.exists(p):
            continue
        cand = pd.read_parquet(p)
        s1 = _s1(w, a.split, country, FEAT_NEED)
        d = _d(w, a.split, country, src, FEAT_NEED)
        cand = pairs.fill_missing(cand, cols)
        # Blocks reach 47M pairs once dense retrieval is unioned in; one float32 matrix over
        # all of them is >10 GB, so score in row blocks and keep only the probabilities.
        prob1 = np.empty(len(cand), np.float32)
        for s in range(0, len(cand), PRUNE_BLOCK):
            e = min(s + PRUNE_BLOCK, len(cand))
            Xb = pairs.matrix(cand.iloc[s:e], s1, d, cols, chunk=a.feat_chunk)
            prob1[s:e] = gbdt.predict(model, Xb)
            del Xb
            gc.collect()
        cand["prob1"] = prob1
        n0 = len(cand)
        cand = cand[cand["prob1"].to_numpy() >= a.prune_floor]
        cand = cand.sort_values(["s1_pos", "prob1"], ascending=[True, False])
        cand = cand.groupby("s1_pos", sort=False).head(a.prune_keep).reset_index(drop=True)
        log(f"prune {a.split} {country} src={src}: {n0:,} -> {len(cand):,}")

        if a.expand:
            gid = sibling.group_ids(d, max_size=a.sib_max)
            new = sibling.expand(cand, gid, thr=a.expand_thr, per_s1=a.expand_per_s1)
            if len(new):
                new["src"] = np.int8(src)
                new = pairs.fill_missing(new, pairs.RETRIEVAL_COLS)
                q = _emb_path(w, a, "s1", _emb_key_s1(a.split, country))
                dp = _emb_path(w, a, "d", _emb_key_d(a.split, country, src))
                if os.path.exists(q + ".json") and os.path.exists(dp + ".json"):
                    new["cos_dense"] = dense.cos_for_pairs(q, dp, new["s1_pos"].to_numpy(),
                                                           new["d_pos"].to_numpy())
                both = pd.concat([cand.drop(columns=["prob1"]), new], ignore_index=True)
                both = pairs.add_group_features(both)
                both = pairs.fill_missing(both, cols)
                Xn = pairs.matrix(both, s1, d, cols, chunk=a.feat_chunk)
                both["prob1"] = gbdt.predict(model, Xn)
                del Xn
                cand = both.sort_values(["s1_pos", "prob1"], ascending=[True, False])
                cand = cand.groupby("s1_pos", sort=False).head(a.prune_keep + a.expand_per_s1)
                cand = cand.reset_index(drop=True)
                log(f"  view F added {len(new):,} sibling pairs -> {len(cand):,}")
        cand[[c for c in PRUNED_COLS if c in cand.columns]].to_parquet(
            out, index=False, compression="zstd")
        del cand, s1, d
        gc.collect()


def _ce_rows(w, a):
    """Judge training pairs: every positive that survived pruning, plus sampled negatives
    from the uncertain band. Those negatives are hard by construction -- both retrieval and
    the pruner already thought they were plausible."""
    rng = np.random.default_rng(a.seed)
    A, B, Y = [], [], []
    for country, src in _blocks(w, "fit"):
        p = w.pruned("fit", country, src)
        if not os.path.exists(p):
            continue
        cand = pd.read_parquet(p)
        s1 = _s1(w, "fit", country, FEAT_NEED + TEXT_NEED)
        d = _d(w, "fit", country, src, FEAT_NEED + TEXT_NEED, owner=True)
        y = pairs.label(cand, s1, d)
        band = crossenc.gate(cand["prob1"].to_numpy(), a.ce_lo, a.ce_hi)
        take = (y == 1) | (band & (rng.random(len(cand)) < a.ce_neg_rate))
        cand, y = cand[take].reset_index(drop=True), y[take]
        ta, tb = crossenc.build_texts(cand, s1, d)
        A.append(ta)
        B.append(tb)
        Y.append(y)
        log(f"  ce rows {country} src={src}: {len(y):,} ({int(y.sum()):,} pos)")
        del cand, s1, d
        gc.collect()
    if not A:
        raise SystemExit("no pruned fit blocks - run `prune --split fit` first")
    ta, tb, y = np.concatenate(A), np.concatenate(B), np.concatenate(Y)
    if a.ce_max_rows and len(y) > a.ce_max_rows:
        idx = rng.choice(len(y), a.ce_max_rows, replace=False)
        ta, tb, y = ta[idx], tb[idx], y[idx]
    return ta, tb, y.astype(np.float32)


def stage_train_ce(a, w):
    ta, tb, y = _ce_rows(w, a)
    if a.pseudo and os.path.exists(w.p("pseudo.parquet")):
        ps = pd.read_parquet(w.p("pseudo.parquet"))
        ta = np.concatenate([ta, ps["text_a"].to_numpy(dtype=object)])
        tb = np.concatenate([tb, ps["text_b"].to_numpy(dtype=object)])
        y = np.concatenate([y, ps["y"].to_numpy(np.float32)])
        log(f"  mixed in {len(ps):,} France pseudo-labelled pairs")
    log(f"cross-encoder training set: {len(y):,} pairs ({int(y.sum()):,} positive)")
    crossenc.train(w.model("ce"), ta, tb, y, a)


def stage_ce(a, w):
    model_dir = w.model("ce")
    if not os.path.isdir(model_dir):
        raise SystemExit("no judge at models/ce - run `train-ce` first")
    judge = None
    for country, src in _blocks(w, a.split):
        out = w.ce(a.split, country, src)
        if done(out, a.force):
            continue
        p = w.pruned(a.split, country, src)
        if not os.path.exists(p):
            continue
        cand = pd.read_parquet(p, columns=pairs.LEAN + ["prob1"])
        band = crossenc.gate(cand["prob1"].to_numpy(), a.ce_lo, a.ce_hi)
        sub = cand[band].reset_index(drop=True)
        log(f"ce {a.split} {country} src={src}: {len(sub):,}/{len(cand):,} pairs in the band")
        if sub.empty:
            sub = sub.assign(ce_logit=np.zeros(0, np.float32))
            sub[pairs.LEAN + ["ce_logit"]].to_parquet(out, index=False)
            continue
        if judge is None:  # loaded once for the whole pass, not per block
            judge = crossenc.Judge(model_dir, a)
        s1 = _s1(w, a.split, country, TEXT_NEED)
        d = prep.load_d(w, a.split, country, src, TEXT_NEED)
        logits = np.empty(len(sub), np.float32)
        # Texts are built per chunk: materialising 7M name+address strings at once costs
        # most of a gigabyte before the first batch runs.
        for s in range(0, len(sub), a.ce_chunk):
            e = min(s + a.ce_chunk, len(sub))
            ta, tb = crossenc.build_texts(sub.iloc[s:e], s1, d)
            logits[s:e] = judge.score(ta, tb)
            del ta, tb
            gc.collect()
        sub["ce_logit"] = logits
        sub[pairs.LEAN + ["ce_logit"]].to_parquet(out, index=False, compression="zstd")
        del cand, sub, s1, d, logits
        gc.collect()


def _attach_late(w, a, split, cand, s1, d, country, src):
    """stage-2 competition features, the judge logit and the sibling features.

    Order matters: the sibling features are computed from stage-1 probabilities, and the
    judge logit is merged before the stacker so absence is explicit (``ce_run=0``) rather
    than silently zero.
    """
    cep = w.ce(split, country, src)
    cand = pairs.add_stage2(cand, prob_col="prob1")
    if os.path.exists(cep):
        ce = pd.read_parquet(cep)
        cand = cand.merge(ce, on=pairs.LEAN, how="left")
        cand["ce_run"] = (~cand["ce_logit"].isna()).astype(np.float32)
        cand["ce_logit"] = cand["ce_logit"].fillna(0.0).astype(np.float32)
    else:
        cand["ce_logit"] = np.float32(0.0)
        cand["ce_run"] = np.float32(0.0)
    gid = sibling.group_ids(d, max_size=a.sib_max)
    cand = sibling.add_features(cand, gid, prob_col="prob1")
    return _add_other_source(w, split, cand, country, src)


def _add_other_source(w, split, cand, country, src):
    """oth_best / oth_cnt: what this S1's candidates in the *other* source scored."""
    other = 3 if src == 2 else 2
    p = w.pruned(split, country, other)
    cand["oth_best"] = np.float32(0.0)
    cand["oth_cnt"] = np.float32(0.0)
    if not os.path.exists(p):
        return cand
    o = pd.read_parquet(p, columns=["s1_pos", "prob1"])
    g = o.groupby("s1_pos")["prob1"]
    best = g.max()
    cnt = (o["prob1"] >= 0.5).groupby(o["s1_pos"]).sum()
    sp = cand["s1_pos"].to_numpy()
    cand["oth_best"] = best.reindex(sp).fillna(0.0).to_numpy(np.float32)
    cand["oth_cnt"] = cnt.reindex(sp).fillna(0.0).to_numpy(np.float32)
    del o
    return cand


def stage_train_stack(a, w):
    cols = pairs.STACK_COLS
    X, y, country, s1_pos = _fit_rows(w, a, cols, "pruned", a.stack_neg_rate,
                                      groups=(0, 1, 2), max_rows=a.max_fit_rows,
                                      extra=lambda c, s1, d, co, sr: _attach_late(w, a, "fit", c, s1, d, co, sr))
    g = _group_of(s1_pos, seed=a.seed)
    tr, va, cal = g == 0, g == 1, g == 2
    log(f"stacker: {int(tr.sum()):,} train / {int(va.sum()):,} val / {int(cal.sum()):,} cal rows")
    model = gbdt.fit(X[tr], y[tr], X[va], y[va], name="stacker",
                     n_estimators=a.trees, leaves=a.leaves)
    log("  top features: " + gbdt.importances(model, cols))
    p_cal = gbdt.predict(model, X[cal])
    calib = gbdt.Calibrator().fit(p_cal, y[cal], country[cal])
    joblib.dump({"model": model, "cols": cols, "cal": calib}, w.model("stack.joblib"))
    del X, y
    gc.collect()


def stage_stack(a, w):
    bundle = joblib.load(w.model("stack.joblib"))
    model, cols, calib = bundle["model"], bundle["cols"], bundle["cal"]
    for country, src in _blocks(w, a.split):
        out = w.scored(a.split, country, src)
        if done(out, a.force):
            continue
        p = w.pruned(a.split, country, src)
        if not os.path.exists(p):
            continue
        cand = pd.read_parquet(p)
        s1 = _s1(w, a.split, country, FEAT_NEED)
        d = _d(w, a.split, country, src, FEAT_NEED)
        cand = _attach_late(w, a, a.split, cand, s1, d, country, src)
        cand = pairs.fill_missing(cand, cols)
        X = pairs.matrix(cand, s1, d, cols, chunk=a.feat_chunk)
        raw = gbdt.predict(model, X)
        del X
        gc.collect()
        cand["prob"] = calib.apply(raw, country)
        cand[pairs.LEAN + ["src", "prob"]].to_parquet(out, index=False, compression="zstd")
        log(f"stack {a.split} {country} src={src}: {len(cand):,} scored pairs")
        del cand, s1, d
        gc.collect()


def _scored_country(w, split, country):
    frames = []
    for src in SOURCES:
        p = w.scored(split, country, src)
        if os.path.exists(p):
            frames.append(pd.read_parquet(p))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _truth_country(w, split, country, s1):
    """Ground truth restricted to this split's S1 entities.

    The filter is not an optimisation. The train pool holds 7,638,365 matched records; a
    dict over all of them costs several GB, while the hold-out only ever asks about its own
    150k entities.
    """
    want = pd.Index(s1["entity_id"].to_numpy())
    truth = {e: set() for e in want}
    for src in SOURCES:
        d = _d(w, split, country, src, ["entity_id"], owner=True)
        owner = d["owner"]
        m = owner.isin(want).to_numpy()
        for o, i in zip(owner.to_numpy()[m], d["entity_id"].to_numpy()[m]):
            truth[o].add(i)
        del d, owner
        gc.collect()
    return truth


def _score_split(w, a, mode, thr, margin, miss_rate):
    per, countries = [], []
    for country in w.countries(a.split):
        scored = _scored_country(w, a.split, country)
        if scored is None:
            continue
        s1 = _s1(w, a.split, country, ["entity_id"])
        ids_by_src = {src: prep.load_d(w, a.split, country, src, ["entity_id"])["entity_id"].to_numpy()
                      for src in SOURCES}
        sel = resolve.select(scored, mode, thr, margin, miss_rate, cap=a.cap)
        pred = resolve.to_lists(sel, s1["entity_id"].to_numpy(), ids_by_src)
        truth = _truth_country(w, a.split, country, s1)
        vals = np.array([f05_single(set(pred.get(e, ())), truth[e])
                         for e in s1["entity_id"].to_numpy()])
        per.append(vals)
        countries.append((country, vals, s1, truth, pred))
        del scored, sel
        gc.collect()
    if not per:
        raise SystemExit("nothing scored - run `stack` for this split first")
    allv = np.concatenate(per)
    return allv, countries


def stage_tune(a, w):
    best = ("threshold", 0.5, 0.0, -1.0)
    for thr in a.thr_grid:
        for margin in a.margin_grid:
            v, _ = _score_split(w, a, "threshold", thr, margin, a.miss_rate)
            log(f"  thr={thr} margin={margin}: F0.5={v.mean():.4f}")
            if v.mean() > best[3]:
                best = ("threshold", thr, margin, float(v.mean()))
    v, _ = _score_split(w, a, "expected", 0.0, a.expected_margin, a.miss_rate)
    log(f"  expected-F0.5 (margin={a.expected_margin}): F0.5={v.mean():.4f}")
    if v.mean() > best[3]:
        best = ("expected", 0.0, a.expected_margin, float(v.mean()))
    log(f"BEST mode={best[0]} thr={best[1]} margin={best[2]} F0.5={best[3]:.4f}")
    w.write_json({"mode": best[0], "thr": best[1], "margin": best[2],
                  "tuned_f05": best[3], "miss_rate": a.miss_rate, "cap": a.cap,
                  "split": a.split}, "models", "decision.json")


def stage_score(a, w):
    cfg = w.read_json("models", "decision.json") or {"mode": "expected", "thr": 0.0, "margin": 0.0}
    allv, countries = _score_split(w, a, cfg["mode"], cfg["thr"], cfg["margin"],
                                   cfg.get("miss_rate", a.miss_rate))
    log(f"{a.split.upper()} macro F0.5 = {allv.mean():.4f}  over {len(allv):,} S1 entities")
    for country, vals, s1, truth, pred in countries:
        nt = np.array([len(truth[e]) for e in s1["entity_id"].to_numpy()])
        got = np.array([1 if pred.get(e) else 0 for e in s1["entity_id"].to_numpy()])
        log(f"  [{country}] F0.5={vals.mean():.4f}  n={len(vals):,}  "
            f"predicted>=1: {got.mean():.4f}  true>=1: {(nt > 0).mean():.4f}")
        for k in range(0, 7):
            m = (nt == k) if k < 6 else (nt >= 6)
            if m.any():
                log(f"    #true={k}{'+' if k == 6 else ''}: F0.5={vals[m].mean():.4f} (n={int(m.sum()):,})")


def stage_errors(a, w):
    cfg = w.read_json("models", "decision.json") or {"mode": "expected", "thr": 0.0, "margin": 0.0}
    rows = []
    fp_total = 0
    for country in w.countries(a.split):
        scored = _scored_country(w, a.split, country)
        if scored is None:
            continue
        s1 = _s1(w, a.split, country, ["entity_id"])
        sel = resolve.select(scored, cfg["mode"], cfg["thr"], cfg["margin"],
                             cfg.get("miss_rate", 0.0), cap=a.cap)
        sel_key = sel[pairs.LEAN + ["src"]].assign(_sel=np.int8(1))
        scored_key = scored[pairs.LEAN + ["src"]].assign(_scored=np.int8(1))
        for src in SOURCES:
            d = _d(w, a.split, country, src, ["entity_id", "native", "addr_missing"], owner=True)
            truth = diag.true_pairs(s1, d, src)
            up = w.cand("union", a.split, country, src)
            pp = w.pruned(a.split, country, src)
            t = truth
            if os.path.exists(up):
                u = pd.read_parquet(up, columns=pairs.LEAN).assign(_union=np.int8(1))
                t = t.merge(u, on=pairs.LEAN, how="left")
            if os.path.exists(pp):
                pr = pd.read_parquet(pp, columns=pairs.LEAN).assign(_pruned=np.int8(1))
                t = t.merge(pr, on=pairs.LEAN, how="left")
            t = t.merge(scored_key, on=pairs.LEAN + ["src"], how="left")
            t = t.merge(sel_key, on=pairs.LEAN + ["src"], how="left")
            bucket = np.where(t.get("_union", pd.Series(np.nan, index=t.index)).isna(), "a_not_retrieved",
                              np.where(t.get("_pruned", pd.Series(np.nan, index=t.index)).isna(), "b_pruned_away",
                                       np.where(t["_sel"].isna(), "c_scored_low", "hit")))
            t["bucket"] = bucket
            t["country"] = country
            t["b_native"] = d["native"].to_numpy()[t["d_pos"].to_numpy()]
            t["b_missing"] = d["addr_missing"].to_numpy()[t["d_pos"].to_numpy()]
            nt = truth.groupby("s1_pos").size()
            t["n_true_bin"] = diag.bin_n_true(t["s1_pos"].map(nt).fillna(0).to_numpy(np.int64))
            rows.append(t[["country", "src", "bucket", "b_native", "b_missing", "n_true_bin"]])
            del d, truth, t
            gc.collect()
        # false positives: selected pairs that are not true pairs
        hit = 0
        for src in SOURCES:
            d = _d(w, a.split, country, src, ["entity_id"], owner=True)
            tr = diag.true_pairs(s1, d, src).assign(_true=np.int8(1))
            s = sel[sel["src"] == src].merge(tr, on=pairs.LEAN + ["src"], how="left")
            hit += int(s["_true"].notna().sum())
            fp_total += int(s["_true"].isna().sum())
            del d, tr, s
        log(f"  [{country}] selected pairs correct: {hit:,}")
        del scored, sel
        gc.collect()
    if rows:
        diag.miss_taxonomy(pd.concat(rows, ignore_index=True))
    log(f"FALSE POSITIVES among selected pairs: {fp_total:,}")


def stage_resolve(a, w):
    cfg = w.read_json("models", "decision.json")
    if not cfg:
        log("no models/decision.json - falling back to expected-F0.5 with no margin")
        cfg = {"mode": "expected", "thr": 0.0, "margin": 0.0, "miss_rate": a.miss_rate}
    os.makedirs(a.out, exist_ok=True)
    s1_all = read_source(f"{a.data_dir}/test/test_source1.tsv")["entity_id"].to_numpy()
    if a.limit_s1:
        s1_all = s1_all[:a.limit_s1]

    matches = {}
    parts = []
    for country in w.countries(a.split):
        s1 = _s1(w, a.split, country, ["entity_id"])
        s1_ids = s1["entity_id"].to_numpy()
        scored = _scored_country(w, a.split, country)
        ids_by_src = {}
        for src in SOURCES:
            ids_by_src[src] = prep.load_d(w, a.split, country, src, ["entity_id"])["entity_id"].to_numpy()
        if scored is not None:
            sel = resolve.select(scored, cfg["mode"], cfg["thr"], cfg["margin"],
                                 cfg.get("miss_rate", 0.0), cap=a.cap)
            matches.update(resolve.to_lists(sel, s1_ids, ids_by_src))
            log(f"resolve [{country}]: {len(sel):,} matches over {len(s1_ids):,} S1")
            del sel
        # candidate_pairs.tsv, one part per source, streamed and merged afterwards
        cparts = []
        for src in SOURCES:
            p = w.cand("union", a.split, country, src)
            if not os.path.exists(p):
                continue
            part = w.p("logs", f"candpart_{country}_{src}.txt")
            if not done(part, a.force):
                u = pd.read_parquet(p, columns=pairs.LEAN)
                if scored is not None:
                    # View-F sibling pairs are added after the union, so a final match can
                    # sit outside it; the candidate list has to contain every match.
                    extra = scored.loc[scored["src"] == src, pairs.LEAN]
                    u = pd.concat([u, extra], ignore_index=True).drop_duplicates()
                resolve.write_candidate_part(part, len(s1_ids), u["s1_pos"].to_numpy(),
                                             u["d_pos"].to_numpy(), ids_by_src[src])
                del u
                gc.collect()
            cparts.append(part)
        parts.append((s1_ids, cparts))
        del scored, s1, ids_by_src
        gc.collect()

    mpath = os.path.join(a.out, "matching_results.tsv")
    resolve.write_matching(mpath, s1_all, matches)
    n = sum(1 for s in s1_all if matches.get(s))
    log(f"wrote {mpath}: {n:,}/{len(s1_all):,} S1 with >=1 match (train prior ~0.944)")
    del matches
    gc.collect()
    cpath = os.path.join(a.out, "candidate_pairs.tsv")
    first = True
    for s1_ids, cparts in parts:
        resolve.merge_candidate_parts(cpath, [(s1_ids, cparts)], append=not first)
        first = False
    log(f"wrote {cpath}")


def stage_pseudo(a, w):
    """France pseudo-labels: confident, mutually-best test pairs become training data.

    France is 15% of the test S1 rows and has zero labels, so this is the only way to give
    either neural stage a France-shaped gradient. The filter is deliberately strict -- a
    calibrated probability above ``--pseudo-thr`` *and* mutual best match -- because a wrong
    pseudo-label teaches the model the noise pattern instead of the signal.
    """
    out = []
    for country in w.countries(a.split):
        if a.pseudo_country and country != a.pseudo_country:
            continue
        scored = _scored_country(w, a.split, country)
        if scored is None:
            continue
        best_d = scored.sort_values("prob", ascending=False).drop_duplicates(["src", "d_pos"])
        best_s = scored.sort_values("prob", ascending=False).drop_duplicates(["s1_pos"])
        mutual = best_d.merge(best_s, on=pairs.LEAN + ["src"], how="inner")
        conf = mutual[mutual["prob_x"].to_numpy() >= a.pseudo_thr]
        log(f"  [{country}] {len(conf):,} confident mutual-best pairs")
        for src in SOURCES:
            sub = conf[conf["src"] == src]
            if sub.empty:
                continue
            s1 = _s1(w, a.split, country, TEXT_NEED)
            d = prep.load_d(w, a.split, country, src, TEXT_NEED)
            ta, tb = crossenc.build_texts(sub, s1, d)
            out.append(pd.DataFrame({"text_a": ta, "text_b": tb, "y": np.float32(1.0)}))
            # Negatives: the same S1's other candidates, which are plausible but not the
            # mutual best. Without them the model only ever sees agreement.
            low = scored[(scored["src"] == src) & (scored["prob"].to_numpy() < a.pseudo_neg_hi)]
            low = low[low["s1_pos"].isin(sub["s1_pos"])].groupby("s1_pos", sort=False).head(2)
            if len(low):
                na, nb = crossenc.build_texts(low, s1, d)
                out.append(pd.DataFrame({"text_a": na, "text_b": nb, "y": np.float32(0.0)}))
            del s1, d
            gc.collect()
        del scored
        gc.collect()
    if out:
        df = pd.concat(out, ignore_index=True)
        df.to_parquet(w.p("pseudo.parquet"), index=False)
        log(f"wrote {w.p('pseudo.parquet')}: {len(df):,} rows "
            f"({int(df.y.sum()):,} positive)")


def stage_train_bi(a, w):
    """Mine anchors/positives/hard negatives from the fit split, then fine-tune."""
    rng = np.random.default_rng(a.seed)
    A, P, N = [], [], []
    for country, src in _blocks(w, "fit"):
        p = w.cand("union", "fit", country, src)
        if not os.path.exists(p):
            continue
        cand = pd.read_parquet(p, columns=pairs.LEAN + ["rrf"])
        s1 = _s1(w, "fit", country, FEAT_NEED + TEXT_NEED)
        d = _d(w, "fit", country, src, FEAT_NEED + TEXT_NEED, owner=True)
        y = pairs.label(cand, s1, d)
        s1_text = crossenc.pair_texts(s1)
        d_text = crossenc.pair_texts(d)
        pos = cand[y == 1]
        if pos.empty:
            del cand, s1, d
            continue
        # Hard negatives: the highest-fused wrong candidates for the same anchor.
        neg = cand[y == 0].sort_values(["s1_pos", "rrf"], ascending=[True, False])
        neg = neg.groupby("s1_pos", sort=False).head(a.bi_hard)
        bysrc = neg.groupby("s1_pos")["d_pos"].apply(list).to_dict()
        take = rng.permutation(len(pos))[:a.bi_per_block]
        pos = pos.iloc[np.sort(take)]
        anchors = s1_text[pos["s1_pos"].to_numpy()]
        positives = d_text[pos["d_pos"].to_numpy()]
        negs = np.full((len(pos), a.bi_hard), "", dtype=object)
        for i, sp in enumerate(pos["s1_pos"].to_numpy()):
            cands = bysrc.get(sp, ())
            for j, dpos in enumerate(cands[:a.bi_hard]):
                negs[i, j] = d_text[dpos]
        A.append(anchors)
        P.append(positives)
        N.append(negs)
        log(f"  bi rows {country} src={src}: {len(pos):,} anchors")
        del cand, s1, d
        gc.collect()
    if not A:
        raise SystemExit("no fit unions - run `lexical`/`union --split fit` first")
    anchors = np.concatenate(A)
    positives = np.concatenate(P)
    negatives = np.concatenate(N)
    if a.bi_max_rows and len(anchors) > a.bi_max_rows:
        idx = rng.choice(len(anchors), a.bi_max_rows, replace=False)
        anchors, positives, negatives = anchors[idx], positives[idx], negatives[idx]
    log(f"bi-encoder training set: {len(anchors):,} anchors")
    train_bi.train(w.model("bi"), anchors, positives, negatives, a)


STAGES = {
    "prep": stage_prep, "lexical": stage_lexical, "embed": stage_embed, "dense": stage_dense,
    "union": stage_union, "diag": stage_diag, "train-bi": stage_train_bi,
    "train-prune": stage_train_prune, "prune": stage_prune, "train-ce": stage_train_ce,
    "ce": stage_ce, "train-stack": stage_train_stack, "stack": stage_stack,
    "tune": stage_tune, "score": stage_score, "errors": stage_errors,
    "resolve": stage_resolve, "pseudo": stage_pseudo,
}


def build_parser():
    ap = argparse.ArgumentParser(description="v3 entity-resolution pipeline")
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--data-dir", default="../../student_resource/dataset")
    ap.add_argument("--work-dir", default="../../work3")
    ap.add_argument("--out", default="../../output3")
    ap.add_argument("--split", default="test", help="fit | holdout | test | all (prep only)")
    ap.add_argument("--force", action="store_true", help="recompute shards that already exist")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-s1", type=int, default=0, help="quick dry run")

    g = ap.add_argument_group("prep")
    g.add_argument("--fit-frac", type=float, default=0.12, help="share of train S1 used to fit")
    g.add_argument("--holdout-n", type=int, default=150_000)
    g.add_argument("--seg-min-count", type=int, default=10,
                   help="minimum Source-1 frequency for a word in the segmentation vocabulary")
    g.add_argument("--no-segment", action="store_true", help="disable glued-name segmentation")

    g = ap.add_argument_group("lexical retrieval")
    g.add_argument("--k", type=int, default=25, help="top-k per view")
    g.add_argument("--keep", type=int, default=45, help="cap per S1 after fusing the sparse views")
    g.add_argument("--threads", type=int, default=4)
    g.add_argument("--doc-shard", type=int, default=400_000)
    g.add_argument("--q-chunk", type=int, default=512)
    g.add_argument("--max-df-frac", type=float, default=0.0015)
    g.add_argument("--vocab-sample", type=int, default=400_000)
    g.add_argument("--key-cap", type=int, default=50)

    g = ap.add_argument_group("bi-encoder")
    g.add_argument("--bi-model", default=dense.DEFAULT_MODEL)
    g.add_argument("--emb-tag", default="e5s", help="namespace for the embedding files")
    g.add_argument("--dim", type=int, default=0, help="truncate embeddings to this width (0 = full)")
    g.add_argument("--batch", type=int, default=256)
    g.add_argument("--max-len", type=int, default=64)
    g.add_argument("--flush-every", type=int, default=200)
    g.add_argument("--k-dense", type=int, default=25)
    g.add_argument("--q-block", type=int, default=4096)
    g.add_argument("--d-block", type=int, default=65_536)
    g.add_argument("--bi-batch", type=int, default=24)
    g.add_argument("--bi-epochs", type=int, default=1)
    g.add_argument("--bi-lr", type=float, default=2e-5)
    g.add_argument("--bi-max-len", type=int, default=64)
    g.add_argument("--bi-temp", type=float, default=0.05)
    g.add_argument("--bi-hard", type=int, default=2)
    g.add_argument("--bi-per-block", type=int, default=200_000)
    g.add_argument("--bi-max-rows", type=int, default=600_000)
    g.add_argument("--bi-log-every", type=int, default=200)
    g.add_argument("--bi-grad-ckpt", action="store_true")

    g = ap.add_argument_group("candidates and tabular models")
    g.add_argument("--keep-union", type=int, default=60)
    g.add_argument("--neg-rate", type=float, default=0.25)
    g.add_argument("--stack-neg-rate", type=float, default=0.5)
    g.add_argument("--max-fit-rows", type=int, default=6_000_000,
                   help="cap on labelled rows per model fit; the concatenate doubles peak")
    g.add_argument("--trees", type=int, default=1500)
    g.add_argument("--leaves", type=int, default=127)
    g.add_argument("--feat-chunk", type=int, default=1_000_000)
    g.add_argument("--prune-floor", type=float, default=0.02)
    g.add_argument("--prune-keep", type=int, default=8)

    g = ap.add_argument_group("cross-encoder judge")
    g.add_argument("--ce-model", default=crossenc.DEFAULT_MODEL)
    g.add_argument("--ce-lo", type=float, default=0.03)
    g.add_argument("--ce-hi", type=float, default=0.97)
    g.add_argument("--ce-batch", type=int, default=32)
    g.add_argument("--ce-infer-batch", type=int, default=128)
    g.add_argument("--ce-epochs", type=int, default=1)
    g.add_argument("--ce-lr", type=float, default=2e-5)
    g.add_argument("--ce-max-len", type=int, default=128)
    g.add_argument("--ce-neg-rate", type=float, default=0.35)
    g.add_argument("--ce-max-rows", type=int, default=1_500_000)
    g.add_argument("--ce-log-every", type=int, default=200)
    g.add_argument("--ce-chunk", type=int, default=1_000_000,
                   help="pairs per text-building chunk during gated inference")

    g = ap.add_argument_group("sibling graph and decision")
    g.add_argument("--sib-max", type=int, default=12)
    g.add_argument("--expand", action="store_true", help="view F: sibling expansion")
    g.add_argument("--expand-thr", type=float, default=0.9)
    g.add_argument("--expand-per-s1", type=int, default=4)
    g.add_argument("--cap", type=int, default=11, help="max matches per S1 (train maximum)")
    g.add_argument("--miss-rate", type=float, default=0.03,
                   help="1 - candidate recall, from `diag`; feeds expected-F0.5")
    g.add_argument("--expected-margin", type=float, default=0.0)
    g.add_argument("--thr-grid", type=float, nargs="*",
                   default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])
    g.add_argument("--margin-grid", type=float, nargs="*", default=[0.0, 0.1, 0.2])

    g = ap.add_argument_group("France pseudo-labelling")
    g.add_argument("--pseudo", action="store_true", help="mix pseudo-labels into train-ce")
    g.add_argument("--pseudo-thr", type=float, default=0.98)
    g.add_argument("--pseudo-neg-hi", type=float, default=0.5)
    g.add_argument("--pseudo-country", default="France")
    return ap


def main():
    a = build_parser().parse_args()
    w = Work(a.work_dir)
    log(f"stage={a.stage} split={a.split} work={w.root}")
    STAGES[a.stage](a, w)
    log(f"stage={a.stage} done")


if __name__ == "__main__":
    main()
