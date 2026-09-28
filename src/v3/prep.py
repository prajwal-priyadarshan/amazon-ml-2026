"""L0 data layer: normalise once, write per-(country, source) Parquet shards.

Two text views are kept side by side and they are used by different stages on purpose:

* the **normalised** view (``name_lat``/``name_core``/``name_skel``/``addr_norm``/``nums``)
  is Latin-folded and abbreviation-canonicalised. Sparse retrieval and the rapidfuzz
  features need that, because character n-grams and edit distances only line up once the
  strings are folded.
* the **raw** view (``business_name``/``business_address``) is kept untouched for the
  neural stages. A multilingual encoder relates "राम मार्केटिंग" to "Ram Marketing"
  directly, whereas the folded form ("raam maarkettinng") destroys exactly the signal the
  encoder was pretrained to use. Folding before an mBERT-family model is a loss, not a
  normalisation.

For the train pool each S2/S3 shard also carries an ``owner`` column (the S1 entity that
owns the record, or ""), so every later stage can label a candidate pair with a vectorised
string comparison instead of a 7.6M-entry Python dict.
"""
import gc
import os

import numpy as np
import pandas as pd

from .. import textnorm
from ..data import read_ground_truth, read_source
from . import segment
from .paths import SOURCES, log, pool_of

KEEP = ["entity_id", "country", "business_name", "business_address",
        "name_lat", "name_core", "name_key", "name_skel", "legal", "native",
        "addr_norm", "nums", "addr_missing", "name_seg", "seg_key"]

_SEG = None


def _owner_series(gt_path):
    """matched S2/S3 id -> owning S1 id, as an Arrow-backed Series (7.6M rows)."""
    gt = read_ground_truth(gt_path)
    ids, own = [], []
    for s, ms in gt.items():
        for m in ms:
            ids.append(m)
            own.append(s)
    return pd.Series(own, index=pd.Index(ids, name="entity_id"), dtype="string")


def _add_segmented(part):
    """name_seg: name_core with glued tokens split. seg_key: its sorted unique token set."""
    seg = _SEG
    if seg is None:
        part["name_seg"] = part["name_core"]
        part["seg_key"] = part["name_key"]
        return part
    vals = [seg.text(t) for t in part["name_core"].to_numpy()]
    part["name_seg"] = vals
    part["seg_key"] = [" ".join(sorted(set(v.split()))) for v in vals]
    return part


def _write_shards(df, jobs, path_of, owner=None):
    """Normalise and write one Parquet file per country. Slice first, so the normaliser's
    worker pool only ever copies one country at a time."""
    for country, part in df.groupby("country", sort=True):
        out = path_of(country)
        part = part.reset_index(drop=True)
        part = textnorm.normalize_frame(part, n_jobs=jobs)
        part = _add_segmented(part)
        if owner is not None:
            part["owner"] = owner.reindex(part["entity_id"].to_numpy()).fillna("").to_numpy()
        cols = KEEP + (["owner"] if owner is not None else [])
        part[cols].to_parquet(out, index=False, compression="zstd")
        log(f"  {os.path.basename(out)}: {len(part):,} rows")
        del part
        gc.collect()


def run(a, w):
    """Build shards for the S1 split(s) requested and for their record pool."""
    global _SEG
    splits = [a.split] if a.split != "all" else ["fit", "holdout", "test"]
    pools = sorted({pool_of(s) for s in splits})

    # The segmentation vocabulary is mined from the Source-1 names of the provided files and
    # nothing else, and it must be identical for every shard, so it is built once up front.
    vpath = w.p("prep", "vocab.json")
    vocab = segment.load_vocab(vpath)
    if not vocab and not a.no_segment:
        names = []
        for f in (f"{a.data_dir}/train/train_source1.tsv", f"{a.data_dir}/test/test_source1.tsv"):
            if os.path.exists(f):
                names.append(read_source(f)["business_name"].to_numpy())
        vocab = segment.build_vocab(names, min_count=a.seg_min_count)
        segment.save_vocab(vocab, vpath)
        del names
        gc.collect()
    _SEG = segment.Segmenter(vocab) if vocab and not a.no_segment else None

    for pool in pools:
        pre = f"{a.data_dir}/{'test' if pool == 'test' else 'train'}/{'test' if pool == 'test' else 'train'}"
        owner = _owner_series(f"{a.data_dir}/train/train_ground_truth.tsv") if pool == "train" else None
        if owner is not None:
            log(f"ground truth: {len(owner):,} matched records")
        for src in SOURCES:
            marker = w.p("prep", f"done_{pool}_{src}.json")
            if os.path.exists(marker) and not a.force:
                log(f"pool={pool} src={src}: shards present, skipping")
                continue
            df = read_source(f"{pre}_source{src}.tsv")
            log(f"pool={pool} src={src}: {len(df):,} raw rows")
            _write_shards(df, a.jobs, lambda c, p=pool, s=src: w.d(p, c, s), owner)
            w.write_json({"rows": int(len(df))}, "prep", f"done_{pool}_{src}.json")
            del df
            gc.collect()
        del owner
        gc.collect()

    # S1 side. fit and holdout are disjoint samples of train_source1 drawn once, with the
    # split recorded in meta.json so every later stage agrees on who was trained on.
    for split in splits:
        if pool_of(split) == "test":
            s1 = read_source(f"{a.data_dir}/test/test_source1.tsv")
            if a.limit_s1:
                s1 = s1.iloc[:a.limit_s1]
        else:
            allrows = read_source(f"{a.data_dir}/train/train_source1.tsv")
            rng = np.random.default_rng(a.seed)
            perm = rng.permutation(len(allrows))
            n_fit = int(a.fit_frac * len(allrows))
            if split == "fit":
                take = perm[:n_fit]
            elif split == "hold2":  # a second never-trained-on sample, disjoint from both
                lo = n_fit + a.holdout_n
                take = perm[lo:lo + a.hold2_n]
            elif split == "hold3":  # a third, after hold2
                lo = n_fit + a.holdout_n + a.hold2_n
                take = perm[lo:lo + a.hold3_n]
            else:
                take = perm[n_fit:n_fit + a.holdout_n]
            if a.limit_s1:
                take = take[:a.limit_s1]
            s1 = allrows.iloc[np.sort(take)].reset_index(drop=True)
            del allrows
        log(f"split={split}: {len(s1):,} S1 rows")
        _write_shards(s1, a.jobs, lambda c, sp=split: w.s1(sp, c))
        del s1
        gc.collect()

    if a.split in ("all", "fit", "holdout"):
        w.write_json({"seed": a.seed, "fit_frac": a.fit_frac, "holdout_n": a.holdout_n},
                     "prep", "meta.json")


def _countries_of(path):
    """Cheap country list without loading the file body."""
    seen = set()
    for chunk in pd.read_csv(path, sep="\t", usecols=["country"], dtype=str, chunksize=2_000_000):
        seen.update(chunk["country"].unique())
    return sorted(seen)


def load_s1(w, split, country, cols=None):
    return pd.read_parquet(w.s1(split, country), columns=cols)


def load_d(w, split, country, src, cols=None):
    return pd.read_parquet(w.d(pool_of(split), country, src), columns=cols)
