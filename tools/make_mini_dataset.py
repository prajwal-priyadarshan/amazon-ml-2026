"""Carve a small, self-consistent dataset out of the real one, for smoke tests.

A dry run on the full data costs hours before the first bug shows up. This builds a mini
dataset with the same schema and the same shape of noise, so every stage of the pipeline can
be exercised end to end in a couple of minutes:

* train: ``--n-s1`` Source-1 entities, *all* of their true matches, plus ``--decoys``
  unmatched records per source so precision still has something to get wrong.
* test: a slice of the real test files, including France rows, so the unseen-country path
  is exercised too.

    python tools/make_mini_dataset.py --data-dir student_resource/dataset \
        --out /tmp/mini --n-s1 4000 --decoys 30000

The ground-truth file is rewritten to cover exactly the S1 rows kept, so hold-out scoring
on the mini set is meaningful (small, but not wrong).
"""
import argparse
import csv
import os

import pandas as pd


def read(path, nrows=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, nrows=nrows)


def write(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-s1", type=int, default=4000)
    ap.add_argument("--decoys", type=int, default=30_000)
    ap.add_argument("--test-rows", type=int, default=4000)
    ap.add_argument("--test-pool", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tr = f"{a.data_dir}/train"
    gt = read(f"{tr}/train_ground_truth.tsv")
    s1 = read(f"{tr}/train_source1.tsv")

    # Keep a country-balanced sample so both regimes (short reordered US addresses, long
    # near-copy India addresses) are present.
    per = max(a.n_s1 // max(s1.country.nunique(), 1), 1)
    keep = pd.concat([g.sample(min(len(g), per), random_state=a.seed)
                      for _, g in s1.groupby("country", sort=True)], ignore_index=True)
    keep_ids = set(keep.entity_id)
    gt_keep = gt[gt.iloc[:, 0].isin(keep_ids)]
    wanted = set()
    for v in gt_keep.iloc[:, 1]:
        if v:
            wanted.update(v.split(","))
    print(f"S1 kept: {len(keep):,}  true matches wanted: {len(wanted):,}")

    for src in (2, 3):
        full = read(f"{tr}/train_source{src}.tsv")
        hit = full[full.entity_id.isin(wanted)]
        rest = full[~full.entity_id.isin(wanted)].sample(
            min(a.decoys, len(full) - len(hit)), random_state=a.seed)
        out = pd.concat([hit, rest], ignore_index=True).sample(frac=1.0, random_state=a.seed)
        write(out, f"{a.out}/train/train_source{src}.tsv")
        print(f"  train_source{src}: {len(out):,} rows ({len(hit):,} true, {len(rest):,} decoys)")
        del full, hit, rest, out

    write(keep, f"{a.out}/train/train_source1.tsv")
    write(gt_keep, f"{a.out}/train/train_ground_truth.tsv")

    te = f"{a.data_dir}/test"
    t1 = read(f"{te}/test_source1.tsv", nrows=a.test_rows * 4)
    # Take France rows explicitly: they are the zero-shot country and the reason the
    # pipeline must never key on `country`.
    fr = t1[t1.country == "France"].head(a.test_rows // 3)
    other = t1[t1.country != "France"].head(a.test_rows - len(fr))
    write(pd.concat([fr, other], ignore_index=True), f"{a.out}/test/test_source1.tsv")
    for src in (2, 3):
        t = read(f"{te}/test_source{src}.tsv", nrows=a.test_pool * 3)
        frs = t[t.country == "France"].head(a.test_pool // 3)
        oth = t[t.country != "France"].head(a.test_pool - len(frs))
        write(pd.concat([frs, oth], ignore_index=True), f"{a.out}/test/test_source{src}.tsv")
        print(f"  test_source{src}: {a.test_pool:,} rows")
    print(f"mini dataset written to {a.out}")


if __name__ == "__main__":
    main()
