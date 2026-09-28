"""TSV loading/writing helpers. Always tab-separated, never quoted."""
import csv
import os

import pandas as pd


def read_source(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def read_ground_truth(path):
    """Return dict s1_id -> list of matched ids (empty list for singletons)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)
    return {a: (b.split(",") if b else []) for a, b in zip(df.iloc[:, 0], df.iloc[:, 1])}


def write_id_lists(path, s1_ids, lists, header):
    """lists: dict s1_id -> iterable of ids. One row per s1 id, in the given order."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for s in s1_ids:
            f.write(f"{s}\t{','.join(lists.get(s, ()))}\n")
