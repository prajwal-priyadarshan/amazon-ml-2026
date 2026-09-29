"""Learn address alias pairs (NY<->new york, st<->street, mh<->maharashtra, ...) from train ground truth.

Aligns matched S1/S2-S3 addresses, looks at the 1-2 token leftovers on each side and keeps
alphabetic pairs that co-occur far more often than chance. Uses only the provided data.

Usage: python -m src.learn_aliases --data-dir student_resource/dataset --out work/aliases.json
"""
import argparse
import json
from collections import Counter

import pandas as pd

from .data import read_ground_truth, read_source
from .textnorm import norm_address


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300_000)
    ap.add_argument("--min-count", type=int, default=50)
    ap.add_argument("--min-purity", type=float, default=0.4)
    a = ap.parse_args()

    tr = f"{a.data_dir}/train"
    s1 = read_source(f"{tr}/train_source1.tsv").sample(a.n, random_state=0)
    gt = read_ground_truth(f"{tr}/train_ground_truth.tsv")
    first = {s: gt[s][0] for s in s1.entity_id if gt.get(s)}
    need = set(first.values())
    d = pd.concat([read_source(f"{tr}/train_source2.tsv"), read_source(f"{tr}/train_source3.tsv")])
    d = d[d.entity_id.isin(need)].set_index("entity_id")["business_address"]
    pairs, freq, side_a = Counter(), Counter(), Counter()
    for sid, addr in zip(s1.entity_id, s1.business_address):
        m = first.get(sid)
        if m is None or m not in d.index or not isinstance(d[m], str):
            continue
        ta, tb = norm_address(addr)[0].split(), norm_address(d[m])[0].split()
        la = [t for t in ta if t not in tb]
        lb = [t for t in tb if t not in ta]
        if 1 <= len(la) <= 2 and 1 <= len(lb) <= 2:
            ka, kb = " ".join(la), " ".join(lb)
            if ka.replace(" ", "").isalpha() and kb.replace(" ", "").isalpha():
                pairs[(ka, kb)] += 1
                side_a[ka] += 1
                freq[ka] += 1
                freq[kb] += 1
    aliases = {}
    for (ka, kb), n in pairs.most_common():
        if n < a.min_count or n / side_a[ka] < a.min_purity:
            continue
        src, dst = (ka, kb) if freq[ka] <= freq[kb] else (kb, ka)
        if src not in aliases and dst not in aliases:
            aliases[src] = dst
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(aliases, f, ensure_ascii=False, indent=1)
    print(f"learned {len(aliases)} aliases -> {a.out}")
    for k, v in list(aliases.items())[:30]:
        print(f"  {k!r} -> {v!r}")


if __name__ == "__main__":
    main()
