"""Work-directory layout, logging and the resume primitive shared by every v3 stage.

Every stage is a pure function from files on disk to files on disk, keyed by
(split, country, source). That is what makes the pipeline resumable: a stage skips any
shard whose output already exists, so a run that dies at hour nine restarts at the shard
it died on. It is also what keeps peak RSS flat -- no stage ever holds more than one
(country, source) block, regardless of corpus size.

Splits and pools are separate axes:
  split  = which S1 rows we are matching   -- ``fit`` / ``holdout`` / ``test``
  pool   = which S2/S3 records they match against -- ``train`` for fit+holdout, ``test`` for test
``fit`` and ``holdout`` are disjoint S1 samples of train_source1 that share one pool, so a
hold-out score is measured against the same full negative universe as the real test set.
"""
import json
import os
import time

SPLITS = ("fit", "holdout", "test")
SOURCES = (2, 3)


def pool_of(split):
    return "test" if split == "test" else "train"


class Work:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        for sub in ("prep", "emb", "cand", "pruned", "ce", "scored", "models", "logs"):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

    def p(self, *parts):
        return os.path.join(self.root, *parts)

    # --- prep -------------------------------------------------------------
    def s1(self, split, country):
        return self.p("prep", f"s1_{split}_{_slug(country)}.parquet")

    def d(self, pool, country, src):
        return self.p("prep", f"d_{pool}_{_slug(country)}_{src}.parquet")

    def countries(self, split):
        pre = f"s1_{split}_"
        out = []
        for f in sorted(os.listdir(self.p("prep"))):
            if f.startswith(pre) and f.endswith(".parquet"):
                out.append(f[len(pre):-len(".parquet")])
        return out

    # --- embeddings -------------------------------------------------------
    def emb(self, kind, key):
        return self.p("emb", f"{kind}_{_slug(key)}.f16")

    # --- candidates and scores -------------------------------------------
    def cand(self, view, split, country, src):
        return self.p("cand", f"{view}_{split}_{_slug(country)}_{src}.parquet")

    def pruned(self, split, country, src):
        return self.p("pruned", f"{split}_{_slug(country)}_{src}.parquet")

    def ce(self, split, country, src):
        return self.p("ce", f"{split}_{_slug(country)}_{src}.parquet")

    def scored(self, split, country, src):
        return self.p("scored", f"{split}_{_slug(country)}_{src}.parquet")

    def model(self, name):
        return self.p("models", name)

    # --- small json state -------------------------------------------------
    def read_json(self, *parts):
        path = self.p(*parts)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def write_json(self, obj, *parts):
        path = self.p(*parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))


_T0 = time.time()


def log(msg):
    m, s = divmod(int(time.time() - _T0), 60)
    print(f"[{time.strftime('%H:%M:%S')} +{m:02d}:{s:02d}] {msg}", flush=True)


def done(path, force=False):
    """True when a stage output already exists and may be skipped."""
    if force:
        return False
    ok = os.path.exists(path) and os.path.getsize(path) > 0
    if ok:
        log(f"  skip (exists): {os.path.basename(path)}")
    return ok
