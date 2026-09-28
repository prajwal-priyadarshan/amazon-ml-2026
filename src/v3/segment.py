"""Glued-name segmentation: `akshayagases.com` -> `akshaya gases`.

A measured failure class, not a hypothetical one. Real true pairs in the training data
include `Akshaya Gases Limited` ~ `akshayagases.com` and `wilfordhancock.com`, where the
name channel is a domain with the spaces removed. Every character n-gram view sees
`akshayagases` and the reference sees `akshaya gases`; the 3-grams do overlap, but the word
view, the exact key and the phonetic skeleton all miss, and those are the views that decide
the rare-token matches.

The vocabulary is mined from the Source-1 names in the provided files only -- no external
word list, no gazetteer, which the rules require. Source 1 is the clean, deduplicated side,
so its token frequencies are the best in-domain dictionary available.

Segmentation is deliberately conservative: only tokens long enough to be suspicious, only
splits where *every* part is a known word of at least three characters, at most four parts,
and the best split has to beat leaving the token alone. A wrong split is worse than no
split, because it adds tokens that will match the wrong entity.
"""
import json
import math
import os
import re

from .paths import log

_TOK = re.compile(r"[a-z]+")
MIN_GLUED = 10   # shorter tokens are usually real words
MIN_PART = 3
MAX_PARTS = 4


def build_vocab(name_iters, min_count=25):
    """Token -> count, from raw Source-1 business names (lowercased letter runs)."""
    counts = {}
    for names in name_iters:
        for raw in names:
            if not raw:
                continue
            for t in _TOK.findall(str(raw).lower()):
                if len(t) >= MIN_PART:
                    counts[t] = counts.get(t, 0) + 1
    return {t: c for t, c in counts.items() if c >= min_count}


def save_vocab(vocab, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    log(f"  vocabulary: {len(vocab):,} tokens -> {os.path.basename(path)}")


def load_vocab(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Segmenter:
    def __init__(self, vocab):
        total = sum(vocab.values()) or 1
        # Longer words are penalised less: without the length term the search prefers many
        # short common words ("a ksha ya") over the one right split.
        self.cost = {t: -math.log(c / total) - 0.9 * len(t) for t, c in vocab.items()}
        self.maxlen = max((len(t) for t in vocab), default=0)

    def split(self, tok):
        """Best word split of ``tok``, or None when nothing beats leaving it alone."""
        n = len(tok)
        if n < MIN_GLUED or tok in self.cost:
            return None
        best = [None] * (n + 1)
        best[0] = (0.0, 0, None)  # (cost, parts, start of the last word)
        for i in range(1, n + 1):
            for j in range(max(0, i - self.maxlen), i - MIN_PART + 1):
                if best[j] is None:
                    continue
                c = self.cost.get(tok[j:i])
                if c is None:
                    continue
                parts = best[j][1] + 1
                if parts > MAX_PARTS:
                    continue
                cand = (best[j][0] + c, parts, j)
                if best[i] is None or cand[0] < best[i][0]:
                    best[i] = cand
        if best[n] is None or best[n][1] < 2:
            return None
        out, i = [], n
        while i > 0:
            j = best[i][2]
            out.append(tok[j:i])
            i = j
        return " ".join(reversed(out))

    def text(self, s):
        """Segment every glued token in a normalised name; unchanged if nothing splits."""
        if not s:
            return s
        parts, hit = [], False
        for t in s.split():
            sp = self.split(t) if t.isalpha() else None
            if sp:
                parts.append(sp)
                hit = True
            else:
                parts.append(t)
        return " ".join(parts) if hit else s
