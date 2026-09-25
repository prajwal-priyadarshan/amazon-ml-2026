"""Language-agnostic normalisation of business names and addresses.

Everything works on a Latin-folded (unidecode) view so the same code runs for US,
India and France. Abbreviation tables are a small static seed that can be extended
with data-driven aliases from ``learn_aliases.py`` (loaded from the path in the
``ER_ALIASES`` environment variable).
"""
import json
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from unidecode import unidecode

LEGAL_CANON = {
    "inc": "inc", "incorporated": "inc", "lnc": "inc", "incorp": "inc",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "gmbh": "gmbh",
    "ltd": "ltd", "limited": "ltd", "lted": "ltd",
    "pvt": "pvt", "private": "pvt", "privaet": "pvt", "prvt": "pvt",
    "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sci": "sci",
    "sa": "sa", "snc": "snc", "scop": "scop",
}
NAME_STOP = {"and", "the", "of", "de", "la", "le", "les", "du", "des", "et"}

ADDR_CANON = {
    "street": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd",
    "bd": "blvd", "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy", "place": "pl", "terrace": "ter",
    "r": "rue", "square": "sq", "suite": "ste",
}
ADDR_DROP = {
    "unit", "ste", "apt", "flr", "floor", "room", "no", "number", "nr", "near",
    "opp", "opposite", "behind", "h", "hn", "hno", "door", "plot", "null", "nan",
    "none", "and",
}
ORDINAL_SUFFIX = {"nd", "rd", "th"}

_DOT3 = re.compile(r"\b([a-z])\.([a-z])\.([a-z])\.?")
_TLD = re.compile(r"\.(com|net|org|in|fr|io|co\.in|co\.uk)\b")
_STORE = re.compile(r"#\s*\d+")
_ALNUM = re.compile(r"[a-z0-9]+")
_SPLIT = re.compile(r"[a-z]+|\d+")
_LEET = str.maketrans("013457", "oleass")

_ALIASES = {}


def load_aliases(path):
    global _ALIASES
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            _ALIASES = json.load(f)
    return len(_ALIASES)


load_aliases(os.environ.get("ER_ALIASES"))


def _fold(s):
    return unidecode(unicodedata.normalize("NFKC", s)).lower()


def _has_native(s):
    return any(ord(c) > 0x24F for c in s)


def norm_name(raw):
    """Return (name_lat, name_core, name_key, legal, native)."""
    s = raw if isinstance(raw, str) else ""
    native = _has_native(s)
    s = _fold(s).replace("&", " and ")
    s = _DOT3.sub(r"\1\2\3", s)
    s = _TLD.sub(" ", s)
    s = _STORE.sub(" ", s)
    toks = []
    for t in _ALNUM.findall(s):
        if t.isdigit():
            if len(t) >= 7:  # phone number
                continue
            toks.append(t)
        elif t.isalpha():
            toks.append(t)
        else:
            letters = sum(c.isalpha() for c in t)
            digits = len(t) - letters
            if letters >= 3 and digits <= 2:  # leetspeak: transp0rt
                toks.append(t.translate(_LEET))
            else:
                toks.extend(_SPLIT.findall(t))
    if toks and toks[0] == "the":
        toks = toks[1:]
    legal = sorted({LEGAL_CANON[t] for t in toks if t in LEGAL_CANON})
    core = [t for t in toks if t not in LEGAL_CANON] or toks
    key = sorted(t for t in core if t not in NAME_STOP) or sorted(core)
    return " ".join(toks), " ".join(core), " ".join(key), " ".join(legal), native


def norm_address(raw):
    """Return (addr_norm, nums)."""
    s = raw if isinstance(raw, str) else ""
    toks = _SPLIT.findall(_fold(s))
    out = []
    for t in toks:
        if t in ORDINAL_SUFFIX and out and out[-1].isdigit():
            continue
        out.append(t)
    if _ALIASES:
        merged, i = [], 0
        while i < len(out):
            bi = " ".join(out[i:i + 2])
            if i + 1 < len(out) and bi in _ALIASES:
                merged.append(_ALIASES[bi])
                i += 2
            else:
                merged.append(_ALIASES.get(out[i], out[i]))
                i += 1
        out = merged
    out = [ADDR_CANON.get(t, t) for t in out]
    out = [t for t in out if t not in ADDR_DROP]
    nums = [t for t in out if t.isdigit()]
    return " ".join(out), " ".join(nums)


def _chunk(args):
    names, addrs = args
    return [norm_name(n) + norm_address(a) for n, a in zip(names, addrs)]


def normalize_frame(df, n_jobs=8, chunk=200_000):
    """Add normalised columns to a source frame. Row order is preserved."""
    names, addrs = df["business_name"].tolist(), df["business_address"].tolist()
    parts = [(names[i:i + chunk], addrs[i:i + chunk]) for i in range(0, len(df), chunk)]
    if n_jobs > 1 and len(parts) > 1:
        with ProcessPoolExecutor(n_jobs) as ex:
            res = [r for part in ex.map(_chunk, parts) for r in part]
    else:
        res = [r for p in parts for r in _chunk(p)]
    cols = ["name_lat", "name_core", "name_key", "legal", "native", "addr_norm", "nums"]
    out = pd.DataFrame(res, columns=cols, index=df.index)
    out["addr_missing"] = (out["addr_norm"] == "").astype("int8")
    out["native"] = out["native"].astype("int8")
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
