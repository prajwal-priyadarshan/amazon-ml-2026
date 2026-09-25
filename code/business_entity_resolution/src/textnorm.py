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

# Long form -> canonical short form, so "International"/"Intl" and "Services"/"Svcs" agree.
# Deliberately country-agnostic: no state names, no postal conventions.
NAME_CANON = {
    "international": "intl", "intl": "intl", "natl": "natl", "national": "natl",
    "manufacturing": "mfg", "manufacturers": "mfg", "manufacturer": "mfg", "mfg": "mfg",
    "services": "svc", "service": "svc", "svcs": "svc", "svc": "svc",
    "associates": "assoc", "associate": "assoc", "assoc": "assoc", "assocs": "assoc",
    "brothers": "bros", "bros": "bros", "management": "mgmt", "mgmt": "mgmt",
    "technologies": "tech", "technology": "tech", "tech": "tech", "technical": "tech",
    "industries": "ind", "industrial": "ind", "industry": "ind",
    "enterprises": "ent", "enterprise": "ent", "entreprise": "ent", "entreprises": "ent",
    "solutions": "soln", "solution": "soln", "systems": "sys", "system": "sys",
    "engineering": "engg", "engineers": "engg", "engineer": "engg",
    "construction": "constr", "constructions": "constr",
    "development": "dev", "developers": "dev", "developer": "dev",
    "trading": "trdg", "traders": "trdg", "trader": "trdg",
    "distributors": "distr", "distributor": "distr", "distribution": "distr",
    "marketing": "mktg", "agency": "agcy", "agencies": "agcy",
    "holdings": "hldg", "holding": "hldg", "laboratories": "lab", "laboratory": "lab",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma",
    "products": "prod", "product": "prod", "supplies": "sply", "supply": "sply",
    "equipments": "equip", "equipment": "equip", "machinery": "mach",
    "transport": "trans", "transportation": "trans", "logistics": "logi",
    "consultancy": "cons", "consultants": "cons", "consulting": "cons", "consultant": "cons",
    "societe": "ste2", "restaurant": "rest", "hospital": "hosp",
}

ADDR_CANON = {
    "street": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd",
    "bd": "blvd", "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy", "place": "pl", "terrace": "ter",
    "r": "rue", "square": "sq", "suite": "ste",
    # French street vocabulary (France is zero-shot: it only appears in test).
    "chemin": "ch", "route": "rte", "allee": "all", "allees": "all",
    "impasse": "imp", "quai": "qai", "cours": "crs", "faubourg": "fbg",
    "residence": "res", "batiment": "bat", "zone": "zi", "industrielle": "zi",
}
ADDR_DROP = {
    "unit", "ste", "apt", "flr", "floor", "room", "no", "number", "nr", "near",
    "opp", "opposite", "behind", "h", "hn", "hno", "door", "plot", "null", "nan",
    "none", "and",
    # French articles/prepositions carry no identifying signal ("rue de la Paix").
    "de", "du", "des", "la", "le", "les", "et", "aux",
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


_VOWELS = str.maketrans("", "", "aeiou")
# Aspirated/compound digraphs collapse to their base sound. Romanised Indic text is the
# reason this matters: unidecode renders महाराष्ट्र as "mhaaraassttr", which shares almost
# no character n-grams with "maharashtra" until both are folded this way.
_DIGRAPH = (("sh", "s"), ("ch", "c"), ("th", "t"), ("ph", "f"), ("bh", "b"), ("dh", "d"),
            ("gh", "g"), ("kh", "k"), ("jh", "j"), ("zh", "z"), ("ck", "k"), ("qu", "k"),
            ("ee", "i"), ("oo", "u"), ("aa", "a"), ("ii", "i"))


def _phon(t):
    """maharashtra -> mhrstr  and  mhaaraassttr -> mhrstr"""
    for a, b in _DIGRAPH:
        t = t.replace(a, b)
    s = t[:1] + t[1:].translate(_VOWELS)
    out = []
    for ch in s:
        if not out or out[-1] != ch:  # collapse doubled consonants
            out.append(ch)
    return "".join(out)


def _skeleton(tokens):
    """Phonetic consonant skeleton: survives transliteration wobble (Rajender/Rajinder,
    Devanagari round-trips) without pulling in a new dependency."""
    out = []
    for t in tokens:
        if t.isdigit():
            out.append(t)
            continue
        s = _phon(t)
        if len(s) >= 2:
            out.append(s)
    return " ".join(sorted(out))


def norm_name(raw):
    """Return (name_lat, name_core, name_key, name_skel, legal, native)."""
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
    core = [NAME_CANON.get(t, t) for t in toks if t not in LEGAL_CANON] or toks
    key = sorted(t for t in core if t not in NAME_STOP) or sorted(core)
    return " ".join(toks), " ".join(core), " ".join(key), _skeleton(key), " ".join(legal), native


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
    cols = ["name_lat", "name_core", "name_key", "name_skel", "legal", "native", "addr_norm", "nums"]
    out = pd.DataFrame(res, columns=cols, index=df.index)
    out["addr_missing"] = (out["addr_norm"] == "").astype("int8")
    out["native"] = out["native"].astype("int8")
    return pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)
