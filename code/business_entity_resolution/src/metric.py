"""Exact macro F0.5 as defined in the problem statement (singletons included)."""


def f05_single(pred, truth):
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred, truth, entities=None):
    """pred/truth: dict s1_id -> set of ids. Averaged over ``entities`` (default: truth keys)."""
    entities = list(truth) if entities is None else entities
    if not entities:
        return 0.0
    return sum(f05_single(set(pred.get(e, ())), set(truth.get(e, ()))) for e in entities) / len(entities)
