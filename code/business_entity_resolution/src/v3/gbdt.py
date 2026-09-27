"""The two tabular models: the L2 pruner and the L3 stacker, plus calibration.

LightGBM stays the tabular model here, and the reason is specific rather than general.
Tabular foundation models now lead single-model benchmarks, but the two that matter are
licence (TabPFN-2 ships under a modified Apache with attribution; 2.5 and later are
non-commercial, and the rules demand plain MIT or Apache-2.0) and scale (in-context
learners degrade past ~10k training rows, and this problem has hundreds of millions of
candidate pairs at inference). LightGBM scores those in minutes on CPU.

``max_bin=63`` is not a tuning knob: LightGBM bins features to uint8, so 10M rows x 60
features costs about 540 MB once the raw float32 frame is released. At the default 255 bins
the same fit does not fit alongside the rest of the pipeline.

Calibration is per country and isotonic. The decision layer maximises expected F0.5 over
*absolute* probabilities, so a model that ranks well but is miscalibrated picks the wrong
list length -- and US and India have measurably different score distributions, which a
single global calibrator averages into being wrong for both.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .paths import log


def make_model(n_estimators=1500, lr=0.05, leaves=127, min_child=50, seed=None):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=lr, num_leaves=leaves,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            min_child_samples=min_child, max_bin=63, n_jobs=-1, verbose=-1,
            random_state=seed), True
    except Exception:  # missing libomp and the like: still runnable, just slower and weaker
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.08,
                                              max_leaf_nodes=63, max_bins=63), False


def fit(X, y, Xv, yv, name="model", sample_weight=None, **kw):
    # kw may carry seed= for bagging; make_model takes it
    model, is_lgb = make_model(**kw)
    if is_lgb:
        import lightgbm as lgb
        stop = [lgb.early_stopping(60, verbose=False)]
        try:  # lightgbm >= 4.7 renamed eval_set to eval_X/eval_y
            model.fit(X, y, sample_weight=sample_weight, eval_X=Xv, eval_y=yv, callbacks=stop)
        except TypeError:
            model.fit(X, y, sample_weight=sample_weight, eval_set=[(Xv, yv)], callbacks=stop)
        best = getattr(model, "best_iteration_", None)
        log(f"  {name}: fitted on {len(y):,} rows, best_iteration={best}")
    else:
        model.fit(X, y)
        log(f"  {name}: fitted on {len(y):,} rows (sklearn fallback)")
    return model


def predict(model, X, chunk=2_000_000):
    out = np.empty(len(X), np.float32)
    for s in range(0, len(X), chunk):
        e = min(s + chunk, len(X))
        out[s:e] = model.predict_proba(X[s:e])[:, 1].astype(np.float32)
    return out


def importances(model, cols, n=15):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return ""
    pairs = sorted(zip(cols, imp), key=lambda t: -t[1])[:n]
    return ", ".join(f"{c}:{int(v)}" for c, v in pairs)


class Calibrator:
    """Per-country isotonic regression with a global fallback for unseen countries.

    France never appears in train, so it necessarily falls back to the global fit. That is
    the honest behaviour: there is no France calibration data, and pretending otherwise by
    reusing (say) the US curve would be an unmeasured guess.
    """

    def __init__(self):
        self.by_country = {}
        self.global_ = None

    def fit(self, p, y, country):
        self.global_ = IsotonicRegression(out_of_bounds="clip").fit(p, y)
        for c in np.unique(country):
            m = country == c
            if m.sum() >= 5000 and 0 < y[m].sum() < m.sum():
                self.by_country[str(c)] = IsotonicRegression(out_of_bounds="clip").fit(p[m], y[m])
                log(f"  calibrated [{c}] on {int(m.sum()):,} rows")
        return self

    def apply(self, p, country):
        iso = self.by_country.get(str(country), self.global_)
        return iso.predict(p).astype(np.float32)
