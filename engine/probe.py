"""Linear probe on the fast model's hidden states: P(fast answer is wrong) (O1).

train_probe() picks the feature (layer x position) and regularisation by 5-fold
cross-validated AUROC on probe_train only, then refits on all of probe_train. calib,
dev and test are never seen. The probe is saved as plain numpy arrays, so applying it
needs only numpy (no scikit-learn in the API or F5 router).

Feature names: "<pos>_L<layer>" with pos = last (final answer token, after the answer
exists: the method), mean (mean over answer tokens), prompt (last prompt token, before
answering: the ablation that mirrors prior pre-generation probes).
"""
from __future__ import annotations

import json

import numpy as np

from .stats import auroc


class LinearProbe:
    def __init__(self, feature: str, mean, scale, coef, intercept: float, meta: dict | None = None):
        self.feature = feature
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = float(intercept)
        self.meta = meta or {}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: (n, d) features for self.feature -> P(wrong) per row."""
        z = ((np.asarray(x, dtype=np.float32) - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path) -> None:
        np.savez(path, feature=np.array(self.feature), mean=self.mean, scale=self.scale,
                 coef=self.coef, intercept=np.array(self.intercept),
                 meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path) -> "LinearProbe":
        z = np.load(path)
        return cls(str(z["feature"]), z["mean"], z["scale"], z["coef"], float(z["intercept"]),
                   json.loads(str(z["meta"])))


def _folds(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """Stratified fold assignment without scikit-learn's splitter (stable across versions)."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return [np.flatnonzero(fold == i) for i in range(k)]


def train_probe(features: dict[str, np.ndarray], wrong: np.ndarray, *, k: int = 5,
                cs: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0), seed: int = 42,
                candidates: list[str] | None = None, log=print) -> tuple[LinearProbe, dict]:
    """features: name -> (n, d); wrong: (n,) 1 if the fast answer was wrong."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(wrong, dtype=int)
    if y.min() == y.max():
        raise ValueError("probe needs both correct and wrong fast answers in probe_train")
    folds = _folds(y, k, seed)
    names = candidates or sorted(features)
    table = []
    for name in names:
        X = features[name].astype(np.float32)
        for c in cs:
            oof = np.zeros(len(y))
            for f in folds:
                tr = np.setdiff1d(np.arange(len(y)), f)
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                clf = LogisticRegression(C=c, max_iter=3000)
                clf.fit((X[tr] - mu) / sd, y[tr])
                oof[f] = clf.predict_proba((X[f] - mu) / sd)[:, 1]
            a = auroc(oof.tolist(), y.tolist())
            table.append({"feature": name, "C": c, "cv_auroc": a})
        best_c = max((t for t in table if t["feature"] == name), key=lambda t: t["cv_auroc"] or 0)
        log(f"  {name:>12}: CV AUROC {best_c['cv_auroc']:.3f} (C={best_c['C']})")
    best = max(table, key=lambda t: t["cv_auroc"] or 0)
    X = features[best["feature"]].astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=best["C"], max_iter=3000).fit((X - mu) / sd, y)
    report = {"n": int(len(y)), "n_wrong": int(y.sum()), "folds": k, "seed": seed,
              "selected": best, "table": table}
    probe = LinearProbe(best["feature"], mu, sd, clf.coef_[0], clf.intercept_[0],
                        {"C": best["C"], "cv_auroc": best["cv_auroc"], "n_train": int(len(y))})
    return probe, report
