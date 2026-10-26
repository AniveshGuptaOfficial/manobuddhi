"""The F4 baselines, computed from paired fast/slow runs (engine/collect.py). Pure Python
+ numpy, so it runs anywhere (Kaggle CPU, your PC, tests).

Policies
  always_fast    fast model only (reference: lowest cost)
  always_slow    slow model only (baseline 1: the accuracy target)
  confidence     log-prob confidence cascade: escalate if fast confidence is low (baseline 2)
  probe_only     escalate if the hidden-state probe says the fast answer is likely wrong (baseline 3)
  oracle         escalate exactly the items the fast model gets wrong (upper bound, not a method)

Every policy's per-item cost is built from measured paths at batch 1:
  fast path  (+ probe forward pass, probe_only)  (+ slow path, if escalated)
always_slow pays only the slow path.

Thresholds are chosen on the calib split only: the lowest escalation rate whose calib
accuracy is within `tol` (default 1 point, the O2 target) of always-slow on calib. That
threshold is then applied unchanged to test. Sweeps over thresholds are saved for the
accuracy-vs-cost curves in the paper.
"""
from __future__ import annotations

import math

import numpy as np

from .stats import auroc, percentile

POLICIES = ("always_fast", "always_slow", "confidence", "probe_only", "oracle")
REFERENCE = {"always_fast", "oracle"}


# ---------------------------------------------------------------- signals
def risk(rec: dict, policy: str, probe: dict[str, float] | None = None,
         conf_key: str = "conf_seq") -> float | None:
    """Higher = more likely the fast answer is wrong = more reason to escalate."""
    if policy == "confidence":
        c = (rec.get("fast") or {}).get(conf_key)
        if c is None and conf_key != "conf_seq":
            c = (rec.get("fast") or {}).get("conf_seq")
        return None if c is None else 1.0 - c
    if policy == "probe_only":
        return (probe or {}).get(rec["id"])
    return None


def escalate(rec: dict, policy: str, threshold: float | None, probe=None,
             conf_key: str = "conf_seq") -> bool:
    if policy == "always_fast":
        return False
    if policy == "always_slow":
        return True
    if policy == "oracle":
        return rec.get("fast_correct") is False
    r = risk(rec, policy, probe, conf_key)
    return True if r is None else r >= threshold  # missing signal -> play safe


def _paths(rec: dict, policy: str, escalated: bool) -> list[tuple[str, dict]]:
    out = []
    if policy != "always_slow":
        out.append(("fast", rec["fast"]))
        if policy == "probe_only" and rec.get("probe"):
            out.append(("probe", rec["probe"]))
    if escalated:
        out.append(("slow", rec["slow"]))
    return out


def item_outcome(rec: dict, policy: str, escalated: bool) -> dict:
    """Correctness + measured cost of one item under one policy (sequential batch-1 serving)."""
    paths = _paths(rec, policy, escalated)
    final = "slow" if escalated else "fast"

    def tot(field):
        vals = [p.get(field) for _, p in paths]
        return None if any(v is None for v in vals) else sum(vals)

    lat = tot("latency_ms")
    pre = sum((p.get("latency_ms") or 0) for name, p in paths if name != final)
    fin = rec[final] or {}
    ttft = None if fin.get("ttft_ms") is None else pre + fin["ttft_ms"]
    return {
        "correct": rec.get(f"{final}_correct"),
        "lure": rec.get(f"{final}_lure"),
        "escalated": escalated,
        # probe forward pass re-reads tokens but generates none: counted in GPU-s, not tokens
        "tokens": sum((p.get("prompt_tokens") or 0) + (p.get("completion_tokens") or 0)
                      for name, p in paths if name != "probe"),
        "completion_tokens": sum(p.get("completion_tokens") or 0 for name, p in paths if name != "probe"),
        "gpu_seconds": tot("gpu_seconds"),
        "gpu_busy_seconds": tot("gpu_busy_seconds"),
        "energy_j": tot("energy_j"),
        "latency_ms": lat,
        "ttft_ms": ttft,
        "paths": [name for name, _ in paths],
    }


# ---------------------------------------------------------------- thresholds
def _acc(outcomes) -> float | None:
    g = [o["correct"] for o in outcomes if o["correct"] is not None]
    return sum(g) / len(g) if g else None


def choose_threshold(records: list[dict], policy: str, probe=None, tol: float = 0.01,
                     conf_key: str = "conf_seq") -> dict:
    """Lowest-escalation threshold on these (calib) records with accuracy >= always_slow - tol."""
    graded = [r for r in records if r.get("fast_correct") is not None and r.get("slow_correct") is not None]
    if not graded:
        raise ValueError("threshold selection needs graded calib records with fast and slow answers")
    target = sum(r["slow_correct"] for r in graded) / len(graded) - tol
    risks = [risk(r, policy, probe, conf_key) for r in graded]
    cands = sorted({x for x in risks if x is not None}) + [math.inf]
    best = None
    for t in cands:
        esc = [x is None or x >= t for x in risks]
        acc = sum((r["slow_correct"] if e else r["fast_correct"]) for r, e in zip(graded, esc)) / len(graded)
        rate = sum(esc) / len(esc)
        if acc >= target - 1e-12 and (best is None or rate < best["escalation_rate"]
                                       or (rate == best["escalation_rate"] and acc > best["accuracy"])):
            best = {"threshold": t, "accuracy": acc, "escalation_rate": rate}
    if best is None:  # cannot reach the target: escalate everything
        best = {"threshold": -math.inf, "accuracy": target + tol, "escalation_rate": 1.0}
    best.update({"target_accuracy": target, "tol": tol, "n_calib": len(graded),
                 "calib_always_slow_accuracy": target + tol})
    return best


# ---------------------------------------------------------------- summaries
def _boot(rng, n, fn, B=1000):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(outcomes: list[dict]) -> dict:
    n = len(outcomes)

    def mean(field):
        v = [o[field] for o in outcomes if o[field] is not None]
        return sum(v) / len(v) if v else None

    lat = [o["latency_ms"] for o in outcomes]
    lures = [o["lure"] for o in outcomes if o["lure"] is not None]
    return {
        "n": n,
        "n_graded": sum(1 for o in outcomes if o["correct"] is not None),
        "accuracy": _acc(outcomes),
        "escalation_rate": sum(o["escalated"] for o in outcomes) / n if n else None,
        "tokens_per_query": mean("tokens"),
        "completion_tokens_per_query": mean("completion_tokens"),
        "gpu_seconds_per_query": mean("gpu_seconds"),
        "gpu_busy_seconds_per_query": mean("gpu_busy_seconds"),
        "energy_j_per_query": mean("energy_j"),
        "latency_ms_mean": mean("latency_ms"),
        "latency_ms_p50": percentile(lat, 50),
        "latency_ms_p95": percentile(lat, 95),
        "ttft_ms_mean": mean("ttft_ms"),
        "lure_rate": sum(lures) / len(lures) if lures else None,
    }


def compare(outcomes: list[dict], ref: list[dict], seed: int = 42) -> dict:
    """Paired differences vs always_slow with 95% bootstrap CIs (same items resampled)."""
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    c = np.array([o["correct"] if o["correct"] is not None else np.nan for o in outcomes], dtype=float)
    cr = np.array([o["correct"] if o["correct"] is not None else np.nan for o in ref], dtype=float)

    def arr(lst, f):
        return np.array([o[f] if o[f] is not None else np.nan for o in lst], dtype=float)

    def saving(f):
        a, b = arr(outcomes, f), arr(ref, f)
        if np.isnan(a).all() or np.isnan(b).all():
            return None, None
        pt = 1 - np.nanmean(a) / np.nanmean(b)
        ci = _boot(rng, n, lambda i: float(1 - np.nanmean(a[i]) / np.nanmean(b[i])))
        return float(pt), ci

    out = {}
    if not np.isnan(c).all():
        out["accuracy_delta_vs_always_slow"] = float(np.nanmean(c) - np.nanmean(cr))
        out["accuracy_delta_ci95"] = _boot(rng, n, lambda i: float(np.nanmean(c[i]) - np.nanmean(cr[i])))
    for f, name in (("gpu_seconds", "gpu_seconds"), ("tokens", "tokens"), ("energy_j", "energy"),
                    ("completion_tokens", "completion_tokens")):
        pt, ci = saving(f)
        out[f"{name}_saving_vs_always_slow"] = pt
        out[f"{name}_saving_ci95"] = ci
    a, b = arr(outcomes, "latency_ms"), arr(ref, "latency_ms")
    if not np.isnan(a).all() and not np.isnan(b).all():
        out["p95_latency_ratio_vs_always_slow"] = float(np.nanpercentile(a, 95) / np.nanpercentile(b, 95))
    d, g, t = out.get("accuracy_delta_vs_always_slow"), out.get("gpu_seconds_saving_vs_always_slow"), \
        out.get("tokens_saving_vs_always_slow")
    out["meets_o2_target"] = (None if d is None or g is None or t is None
                              else bool(d >= -0.01 and g >= 0.30 and t >= 0.30))
    return out


def signal_aurocs(records: list[dict], probe: dict[str, float] | None = None) -> dict:
    """O1: how well does each signal rank fast-model errors above fast-model successes?"""
    graded = [r for r in records if r.get("fast_correct") is not None and r.get("fast")]
    wrong = [not r["fast_correct"] for r in graded]
    f = [r["fast"] for r in graded]
    out = {"n": len(graded), "fast_error_rate": (sum(wrong) / len(wrong)) if wrong else None,
           "logprob_seq": auroc([None if x.get("conf_seq") is None else 1 - x["conf_seq"] for x in f], wrong),
           "logprob_answer": auroc([None if x.get("conf_answer") is None else 1 - x["conf_answer"] for x in f], wrong),
           "first_token_entropy": auroc([x.get("first_token_entropy") for x in f], wrong)}
    if probe:
        out["probe"] = auroc([probe.get(r["id"]) for r in graded], wrong)
    lure = [r for r in graded if r.get("fast_lure") is not None]
    if lure:  # trap items: errors that are the intuitive lure answer (confident errors)
        out["fast_lure_rate"] = sum(r["fast_lure"] for r in lure) / len(lure)
    return out


def sweep(records: list[dict], policy: str, probe=None, conf_key="conf_seq", points: int = 21) -> list[dict]:
    rs = [risk(r, policy, probe, conf_key) for r in records]
    known = sorted(x for x in rs if x is not None)
    if not known:
        return []
    qs = sorted({known[min(len(known) - 1, int(q * (len(known) - 1)))] for q in np.linspace(0, 1, points)})
    out = []
    for t in qs + [math.inf]:
        oc = [item_outcome(r, policy, escalate(r, policy, t, probe, conf_key)) for r in records]
        s = summarize(oc)
        out.append({"threshold": t if t != math.inf else None, "accuracy": s["accuracy"],
                    "escalation_rate": s["escalation_rate"], "gpu_seconds_per_query": s["gpu_seconds_per_query"],
                    "tokens_per_query": s["tokens_per_query"], "latency_ms_p95": s["latency_ms_p95"]})
    return out


def run_baselines(calib: list[dict], test: list[dict], *, probe_calib=None, probe_test=None,
                  tol: float = 0.01, conf_key: str = "conf_seq", seed: int = 42) -> dict:
    """Thresholds from calib, everything reported on test."""
    need = [r for r in test if r.get("fast") and r.get("slow")]
    if len(need) < len(test):
        raise ValueError(f"{len(test) - len(need)} test records lack a fast or slow answer")
    policies = [p for p in POLICIES if p != "probe_only" or probe_test]
    thresholds = {"confidence": choose_threshold(calib, "confidence", tol=tol, conf_key=conf_key)}
    if probe_test:
        thresholds["probe_only"] = choose_threshold(calib, "probe_only", probe_calib, tol=tol)
    outcomes, out = {}, {"policies": {}, "thresholds": thresholds, "conf_key": conf_key, "tol": tol,
                         "n_test": len(test), "n_calib": len(calib)}
    for p in policies:
        t = thresholds.get(p, {}).get("threshold")
        pr = probe_test if p == "probe_only" else None
        outcomes[p] = [item_outcome(r, p, escalate(r, p, t, pr, conf_key)) for r in test]
        out.setdefault("escalated_ids", {})[p] = [r["id"] for r, o in zip(test, outcomes[p]) if o["escalated"]]
    for p in policies:
        s = summarize(outcomes[p])
        if p != "always_slow":
            s.update(compare(outcomes[p], outcomes["always_slow"], seed))
        s["reference_only"] = p in REFERENCE
        out["policies"][p] = s
    out["signal_auroc_test"] = signal_aurocs(test, probe_test)
    out["sweeps_test"] = {"confidence": sweep(test, "confidence", conf_key=conf_key)}
    if probe_test:
        out["sweeps_test"]["probe_only"] = sweep(test, "probe_only", probe_test)
    return out


def jsonable(x):
    """inf/nan -> None for JSON."""
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return None
    return x
