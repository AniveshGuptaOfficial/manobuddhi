"""F4 steps after collection, callable from the Kaggle notebook or scripts/f4.py.
None of these need a GPU."""
from __future__ import annotations

import json

import numpy as np

from . import baselines as B
from .probe import LinearProbe, train_probe
from .results import RunDir, key


def regrade(run: RunDir, log=print) -> dict:
    """Re-grade every saved answer with the current engine/grading.py (no GPU). Run it after a
    grader fix, then train-probe / score / baselines again: probe labels come from grading."""
    import os

    from .grading import grade, is_lure

    changed = {}
    for bench, split in run.collections():
        recs = run.load_records(bench, split)
        n = 0
        for r in recs:
            for p in ("fast", "slow"):
                if r.get(p):
                    new_c, new_l = grade(r["item"], r[p]["text"]), is_lure(r["item"], r[p]["text"])
                    n += (new_c != r.get(f"{p}_correct"))
                    r[f"{p}_correct"], r[f"{p}_lure"] = new_c, new_l
        path = run.records_path(bench, split)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        changed[key(bench, split)] = n
        log(f"regraded {key(bench, split)}: {n} verdicts changed")
    import hashlib
    from pathlib import Path

    g = Path(__file__).with_name("grading.py").read_bytes()
    run.update_manifest(grader_sha256=hashlib.sha256(g).hexdigest()[:16])
    return changed


def train(run: RunDir, benchmark: str = "gsm8k", split: str = "probe_train",
          candidates: list[str] | None = None, log=print) -> dict:
    recs = {r["id"]: r for r in run.load_records(benchmark, split)}
    ids, feats = run.load_features(benchmark, split)
    keep = [i for i, x in enumerate(ids) if recs[x]["fast_correct"] is not None]
    if not keep:
        raise ValueError(f"no graded records with features in {key(benchmark, split)}")
    wrong = np.array([not recs[ids[i]]["fast_correct"] for i in keep], dtype=int)
    feats = {k: v[keep] for k, v in feats.items()}
    log(f"training probe on {key(benchmark, split)}: n={len(keep)}, fast wrong={wrong.mean():.1%}")
    probe, report = train_probe(feats, wrong, candidates=candidates, log=log)
    d = run.path / "probe"
    d.mkdir(parents=True, exist_ok=True)
    probe.save(d / "probe.npz")
    report["trained_on"] = key(benchmark, split)
    (d / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    run.update_manifest(probe={"feature": probe.feature, "cv_auroc": report["selected"]["cv_auroc"],
                               "C": report["selected"]["C"], "trained_on": key(benchmark, split),
                               "n": report["n"]})
    log(f"selected {probe.feature} (C={report['selected']['C']}), CV AUROC {report['selected']['cv_auroc']:.3f}")
    return report


def score(run: RunDir, log=print) -> dict:
    """Apply the trained probe to every collection that has features (except its training set)."""
    probe = LinearProbe.load(run.path / "probe" / "probe.npz")
    trained_on = run.manifest().get("probe", {}).get("trained_on")
    done = {}
    for bench, split in run.collections():
        if key(bench, split) == trained_on:
            continue
        ids, feats = run.load_features(bench, split, names=[probe.feature])
        if not ids:
            continue
        p = probe.predict(feats[probe.feature])
        run.save_scores(bench, split, dict(zip(ids, map(float, p))))
        done[key(bench, split)] = len(ids)
        log(f"scored {key(bench, split)}: {len(ids)} items")
    return done


def baselines(run: RunDir, benchmark: str = "gsm8k", calib: str = "calib", test: str = "test",
              tol: float = 0.01, conf_key: str = "conf_seq", log=print) -> dict:
    c, t = run.load_records(benchmark, calib), run.load_records(benchmark, test)
    if not c or not t:
        raise ValueError(f"need records for {key(benchmark, calib)} and {key(benchmark, test)}")
    pc, pt = run.load_scores(benchmark, calib) or None, run.load_scores(benchmark, test) or None
    res = B.run_baselines(c, t, probe_calib=pc, probe_test=pt, tol=tol, conf_key=conf_key)
    res.update({"benchmark": benchmark, "calib_split": calib, "test_split": test})
    data = run.load_baselines()
    data.setdefault("benchmarks", {})[benchmark] = B.jsonable(res)
    # O1 on other collections (e.g. trap sets): signal AUROCs only, no routing
    data["signal_auroc"] = {}
    for bench, split in run.collections():
        recs = run.load_records(bench, split)
        if recs and recs[0].get("fast"):
            data["signal_auroc"][key(bench, split)] = B.jsonable(
                B.signal_aurocs(recs, run.load_scores(bench, split) or None))
    run.save_baselines(data)
    log(report(run, benchmark))
    return res


def _f(x, d=3, pct=False):
    if x is None:
        return "-"
    return f"{100 * x:.1f}%" if pct else f"{x:.{d}f}"


def report(run: RunDir, benchmark: str = "gsm8k") -> str:
    data = run.load_baselines()
    b = data.get("benchmarks", {}).get(benchmark)
    if not b:
        return "no baselines yet"
    m = run.manifest().get("models", {})
    lines = [f"{benchmark} {b['test_split']} (n={b['n_test']}), thresholds from {b['calib_split']} "
             f"(n={b['n_calib']}, tol {b['tol']:.0%})",
             f"fast={m.get('fast', {}).get('name')}  slow={(m.get('slow') or {}).get('name')}", "",
             f"{'policy':<13}{'acc':>7}{'Δacc':>8}{'esc':>7}{'GPU-s/q':>9}{'save':>8}{'tok/q':>8}"
             f"{'p95 ms':>9}{'J/q':>8}"]
    for p, s in b["policies"].items():
        name = p + (" *" if s.get("reference_only") else "")
        lines.append(f"{name:<13}{_f(s['accuracy'], pct=True):>7}"
                     f"{_f(s.get('accuracy_delta_vs_always_slow'), pct=True):>8}"
                     f"{_f(s['escalation_rate'], pct=True):>7}{_f(s['gpu_seconds_per_query'], 2):>9}"
                     f"{_f(s.get('gpu_seconds_saving_vs_always_slow'), pct=True):>8}"
                     f"{_f(s['tokens_per_query'], 0):>8}{_f(s['latency_ms_p95'], 0):>9}"
                     f"{_f(s['energy_j_per_query'], 0):>8}")
    lines.append("* reference only (not a baseline)")
    a = b.get("signal_auroc_test", {})
    lines += ["", "AUROC for fast-model errors (test): " + ", ".join(
        f"{k} {_f(a.get(k))}" for k in ("logprob_seq", "logprob_answer", "first_token_entropy", "probe"))]
    for k, v in data.get("signal_auroc", {}).items():
        if k.startswith("traps"):
            lines.append(f"  {k}: probe {_f(v.get('probe'))}, logprob {_f(v.get('logprob_seq'))}, "
                         f"fast lure rate {_f(v.get('fast_lure_rate'), pct=True)}")
    return "\n".join(lines)
