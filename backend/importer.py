"""Load a GPU results folder (results/<run>/) into the app database, so the Benchmark Lab
and Savings Dashboard show real, measured numbers (simulated: false).

One eval_run per (benchmark, policy) in baselines.json; one request per test item with its
measured paths (fast, probe, slow). Routing decisions come from baselines.json
(escalated_ids), so the app shows exactly what the paper reports. Importing the same run
twice is skipped unless replace=True.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from engine import baselines as B
from engine.results import RunDir

from . import metrics
from .db_models import EvalRun, PathTelemetry, Request

BACKEND = "hf-transformers"


def _path_row(name: str, p: dict) -> PathTelemetry:
    return PathTelemetry(
        path=name, model=p.get("model") or "probe", prompt_tokens=p.get("prompt_tokens"),
        completion_tokens=p.get("completion_tokens"), reused_tokens=0, ttft_ms=p.get("ttft_ms"),
        latency_ms=p.get("latency_ms"), gpu_seconds=p.get("gpu_seconds"),
        gpu_util_pct=p.get("gpu_util_pct"), gpu_mem_mb=p.get("gpu_mem_mb"),
        energy_j=p.get("energy_j"), queue_depth=0)


def _request(rec: dict, policy: str, escalated: bool, threshold, probe_p, mode, models, run_id) -> Request:
    o = B.item_outcome(rec, policy, escalated)
    fast, slow = rec.get("fast") or {}, rec.get("slow") or {}
    ran_fast = policy != "always_slow"
    conf = fast.get("conf_seq") if ran_fast else None
    signal = {"confidence": None if conf is None else 1 - conf,
              "probe_only": probe_p}.get(policy)
    req = Request(
        source="eval", eval_run_id=run_id, item_id=rec["id"], query=rec["prompt"], mode=mode,
        fast_model=models["fast"], slow_model=models["slow"], route="escalated" if escalated else "fast",
        conflict_score=signal, probe_score=probe_p if policy == "probe_only" else None,
        disagreement=None, k_samples=None, logprob_confidence=conf, threshold=threshold,
        threshold_mode="calibrated" if policy in ("confidence", "probe_only") else policy,
        latency_budget_ms=None, reuse_method="none",
        fast_answer=fast.get("text") if ran_fast else None,
        final_answer=(slow if escalated else fast).get("text"),
        gold=str(rec["item"]["gold"]), correct=o["correct"],
        fast_correct=rec.get("fast_correct") if ran_fast else None,
        lure_hit=rec.get("fast_lure") if ran_fast else None,
        total_tokens=o["tokens"], total_latency_ms=o["latency_ms"], total_gpu_seconds=o["gpu_seconds"])
    paths = []
    if ran_fast:
        paths.append(_path_row("fast", fast))
        if policy == "probe_only" and rec.get("probe"):
            paths.append(_path_row("probe", {**rec["probe"], "model": models["fast"]}))
    if escalated:
        paths.append(_path_row("slow", slow))
    req.paths = paths
    return req


def existing(session: Session, run_name: str) -> list[EvalRun]:
    return [r for r in session.scalars(select(EvalRun))
            if (r.router_config or {}).get("results_run") == run_name]


def import_run(session: Session, path, *, replace: bool = False, log=print) -> list[EvalRun]:
    run = RunDir(path)
    man, data = run.manifest(), run.load_baselines()
    if not man or not data.get("benchmarks"):
        raise ValueError(f"{run.path} has no manifest.json or baselines.json; run `f4.py baselines` first")
    name = man["run"]
    old = existing(session, name)
    if old and not replace:
        log(f"run '{name}' already imported ({len(old)} eval runs); use --replace to re-import")
        return old
    for r in old:
        session.delete(r)
    session.flush()

    m = man["models"]
    mode = m["mode"]
    models = {"fast": m["fast"]["name"], "slow": (m.get("slow") or m["fast"])["name"]}
    if mode == "single_model_nothink_think":
        models = {"fast": models["fast"] + " (no-think)", "slow": models["slow"] + " (think)"}
    created = []
    for bench, b in data["benchmarks"].items():
        test = run.load_records(bench, b["test_split"])
        probe = run.load_scores(bench, b["test_split"])
        for policy, summ in b["policies"].items():
            esc = set(b["escalated_ids"][policy])
            thr = (b["thresholds"].get(policy) or {}).get("threshold")
            cfg = {"backend": BACKEND, "results_run": name, "policy": policy, "mode": mode,
                   "fast_model": models["fast"], "slow_model": models["slow"],
                   "threshold_mode": "calibrated" if policy in b["thresholds"] else policy,
                   "threshold": thr, "calib_split": b["calib_split"], "tol": b["tol"],
                   "conf_key": b["conf_key"], "probe": man.get("probe"),
                   "generation": man.get("generation"), "models_detail": m,
                   "hardware": (man.get("environment") or {}).get("hardware")}
            er = EvalRun(benchmark=bench, split=b["test_split"], n_items=len(test), status="running",
                         progress=1.0, router_config=cfg, baseline=policy, git_commit=man.get("git_commit"))
            session.add(er)
            session.flush()
            for rec in test:
                session.add(_request(rec, policy, rec["id"] in esc, thr, probe.get(rec["id"]),
                                     mode, models, er.id))
            session.flush()
            s = metrics.compute(session, eval_run_id=er.id)
            s.update({"router_backend": BACKEND, "simulated": False, "imported_from": name,
                      "reference_only": summ.get("reference_only", False), "f4": summ,
                      "signal_auroc_test": b.get("signal_auroc_test")})
            er.summary, er.status, er.finished_at = s, "done", datetime.now(timezone.utc)
            created.append(er)
            log(f"imported {bench}/{b['test_split']} {policy}: {len(test)} items, "
                f"acc {s.get('accuracy', 0):.3f}, escalation {s['escalation_rate']:.2f}")
    session.commit()
    return created
