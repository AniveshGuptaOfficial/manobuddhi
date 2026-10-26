"""Eval runs: create, dispatch to RQ or inline, list, CSV export."""
import csv
import io
import subprocess

from fastapi import BackgroundTasks, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from engine.data import manifest

from . import jobs, settings
from .config import PROJECT_ROOT, config
from .db_models import EvalRun, Request
from .models import EvalCreate


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=3).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def benchmarks() -> list[dict]:
    try:
        m = manifest()
    except FileNotFoundError:
        return []
    return [{"name": k, "group": v["group"], "splits": {s: i["n"] for s, i in v["splits"].items()}}
            for k, v in m["benchmarks"].items()]


def create(session: Session, body: EvalCreate, background: BackgroundTasks) -> EvalRun:
    sizes = {b["name"]: b["splits"] for b in benchmarks()}
    if body.benchmark not in sizes:
        raise HTTPException(422, f"unknown benchmark '{body.benchmark}'; options: {sorted(sizes)}")
    if body.split not in sizes[body.benchmark]:
        raise HTTPException(422, f"'{body.benchmark}' has splits {sorted(sizes[body.benchmark])}")
    total = sizes[body.benchmark][body.split]
    run = EvalRun(benchmark=body.benchmark, split=body.split, n_items=min(body.n or total, total),
                  status="queued", progress=0.0, baseline=body.baseline,
                  router_config=settings.router_settings().to_dict(), git_commit=_git_commit())
    session.add(run)
    session.commit()
    dispatch(run.id, background)
    return run


def dispatch(run_id: int, background: BackgroundTasks) -> None:
    if config.eval_executor == "inline":
        background.add_task(jobs.run_eval, run_id)
        return
    from redis import Redis
    from rq import Queue

    Queue(config.eval_queue, connection=Redis.from_url(config.redis_url)).enqueue(
        "backend.jobs.run_eval", run_id, job_timeout=6 * 3600, result_ttl=86400)


def to_out(r: EvalRun) -> dict:
    return {"id": r.id, "benchmark": r.benchmark, "split": r.split, "n_items": r.n_items,
            "status": r.status, "progress": r.progress, "baseline": r.baseline,
            "router_config": r.router_config, "summary": r.summary,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "finished_at": r.finished_at.isoformat() if r.finished_at else None}


def results_csv(session: Session, run_id: int) -> str:
    cols = ["item_id", "route", "conflict_score", "probe_score", "disagreement", "logprob_confidence",
            "threshold", "reuse_method", "gold", "fast_correct", "correct", "lure_hit", "total_tokens",
            "total_latency_ms", "total_gpu_seconds"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in session.scalars(select(Request).where(Request.eval_run_id == run_id).order_by(Request.id)):
        w.writerow([getattr(r, c) for c in cols])
    return buf.getvalue()
