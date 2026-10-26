"""Metrics computed from the database (Savings Dashboard, eval summaries)."""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from engine.stats import auroc, percentile

from .db_models import PathTelemetry, Request

LATENCY_BINS = [0, 100, 200, 500, 1000, 2000, 5000]


def _filtered(stmt, source=None, eval_run_id=None):
    if source:
        stmt = stmt.where(Request.source == source)
    if eval_run_id is not None:
        stmt = stmt.where(Request.eval_run_id == eval_run_id)
    return stmt


def compute(session: Session, source: str | None = None, eval_run_id: int | None = None,
            window: int = 20000) -> dict:
    rows = session.execute(_filtered(
        select(Request.route, Request.total_latency_ms, Request.total_tokens, Request.total_gpu_seconds,
               Request.conflict_score, Request.logprob_confidence, Request.correct,
               Request.fast_correct, Request.lure_hit)
        .order_by(Request.id.desc()).limit(window), source, eval_run_id)).all()
    n = len(rows)
    if n == 0:
        return {"total_requests": 0}
    esc = sum(1 for r in rows if r.route == "escalated")
    lat = [r.total_latency_ms for r in rows]
    gpu_used = sum(r.total_gpu_seconds or 0 for r in rows)

    # Always-slow estimate: every request pays the mean measured slow-path cost (+ one fast pass
    # is NOT added: always-slow skips the fast model entirely).
    slow_mean = session.scalar(_filtered(
        select(func.avg(PathTelemetry.gpu_seconds)).join(Request).where(PathTelemetry.path == "slow"),
        source, eval_run_id))
    always_slow = float(slow_mean) * n if slow_mean is not None else None

    hist = [0] * (len(LATENCY_BINS) - 1)
    for v in lat:
        for i in range(len(hist)):
            if v is not None and LATENCY_BINS[i] <= v < LATENCY_BINS[i + 1]:
                hist[i] += 1
                break
        else:
            if v is not None and v >= LATENCY_BINS[-1]:
                hist[-1] += 1

    out = {
        "total_requests": n,
        "fast_requests": n - esc,
        "escalated_requests": esc,
        "escalation_rate": esc / n,
        "tokens_total": sum(r.total_tokens or 0 for r in rows),
        "tokens_per_request": sum(r.total_tokens or 0 for r in rows) / n,
        "gpu_seconds_used": gpu_used,
        "gpu_seconds_always_slow_est": always_slow,
        "gpu_seconds_saved_est": (always_slow - gpu_used) if always_slow is not None else None,
        "p50_latency_ms": percentile(lat, 50),
        "p95_latency_ms": percentile(lat, 95),
        "mean_conflict": sum(r.conflict_score or 0 for r in rows) / n,
        "latency_histogram": {"bins_ms": LATENCY_BINS, "counts": hist},
    }

    graded = [r for r in rows if r.correct is not None]
    if graded:
        fast_graded = [r for r in rows if r.fast_correct is not None]
        out["accuracy"] = sum(r.correct for r in graded) / len(graded)
        out["fast_only_accuracy"] = (sum(r.fast_correct for r in fast_graded) / len(fast_graded)
                                     if fast_graded else None)
        # O1: does the signal rank fast-path errors above fast-path successes?
        errs = [not r.fast_correct for r in fast_graded]
        out["auroc_conflict"] = auroc([r.conflict_score for r in fast_graded], errs)
        out["auroc_logprob_baseline"] = auroc([1 - (r.logprob_confidence or 0) for r in fast_graded], errs)
    lures = [r.lure_hit for r in rows if r.lure_hit is not None]
    if lures:
        out["fast_lure_rate"] = sum(lures) / len(lures)
    return out
