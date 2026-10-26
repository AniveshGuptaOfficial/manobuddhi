"""Write one RouteResult (+ its per-path telemetry) to the database."""
from sqlalchemy.orm import Session

from engine.grading import extract_final, grade, is_lure
from engine.interface import RouteResult, RouterSettings

from .db_models import PathTelemetry, Request


def save_result(session: Session, res: RouteResult, *, query: str, settings: RouterSettings,
                source: str, eval_run_id: int | None = None, item: dict | None = None,
                commit: bool = True) -> Request:
    req = Request(
        source=source, eval_run_id=eval_run_id, item_id=item["id"] if item else None, query=query,
        mode=settings.mode, fast_model=settings.fast_model, slow_model=settings.slow_model,
        route=res.route, conflict_score=res.conflict_score, probe_score=res.probe_score,
        disagreement=res.disagreement,
        k_samples=settings.k_fast_samples if settings.disagreement_enabled else None,
        logprob_confidence=res.logprob_confidence, threshold=res.threshold,
        threshold_mode=res.threshold_mode, latency_budget_ms=settings.latency_budget_ms,
        reuse_method=res.reuse_method, fast_answer=res.fast_answer, final_answer=res.final_answer,
        total_tokens=res.total_tokens, total_latency_ms=res.total_latency_ms,
        total_gpu_seconds=res.total_gpu_seconds,
    )
    if item is not None:
        req.gold = str(item["gold"])
        req.correct = grade(item, res.final_answer)
        req.fast_correct = grade(item, res.fast_answer)
        req.lure_hit = is_lure(item, res.fast_answer)
    req.paths = [PathTelemetry(path=p.path, model=p.model, prompt_tokens=p.prompt_tokens,
                               completion_tokens=p.completion_tokens, reused_tokens=p.reused_tokens,
                               ttft_ms=p.ttft_ms, latency_ms=p.latency_ms, gpu_seconds=p.gpu_seconds,
                               gpu_util_pct=p.gpu_util_pct, gpu_mem_mb=p.gpu_mem_mb,
                               queue_depth=p.queue_depth) for p in res.paths]
    session.add(req)
    if commit:
        session.commit()
    else:
        session.flush()
    return req


def display_answer(text: str | None, answer_type: str = "free_text") -> str | None:
    """Short form for UI tables; full text stays in the database."""
    if text is None:
        return None
    return extract_final(text, answer_type) or text
