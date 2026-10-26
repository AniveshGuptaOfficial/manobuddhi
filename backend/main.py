"""ManoBuddhi FastAPI application (F2: API gateway)."""
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from engine.interface import RouteRequest

from . import eval as evals
from . import metrics, settings
from .config import config
from .db import SessionLocal, engine, get_session
from .db_models import EvalRun, Request
from .models import ChatRequest, EvalCreate
from .persistence import display_answer, save_result
from .router import active_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        with SessionLocal() as s:
            settings.load_latest(s)
    except Exception:  # noqa: BLE001  DB down: start with .env defaults; /v1/system shows it
        pass
    yield


app = FastAPI(title="ManoBuddhi - Dual-Process LLM Inference Router", version="0.2.0",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=config.cors_origins, allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "manobuddhi-router"}


# ---------------- chat ----------------

@app.post("/v1/chat/completions")
def chat_completions(body: ChatRequest, s: Session = Depends(get_session)) -> dict:
    if body.stream:
        raise HTTPException(400, "stream=true is not supported yet")
    q = body.user_text()
    if not q:
        raise HTTPException(422, "send `messages` with a user message, or `query`")
    rs = settings.router_settings({"latency_budget_ms": body.latency_budget_ms,
                                   "threshold_mode": body.threshold_mode,
                                   "fixed_threshold": body.fixed_threshold, "mode": body.mode})
    res = active_router(rs.backend).route(RouteRequest(q, rs))
    req = save_result(s, res, query=q, settings=rs, source="playground")
    final_model = rs.slow_model if res.route == "escalated" else rs.fast_model
    completion = res.completion_tokens
    return {
        # OpenAI-compatible part
        "id": f"mb-{req.id}", "object": "chat.completion", "created": int(time.time()),
        "model": final_model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": res.final_answer},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": res.total_tokens - completion, "completion_tokens": completion,
                  "total_tokens": res.total_tokens},
        # ManoBuddhi routing details (used by the Playground + Conflict Inspector)
        "request_id": req.id, "route": res.route, "content": res.final_answer,
        "fast_answer": res.fast_answer, "slow_answer": res.slow_answer,
        "conflict_score": res.conflict_score, "probe_score": res.probe_score,
        "disagreement": res.disagreement, "k_samples": rs.k_fast_samples,
        "confidence": res.logprob_confidence, "threshold": res.threshold,
        "threshold_mode": res.threshold_mode, "latency_budget_ms": rs.latency_budget_ms,
        "reuse_method": res.reuse_method, "latency_ms": res.total_latency_ms,
        "gpu_seconds": res.total_gpu_seconds, "router_backend": res.backend, "simulated": res.simulated,
        "paths": [{"path": p.path, "model": p.model, "completion_tokens": p.completion_tokens,
                   "reused_tokens": p.reused_tokens, "ttft_ms": p.ttft_ms, "latency_ms": p.latency_ms,
                   "gpu_seconds": p.gpu_seconds} for p in res.paths],
    }


# ---------------- metrics + request log ----------------

@app.get("/v1/metrics")
def get_metrics(source: str | None = None, eval_run_id: int | None = None,
                s: Session = Depends(get_session)) -> dict:
    return metrics.compute(s, source=source, eval_run_id=eval_run_id)


@app.get("/v1/requests")
def recent_requests(limit: int = Query(50, ge=1, le=500), source: str | None = None,
                    s: Session = Depends(get_session)) -> list[dict]:
    stmt = select(Request).order_by(Request.id.desc()).limit(limit)
    if source:
        stmt = stmt.where(Request.source == source)
    return [{"id": r.id, "created_at": r.created_at.isoformat() if r.created_at else None,
             "source": r.source, "query": r.query[:300], "route": r.route,
             "conflict_score": r.conflict_score, "threshold": r.threshold,
             "latency_ms": r.total_latency_ms, "tokens": r.total_tokens,
             "gpu_seconds": r.total_gpu_seconds, "answer": display_answer(r.final_answer),
             "correct": r.correct} for r in s.scalars(stmt)]


# ---------------- benchmarks + eval ----------------

@app.get("/v1/benchmarks")
def list_benchmarks() -> list[dict]:
    return evals.benchmarks()


@app.post("/v1/eval")
def create_eval(body: EvalCreate, background: BackgroundTasks, s: Session = Depends(get_session)) -> dict:
    return evals.to_out(evals.create(s, body, background))


@app.get("/v1/eval")
def list_evals(limit: int = Query(50, ge=1, le=500), s: Session = Depends(get_session)) -> list[dict]:
    return [evals.to_out(r) for r in s.scalars(select(EvalRun).order_by(EvalRun.id.desc()).limit(limit))]


@app.get("/v1/eval/{run_id}")
def get_eval(run_id: int, s: Session = Depends(get_session)) -> dict:
    r = s.get(EvalRun, run_id)
    if r is None:
        raise HTTPException(404, "eval run not found")
    return evals.to_out(r)


@app.get("/v1/eval/{run_id}/results.csv", response_class=PlainTextResponse)
def eval_csv(run_id: int, s: Session = Depends(get_session)) -> PlainTextResponse:
    if s.get(EvalRun, run_id) is None:
        raise HTTPException(404, "eval run not found")
    return PlainTextResponse(evals.results_csv(s, run_id), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="manobuddhi-eval-{run_id}.csv"'})


# ---------------- settings ----------------

@app.get("/v1/settings")
def get_settings() -> dict:
    return settings.snapshot()


@app.put("/v1/settings")
def put_settings(body: settings.SettingsUpdate, s: Session = Depends(get_session)) -> dict:
    settings.apply(body.model_dump(exclude_none=True))
    settings.save(s)
    return settings.snapshot()


# ---------------- system status ----------------

@app.get("/v1/system")
def system_status() -> dict:
    out = {"router_backend": config.router_backend, "eval_executor": config.eval_executor,
           "splits_ready": bool(evals.benchmarks())}
    try:
        with engine.connect() as c:
            out["postgres"] = {"ok": True, "requests_rows": c.execute(text("select count(*) from requests")).scalar()}
    except Exception as e:  # noqa: BLE001
        out["postgres"] = {"ok": False, "error": str(e).splitlines()[0]}
    try:
        from redis import Redis
        from rq import Queue, Worker

        r = Redis.from_url(config.redis_url, socket_connect_timeout=2)
        r.ping()
        out["redis"] = {"ok": True, "queue_length": len(Queue(config.eval_queue, connection=r)),
                        "workers": len(Worker.all(connection=r))}
    except Exception as e:  # noqa: BLE001
        out["redis"] = {"ok": False, "error": str(e).splitlines()[0]}
    return out


app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "frontend"), html=True),
          name="frontend")
