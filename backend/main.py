"""ManoBuddhi FastAPI application."""

from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path

from . import eval as eval_module
from . import metrics as metrics_module
from . import models as models_module
from . import router as router_module
from . import settings as settings_module

app = FastAPI(title="ManoBuddhi - Dual-Process LLM Inference Router")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings_module.config.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "manobuddhi-router"}


@app.post("/v1/chat/completions", response_model=models_module.ChatResponse)
def chat_completions(payload: models_module.ChatRequest, background: BackgroundTasks) -> dict:
    scorer = router_module.ConflictScorer()
    conflict_score = payload.conflict_score
    if conflict_score is None:
        conflict_score = scorer.score(
            question=payload.query,
            fast_answers=["stub-fast-answer"],
            probe_answer="stub-probe-answer",
            hidden_conf=0.3,
        )

    decision = router_module.Router().decide(
        query=payload.query,
        conflict_score=conflict_score,
        latency_budget_ms=payload.latency_budget_ms,
        threshold_mode=payload.threshold_mode,
        fixed_threshold=payload.fixed_threshold,
    )
    response = router_module.run_chat(
        query=payload.query,
        conflict_score=conflict_score,
        latency_budget_ms=payload.latency_budget_ms,
        threshold_mode=payload.threshold_mode,
        fixed_threshold=payload.fixed_threshold,
    )
    response.update(decision)

    metrics_module.metrics.record(
        route=response["route"],
        tokens=response["tokens_generated"],
        latency_ms=response["latency_ms"],
        gpu_seconds=response["gpu_seconds"],
    )

    return {
        "route": response["route"],
        "content": response["content"],
        "conflict_score": response["conflict_score"],
        "probe_score": 0.3,
        "threshold": response["threshold"],
        "escalated": response["escalated"],
        "escalation_source": response["escalation_source"],
        "mode": response["mode"],
        "tokens_generated": response["tokens_generated"],
        "latency_ms": response["latency_ms"],
        "gpu_seconds": response["gpu_seconds"],
    }


@app.get("/v1/metrics", response_model=models_module.MetricsResponse)
def get_metrics() -> dict:
    return metrics_module.metrics.snapshot().model_dump()


@app.post("/v1/eval", response_model=models_module.EvalJobResponse)
def create_eval(payload: dict, background: BackgroundTasks) -> models_module.EvalJobResponse:
    benchmark = payload.get("benchmark", "GSM8K")
    job = eval_module.eval_manager.create(benchmark=benchmark)
    background.add_task(eval_module.simulate_benchmark_run, job.job_id, benchmark)
    return job


@app.get("/v1/eval/{job_id}", response_model=models_module.EvalJobResponse)
def get_eval(job_id: str) -> dict:
    job = eval_module.eval_manager.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"detail": "job not found"})
    return job.model_dump()


@app.get("/v1/settings", response_model=models_module.SettingsResponse)
def get_settings() -> dict:
    cfg = settings_module.config
    return {
        "mode": cfg.mode,
        "fast_model": cfg.fast_model,
        "slow_model": cfg.slow_model,
        "threshold_mode": cfg.threshold_mode,
        "fixed_threshold": cfg.fixed_threshold,
        "default_latency_budget_ms": cfg.default_latency_budget_ms,
        "k_fast_samples": cfg.k_fast_samples,
    }


@app.put("/v1/settings", response_model=models_module.SettingsResponse)
def put_settings(payload: dict) -> dict:
    cfg = settings_module.config
    if "mode" in payload:
        cfg.mode = str(payload["mode"])
    if "fast_model" in payload:
        cfg.fast_model = str(payload["fast_model"])
    if "slow_model" in payload:
        cfg.slow_model = str(payload["slow_model"])
    if "threshold_mode" in payload:
        cfg.threshold_mode = str(payload["threshold_mode"])
    if "fixed_threshold" in payload:
        cfg.fixed_threshold = float(payload["fixed_threshold"])
    if "default_latency_budget_ms" in payload:
        cfg.default_latency_budget_ms = float(payload["default_latency_budget_ms"])
    if "k_fast_samples" in payload:
        cfg.k_fast_samples = int(payload["k_fast_samples"])
    return {
        "mode": cfg.mode,
        "fast_model": cfg.fast_model,
        "slow_model": cfg.slow_model,
        "threshold_mode": cfg.threshold_mode,
        "fixed_threshold": cfg.fixed_threshold,
        "default_latency_budget_ms": cfg.default_latency_budget_ms,
        "k_fast_samples": cfg.k_fast_samples,
    }


app.mount("/", StaticFiles(directory=str(Path(__file__).resolve().parent.parent / "frontend"), html=True), name="frontend")
