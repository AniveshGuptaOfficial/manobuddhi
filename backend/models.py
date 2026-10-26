"""Pydantic schemas for the ManoBuddhi API."""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    # Optional overrides; server defaults come from Config
    model_pair: dict | None = None
    latency_budget_ms: float | None = None
    threshold_mode: str | None = None
    fixed_threshold: float | None = None
    # Optional pre-computed conflict score (used for ablation / testing of O1)
    conflict_score: float | None = None


class ChatResponse(BaseModel):
    route: str  # "fast" | "escalate"
    content: str
    conflict_score: float
    probe_score: float
    threshold: float
    escalated: bool
    escalation_source: str | None  # None | "speculative_draft" | "think" | "small_large_fallback"
    mode: str  # "single_model_nothink_think" | "small_large"
    tokens_generated: int
    latency_ms: float
    gpu_seconds: float


class MetricsResponse(BaseModel):
    total_queries: int
    fast_queries: int
    escalated_queries: int
    escalation_rate: float
    gpu_seconds_saved: float
    tokens_generated: int
    p95_latency_ms: float
    baseline_always_slow_gpu_seconds: float


class EvalJobResponse(BaseModel):
    job_id: str
    status: str  # queued | running | completed
    benchmark: str | None = None
    progress: float = 0.0
    results: dict | None = None


class SettingsResponse(BaseModel):
    mode: str
    fast_model: str
    slow_model: str
    threshold_mode: str
    fixed_threshold: float
    default_latency_budget_ms: float
    k_fast_samples: int
