"""Database tables (F1).

eval_runs        one benchmark job (Benchmark Lab)
requests         one routed query: playground or eval item; holds the routing decision (O1, O3)
path_telemetry   one row per model path used by a request: fast and/or slow (O2, O4)
settings_history every settings change, so each result can be tied to its router config
"""
from datetime import datetime

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text,
                        func)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark: Mapped[str] = mapped_column(String(64), index=True)
    split: Mapped[str] = mapped_column(String(32))
    n_items: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|done|failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    router_config: Mapped[dict] = mapped_column(JSON)   # mode, models, K, threshold, budget...
    baseline: Mapped[str | None] = mapped_column(String(32))  # always_slow|confidence|probe_only|manobuddhi
    summary: Mapped[dict | None] = mapped_column(JSON)  # accuracy, tokens, gpu_s, p95, escalation_rate
    git_commit: Mapped[str | None] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    requests: Mapped[list["Request"]] = relationship(
        back_populates="eval_run", cascade="all, delete-orphan", passive_deletes=True)


class Request(Base):
    __tablename__ = "requests"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(),
                                                 index=True)
    source: Mapped[str] = mapped_column(String(16))  # playground|eval|api
    eval_run_id: Mapped[int | None] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"),
                                                    index=True)
    item_id: Mapped[str | None] = mapped_column(String(128))  # id from data/splits
    query: Mapped[str] = mapped_column(Text)

    # routing decision
    mode: Mapped[str] = mapped_column(String(32))        # single_model_nothink_think|small_large
    fast_model: Mapped[str] = mapped_column(String(128))
    slow_model: Mapped[str] = mapped_column(String(128))
    route: Mapped[str] = mapped_column(String(16))       # fast|escalated
    conflict_score: Mapped[float | None] = mapped_column(Float)
    probe_score: Mapped[float | None] = mapped_column(Float)
    disagreement: Mapped[float | None] = mapped_column(Float)  # fraction of K samples disagreeing
    k_samples: Mapped[int | None] = mapped_column(Integer)
    logprob_confidence: Mapped[float | None] = mapped_column(Float)  # baseline signal
    threshold: Mapped[float | None] = mapped_column(Float)
    threshold_mode: Mapped[str | None] = mapped_column(String(16))
    latency_budget_ms: Mapped[float | None] = mapped_column(Float)
    reuse_method: Mapped[str | None] = mapped_column(String(32))  # kv_prefix|speculative_draft|none

    # answers + grading
    fast_answer: Mapped[str | None] = mapped_column(Text)
    final_answer: Mapped[str | None] = mapped_column(Text)
    gold: Mapped[str | None] = mapped_column(Text)
    correct: Mapped[bool | None] = mapped_column(Boolean)
    fast_correct: Mapped[bool | None] = mapped_column(Boolean)  # label for probe / AUROC
    lure_hit: Mapped[bool | None] = mapped_column(Boolean)      # trap items: gave the intuitive answer

    # totals across paths
    total_tokens: Mapped[int | None] = mapped_column(Integer)
    total_latency_ms: Mapped[float | None] = mapped_column(Float)
    total_gpu_seconds: Mapped[float | None] = mapped_column(Float)

    eval_run: Mapped[EvalRun | None] = relationship(back_populates="requests")
    paths: Mapped[list["PathTelemetry"]] = relationship(
        back_populates="request", cascade="all, delete-orphan", passive_deletes=True)


class PathTelemetry(Base):
    __tablename__ = "path_telemetry"
    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("requests.id", ondelete="CASCADE"), index=True)
    path: Mapped[str] = mapped_column(String(16))  # fast|fast_sample|slow
    model: Mapped[str] = mapped_column(String(128))
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    reused_tokens: Mapped[int | None] = mapped_column(Integer)  # KV prefix / accepted draft tokens
    ttft_ms: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    gpu_seconds: Mapped[float | None] = mapped_column(Float)
    gpu_util_pct: Mapped[float | None] = mapped_column(Float)
    gpu_mem_mb: Mapped[float | None] = mapped_column(Float)
    queue_depth: Mapped[int | None] = mapped_column(Integer)
    request: Mapped[Request] = relationship(back_populates="paths")


class SettingsHistory(Base):
    __tablename__ = "settings_history"
    id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    settings: Mapped[dict] = mapped_column(JSON)
