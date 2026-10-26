"""In-memory benchmark job manager for ManoBuddhi evaluation."""

import time
import uuid

from .models import EvalJobResponse

VALID_BENCHMARKS = ["GSM8K", "MATH", "StrategyQA", "CommonsenseQA", "CRTrapItems"]


class EvalManager:
    def __init__(self) -> None:
        self.jobs: dict[str, EvalJobResponse] = {}

    def create(self, benchmark: str) -> EvalJobResponse:
        job_id = str(uuid.uuid4())
        job = EvalJobResponse(job_id=job_id, status="queued", benchmark=benchmark, progress=0.0, results=None)
        self.jobs[job_id] = job
        return job

    def get(self, job_id: str) -> EvalJobResponse | None:
        return self.jobs.get(job_id)

    def start(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is not None:
            self.jobs[job_id] = job.model_copy(update={"status": "running", "progress": 0.05})

    def complete(self, job_id: str, results: dict) -> None:
        job = self.jobs.get(job_id)
        if job is not None:
            self.jobs[job_id] = job.model_copy(update={"status": "completed", "progress": 1.0, "results": results})

    def status_response(self, job_id: str) -> EvalJobResponse | None:
        return self.jobs.get(job_id)


eval_manager = EvalManager()


def simulate_benchmark_run(job_id: str, benchmark: str) -> None:
    """Simulated benchmark execution; replace with real harness calls."""
    ev = eval_manager
    ev.start(job_id)
    time.sleep(0.2)
    ev.complete(
        job_id,
        results={
            "benchmark": benchmark,
            "baseline_accuracy": 0.0,  # filled by real runs vs always-slow / always-fast / LBR
            "router_accuracy": 0.0,
            "tokens_generated": 0,
            "gpu_seconds": 0.0,
            "p95_latency_ms": 0.0,
        },
    )
