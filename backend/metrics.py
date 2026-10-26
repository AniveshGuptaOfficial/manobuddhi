"""In-memory metrics store for ManoBuddhi."""

import math

from .models import MetricsResponse

# Cost units (proxy for GPU seconds) - fast=1, slow=7 (Qwen2 scale), per docs.
FAST_COST = 1.0
SLOW_COST = 7.0


class Metrics:
    def __init__(self) -> None:
        self.total: int = 0
        self.fast: int = 0
        self.escalated: int = 0
        self.tokens: int = 0
        self.gpu_seconds_used: float = 0.0
        self.gpu_seconds_saved: float = 0.0
        self.latencies_ms: list[float] = []

    def record(self, route: str, tokens: int, latency_ms: float, gpu_seconds: float) -> None:
        self.total += 1
        self.tokens += tokens
        self.gpu_seconds_used += gpu_seconds
        self.latencies_ms.append(latency_ms)
        if route == "fast":
            self.fast += 1
            self.gpu_seconds_saved += (SLOW_COST - FAST_COST) * gpu_seconds
        else:
            self.escalated += 1

    def p95(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        idx = int(math.ceil(0.95 * len(s))) - 1
        return float(s[max(0, min(idx, len(s) - 1))])

    def snapshot(self) -> MetricsResponse:
        always_slow_gpu_seconds = (SLOW_COST * self.escalated) + (FAST_COST * self.fast)
        return MetricsResponse(
            total_queries=self.total,
            fast_queries=self.fast,
            escalated_queries=self.escalated,
            escalation_rate=(self.escalated / self.total) if self.total else 0.0,
            gpu_seconds_saved=self.gpu_seconds_saved,
            tokens_generated=self.tokens,
            p95_latency_ms=self.p95(),
            baseline_always_slow_gpu_seconds=always_slow_gpu_seconds if self.total else 0.0,
        )


metrics = Metrics()
