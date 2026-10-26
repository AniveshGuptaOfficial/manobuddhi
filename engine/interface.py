"""The one contract between the API and any router implementation.

The placeholder router (now) and the real conflict-gated router (F5) both implement
`Router.route(RouteRequest) -> RouteResult`, so the backend never changes when the
engine does.
"""
from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class RouterSettings:
    mode: str = "small_large"  # single_model_nothink_think (Config A) | small_large (Config B)
    fast_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    slow_model: str = "Qwen/Qwen2.5-7B-Instruct"
    k_fast_samples: int = 3
    threshold_mode: str = "budget_driven"  # fixed | budget_driven
    fixed_threshold: float = 0.5
    latency_budget_ms: float = 1500.0
    probe_enabled: bool = True
    disagreement_enabled: bool = True
    probe_layer: str = "late"  # late | middle | aggregate
    reuse_kv_prefix: bool = True  # Config A
    reuse_spec_draft: bool = True  # Config B
    backend: str = "placeholder"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RouterSettings":
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class RouteRequest:
    prompt: str
    settings: RouterSettings
    item: dict | None = None  # benchmark item (gold, lure, ...) during evaluation
    force_route: str | None = None  # baselines: "fast" (always-fast) | "escalated" (always-slow)


@dataclass
class PathResult:
    path: str  # fast | fast_sample | probe | slow
    model: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reused_tokens: int = 0
    ttft_ms: float = 0.0
    latency_ms: float = 0.0
    gpu_seconds: float = 0.0
    gpu_util_pct: float | None = None
    gpu_mem_mb: float | None = None
    queue_depth: int | None = None
    energy_j: float | None = None


@dataclass
class RouteResult:
    route: str  # fast | escalated
    final_answer: str
    fast_answer: str
    slow_answer: str | None
    conflict_score: float
    probe_score: float | None
    disagreement: float | None  # fraction of K fast samples disagreeing with the fast answer
    logprob_confidence: float | None
    threshold: float
    threshold_mode: str
    reuse_method: str | None
    backend: str
    simulated: bool = False
    paths: list[PathResult] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(p.prompt_tokens + p.completion_tokens for p in self.paths)

    @property
    def completion_tokens(self) -> int:
        return sum(p.completion_tokens for p in self.paths)

    @property
    def total_gpu_seconds(self) -> float:
        return sum(p.gpu_seconds for p in self.paths)

    @property
    def total_latency_ms(self) -> float:
        """Wall-clock: K fast samples run in parallel with the fast answer; slow runs after."""
        fast = max((p.latency_ms for p in self.paths if p.path in ("fast", "fast_sample")), default=0.0)
        slow = sum(p.latency_ms for p in self.paths if p.path == "slow")
        return fast + slow


class Router(Protocol):
    name: str

    def route(self, req: RouteRequest) -> RouteResult: ...
