"""Runtime configuration for the ManoBuddhi dual-process LLM router.

Values come from (highest priority first): environment variables prefixed MB_,
the project-root .env file, then the defaults below. Copy .env.example to .env.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_prefix="MB_", extra="ignore"
    )

    env: str = "dev"  # dev | test | prod

    # Dual-process model pair (G5 / O5)
    # "single_model_nothink_think": one Qwen3 model, no-think -> think (Config A)
    # "small_large": small fast model -> large model (Config B)
    mode: str = "small_large"
    fast_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    slow_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Conflict score (G1 / O1)
    k_fast_samples: int = 3

    # Threshold policy (G3 / O3)
    threshold_mode: str = "budget_driven"  # fixed | budget_driven
    fixed_threshold: float = 0.5
    default_latency_budget_ms: float = 1500.0

    # Conflict signal + fast-path reuse toggles (Settings screen)
    probe_enabled: bool = True
    disagreement_enabled: bool = True
    probe_layer: str = "late"  # late | middle | aggregate
    reuse_kv_prefix: bool = True  # Config A
    reuse_spec_draft: bool = True  # Config B

    # Router implementation: "placeholder" until F5
    router_backend: str = "placeholder"

    # Benchmark jobs: "rq" (Docker worker) or "inline" (inside the API process)
    eval_executor: str = "rq"
    eval_queue: str = "manobuddhi-eval"

    # Infrastructure
    database_url: str = "postgresql+psycopg://manobuddhi:manobuddhi@127.0.0.1:5432/manobuddhi"
    redis_url: str = "redis://127.0.0.1:6379/0"
    vllm_fast_url: str = "http://localhost:8100/v1"
    vllm_slow_url: str = "http://localhost:8200/v1"
    cors_origins: list[str] = ["http://localhost:8000", "http://127.0.0.1:8000"]

    # Data and reproducibility
    data_dir: Path = PROJECT_ROOT / "data"
    seed: int = 42


config = Config()
