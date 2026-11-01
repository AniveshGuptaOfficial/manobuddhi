"""Router settings: validation, the live copy in `config`, and persistence.

Every change is saved to settings_history; the latest row is re-applied on startup,
and each eval run stores the full snapshot it ran with.
"""
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from engine.interface import RouterSettings

from .config import config
from .db_models import SettingsHistory

FIELDS = ["mode", "fast_model", "slow_model", "k_fast_samples", "threshold_mode",
          "fixed_threshold", "default_latency_budget_ms", "probe_enabled",
          "disagreement_enabled", "probe_layer", "reuse_kv_prefix", "reuse_spec_draft",
          "router_backend"]


class SettingsModel(BaseModel):
    mode: Literal["single_model_nothink_think", "small_large"]
    fast_model: str = Field(min_length=1, max_length=128)
    slow_model: str = Field(min_length=1, max_length=128)
    k_fast_samples: int = Field(ge=1, le=8)
    threshold_mode: Literal["fixed", "budget_driven"]
    fixed_threshold: float = Field(ge=0.0, le=1.0)
    default_latency_budget_ms: float = Field(ge=100, le=20000)
    probe_enabled: bool
    disagreement_enabled: bool
    probe_layer: Literal["late", "middle", "aggregate"]
    reuse_kv_prefix: bool
    reuse_spec_draft: bool
    router_backend: Literal["placeholder", "real"]


class SettingsUpdate(BaseModel):
    """Partial update: only the fields sent are changed."""
    mode: Literal["single_model_nothink_think", "small_large"] | None = None
    fast_model: str | None = Field(None, min_length=1, max_length=128)
    slow_model: str | None = Field(None, min_length=1, max_length=128)
    k_fast_samples: int | None = Field(None, ge=1, le=8)
    threshold_mode: Literal["fixed", "budget_driven"] | None = None
    fixed_threshold: float | None = Field(None, ge=0.0, le=1.0)
    default_latency_budget_ms: float | None = Field(None, ge=100, le=20000)
    probe_enabled: bool | None = None
    disagreement_enabled: bool | None = None
    probe_layer: Literal["late", "middle", "aggregate"] | None = None
    reuse_kv_prefix: bool | None = None
    reuse_spec_draft: bool | None = None
    router_backend: Literal["placeholder", "real"] | None = None


def snapshot() -> dict:
    return {f: getattr(config, f) for f in FIELDS}


def apply(values: dict) -> None:
    for k, v in values.items():
        if k in FIELDS and v is not None:
            setattr(config, k, v)


def router_settings(overrides: dict | None = None) -> RouterSettings:
    d = snapshot()
    d["latency_budget_ms"] = d.pop("default_latency_budget_ms")
    d["backend"] = d.pop("router_backend")
    clean_overrides = {}
    for k, v in (overrides or {}).items():
        if v is not None:
            if k == "router_backend":
                clean_overrides["backend"] = v
            else:
                clean_overrides[k] = v
    d.update(clean_overrides)
    return RouterSettings.from_dict(d)


def save(session: Session) -> None:
    session.add(SettingsHistory(settings=snapshot()))
    session.commit()


def load_latest(session: Session) -> bool:
    row = session.scalar(select(SettingsHistory).order_by(SettingsHistory.id.desc()).limit(1))
    if row is None:
        return False
    try:
        valid = SettingsModel(**{**snapshot(), **row.settings})
    except Exception:  # noqa: BLE001  stored row from an older schema; keep defaults
        return False
    apply(valid.model_dump())
    return True
