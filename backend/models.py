"""Pydantic schemas for the ManoBuddhi API."""
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    """OpenAI-style `messages`, or a plain `query`. Router overrides are optional."""
    messages: list[ChatMessage] | None = None
    query: str | None = None
    model: str | None = None
    stream: bool = False
    latency_budget_ms: float | None = Field(None, ge=100, le=20000)
    threshold_mode: Literal["fixed", "budget_driven"] | None = None
    fixed_threshold: float | None = Field(None, ge=0.0, le=1.0)
    mode: Literal["single_model_nothink_think", "small_large"] | None = None

    def user_text(self) -> str:
        if self.messages:
            users = [m.content for m in self.messages if m.role == "user"]
            if users:
                return users[-1]
        return (self.query or "").strip()


class EvalCreate(BaseModel):
    benchmark: str
    split: str = "test"
    n: int | None = Field(None, ge=1, le=100000)  # None = whole split
    baseline: Literal["manobuddhi", "always_fast", "always_slow"] = "manobuddhi"


class EvalRunOut(BaseModel):
    id: int
    benchmark: str
    split: str
    n_items: int
    status: str
    progress: float
    baseline: str | None
    router_config: dict
    summary: dict | None
    created_at: str
    finished_at: str | None
