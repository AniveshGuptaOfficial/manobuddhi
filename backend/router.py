"""Conflict-based dual-process routing core (Gap G1 / Objectives O1, O3, O4, O5).

Implements the ManoBuddhi router:

  * ConflictScorer  - conflict score = divergence between the fast
    answer and the probe prediction, plus disagreement across K fast samples
    (Objective O1 / Gap G1).

  * Router.decide   - compares conflict score against a threshold
    (fixed, or budget-driven from a latency budget - Objective O3 / Gap G3)
    and issues a Fast / Escalate decision (Gap G4 / Objective O4 support via
    escalation source selection).

  * Escalation source - "speculative_draft" (fast answer reused as draft,
    Gap G4 / Objective O4) or "think" (single-model no-think -> think, G5).
"""

from .config import config


def _agreement(a: str, b: str) -> float:
    """Token-overlap agreement between two answer strings (stand-in until
    embedding-based similarity is wired to real model outputs)."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


class ConflictScorer:
    def __init__(self, k: int | None = None) -> None:
        self.k = k if k is not None else config.k_fast_samples

    def score(self, question: str, fast_answers: list[str], probe_answer: str, hidden_conf: float = 0.0) -> float:
        # Objective O1: conflict = divergence between emitted fast answer and
        # probe prediction + disagreement across K fast samples.
        if not fast_answers:
            # No fast answers yet -> fall back to hidden-state confidence signal.
            return float(min(1.0, max(0.0, 1.0 - hidden_conf)))

        agreements = [_agreement(ans, probe_answer) for ans in fast_answers[: self.k]]
        agreement = sum(agreements) / len(agreements) if agreements else 0.0
        conflict = 1.0 - agreement

        # Blend with hidden-state based conflict signal (probe confidence proxy).
        hs_component = 1.0 - float(min(1.0, max(0.0, hidden_conf)))
        conflict = float(min(1.0, max(0.0, 0.7 * conflict + 0.3 * hs_component)))

        return round(conflict, 4)


class Router:
    def __init__(self, scorer: ConflictScorer | None = None) -> None:
        self.scorer = scorer or ConflictScorer()

    # Gap G3 / Objective O3: threshold selection.
    def _threshold(self, latency_budget_ms: float | None = None) -> float:
        budget = float(latency_budget_ms) if latency_budget_ms is not None else float(config.default_latency_budget_ms)
        if budget <= 0:
            return float(config.fixed_threshold)
        # Stand-in budget-driven threshold: tighter budget -> escalate only on
        # high conflict (higher threshold = fewer escalations = lower latency).
        derived = 1.0 - (budget / 500.0)  # budget in ms -> [0,1]
        threshold = float(min(config.fixed_threshold, max(0.0, derived)))
        return round(threshold, 4)

    def decide(
        self,
        query: str,
        conflict_score: float,
        latency_budget_ms: float | None = None,
        threshold_mode: str | None = None,
        fixed_threshold: float | None = None,
        mode: str | None = None,
    ) -> dict:
        theta = float(fixed_threshold if threshold_mode == "fixed" else self._threshold(latency_budget_ms))
        escalate = bool(conflict_score >= theta)

        eff_mode = mode if mode is not None else config.mode
        eff_source = None
        if escalate:
            if eff_mode == "single_model_nothink_think":
                eff_source = "think"  # no-think -> think within same model
            else:
                # Config B: small -> large. Reuse fast answer as speculative draft
                # (Gap G4 / Objective O4) before falling back to the slow model.
                eff_source = "speculative_draft"

        return {
            "route": "escalate" if escalate else "fast",
            "conflict_score": float(conflict_score),
            "threshold": theta,
            "escalated": escalate,
            "escalation_source": eff_source,
            "mode": eff_mode,
        }


def stub_fast_answer(query: str) -> str:
    return f"[Fast:{config.fast_model}] Fast-path answer generated for query: {query}"


def stub_slow_answer(query: str) -> str:
    return f"[Slow:{config.slow_model}] Slow-path reasoning + final answer for query: {query}"


def run_chat(query: str, conflict_score: float, latency_budget_ms: float | None = None, threshold_mode: str | None = None, fixed_threshold: float | None = None, mode: str | None = None) -> dict:
    """End-to-end stub orchestration: score -> route -> (fast answer | slow answer)."""
    router = Router()
    decision = router.decide(
        query=query,
        conflict_score=conflict_score,
        latency_budget_ms=latency_budget_ms,
        threshold_mode=threshold_mode,
        fixed_threshold=fixed_threshold,
        mode=mode,
    )
    content = stub_slow_answer(query) if decision["escalated"] else stub_fast_answer(query)
    return {
        **decision,
        "content": content,
        "tokens_generated": 0 if decision["route"] == "fast" else 0,
        "latency_ms": float(latency_budget_ms or config.default_latency_budget_ms),
        "gpu_seconds": 0.0 if decision["route"] == "fast" else 0.0,
    }
