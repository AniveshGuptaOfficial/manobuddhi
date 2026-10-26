"""Threshold selection. PROVISIONAL: F6 replaces budget_driven with the learn-then-test
certified set + load-adaptive controller (O3). Kept separate so only this file changes."""
from .interface import RouterSettings

REFERENCE_BUDGET_MS = 1500.0


def choose_threshold(s: RouterSettings, queue_depth: int = 0) -> float:
    if s.threshold_mode == "fixed":
        return s.fixed_threshold
    # tighter latency budget or longer queue -> higher threshold -> fewer escalations
    pressure = (REFERENCE_BUDGET_MS - s.latency_budget_ms) / REFERENCE_BUDGET_MS + 0.05 * queue_depth
    pressure = max(-1.0, min(1.0, pressure))
    return max(0.05, min(0.95, s.fixed_threshold + 0.2 * pressure))
