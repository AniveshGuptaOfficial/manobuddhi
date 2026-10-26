"""PlaceholderRouter: stands in for the real engine until F5. No models, no GPU.

Deterministic (seeded by the prompt) so tests and demos are repeatable. Every result
is marked simulated=True and backend="placeholder"; none of its numbers are research
results. With a benchmark item it simulates correct/incorrect answers (and lure answers
on trap items) so grading, AUROC and the dashboard can be exercised end to end.
"""
import hashlib
import random
import re

from .interface import PathResult, RouteRequest, RouteResult, RouterSettings
from .threshold import choose_threshold

_HARD = re.compile(r"\b(prove|integral|how many|how long|how much|if|probability|solve|"
                   r"more than|twice|each|rate|percent)\b", re.IGNORECASE)


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def _difficulty(prompt: str) -> float:
    d = 0.15 + 0.35 * min(len(prompt) / 600, 1.0) + 0.08 * min(len(_HARD.findall(prompt)), 4)
    return min(d, 0.95)


def _wrong(item: dict, rng: random.Random) -> str:
    lure = (item.get("meta") or {}).get("lure")
    if lure and rng.random() < 0.8:
        return lure
    gold = str(item["gold"])
    t = item["answer_type"]
    if t == "yes_no":
        return "no" if gold == "yes" else "yes"
    if t == "multiple_choice":
        others = [c["label"] for c in item.get("choices", []) if c["label"] != gold]
        return rng.choice(others) if others else "Z"
    try:
        return str(float(gold) + rng.choice([-2, -1, 1, 2, 10])).removesuffix(".0")
    except ValueError:
        return "unknown"


class PlaceholderRouter:
    name = "placeholder"

    def route(self, req: RouteRequest) -> RouteResult:
        s: RouterSettings = req.settings
        rng = _rng(req.prompt, s.mode, s.fast_model, s.slow_model)
        diff = _difficulty(req.prompt)

        probe = min(1.0, max(0.0, 0.75 * diff + 0.25 * rng.random()))
        k = max(1, s.k_fast_samples)
        n_disagree = sum(1 for _ in range(k) if rng.random() < diff * 0.9)
        disagreement = n_disagree / k
        parts = []
        if s.probe_enabled:
            parts.append(probe)
        if s.disagreement_enabled:
            parts.append(disagreement)
        conflict = sum(parts) / len(parts) if parts else 0.0
        logprob_conf = round(1.0 - 0.6 * diff + 0.1 * rng.random(), 4)  # baseline signal

        threshold = choose_threshold(s)
        route = req.force_route or ("escalated" if conflict >= threshold else "fast")

        # answers
        item = req.item
        if item is not None:
            fast_ok = rng.random() < max(0.05, 0.92 - 0.75 * probe)
            slow_ok = rng.random() < 0.9
            fast_text = f"Answer: {item['gold'] if fast_ok else _wrong(item, rng)}"
            slow_text = f"Answer: {item['gold'] if slow_ok else _wrong(item, rng)}"
        else:
            fast_text = f"[placeholder fast answer from {s.fast_model}]"
            slow_text = f"[placeholder slow reasoning answer from {s.slow_model}]"

        # costs: fast model is ~5x cheaper; Config A's 'think' mode is one model thinking longer
        prompt_tokens = max(8, len(req.prompt) // 4)
        fast_tokens = rng.randint(15, 60)
        fast_lat = rng.uniform(80, 250)
        fast_model = s.fast_model if s.mode == "small_large" else s.slow_model + " (no-think)"
        paths = [PathResult("fast", fast_model, fast_text, prompt_tokens, fast_tokens,
                            ttft_ms=fast_lat * 0.3, latency_ms=fast_lat, gpu_seconds=fast_lat / 1000 * 0.2)]
        if s.disagreement_enabled:
            for _ in range(k - 1):
                lat = rng.uniform(80, 250)
                paths.append(PathResult("fast_sample", fast_model, "", 0, rng.randint(15, 60),
                                        reused_tokens=prompt_tokens, latency_ms=lat,
                                        gpu_seconds=lat / 1000 * 0.2))

        reuse_method, slow_answer = None, None
        if route == "escalated":
            reused = 0
            if s.mode == "single_model_nothink_think" and s.reuse_kv_prefix:
                reuse_method, reused = "kv_prefix", prompt_tokens
            elif s.mode == "small_large" and s.reuse_spec_draft:
                reuse_method, reused = "speculative_draft", int(fast_tokens * rng.uniform(0.3, 0.7))
            slow_tokens = rng.randint(200, 500)
            saving = 0.85 if reuse_method else 1.0
            slow_lat = rng.uniform(900, 1800) * saving
            slow_model = s.slow_model if s.mode == "small_large" else s.slow_model + " (think)"
            paths.append(PathResult("slow", slow_model, slow_text, 0 if reused == prompt_tokens else prompt_tokens,
                                    slow_tokens, reused_tokens=reused, ttft_ms=slow_lat * 0.15 * saving,
                                    latency_ms=slow_lat, gpu_seconds=slow_lat / 1000 * 0.9))
            slow_answer = slow_text

        return RouteResult(
            route=route, final_answer=slow_text if route == "escalated" else fast_text,
            fast_answer=fast_text, slow_answer=slow_answer,
            conflict_score=round(conflict, 4), probe_score=round(probe, 4) if s.probe_enabled else None,
            disagreement=round(disagreement, 4) if s.disagreement_enabled else None,
            logprob_confidence=logprob_conf, threshold=round(threshold, 4), threshold_mode=s.threshold_mode,
            reuse_method=reuse_method, backend=self.name, simulated=True, paths=paths)


def get_router(backend: str = "placeholder"):
    if backend == "placeholder":
        return PlaceholderRouter()
    raise NotImplementedError(f"router backend '{backend}' arrives in F4/F5")
