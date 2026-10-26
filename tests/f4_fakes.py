"""Synthetic F4 records shaped exactly like engine/collect.py output, for CPU tests."""
import random

import numpy as np


def gen(model, correct_text, *, tokens=100, lat=1000.0, gpu=1.0, conf=0.9, prompt_tokens=80):
    return {"model": model, "text": correct_text, "raw_text": correct_text, "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens, "finished": True, "ttft_ms": lat * 0.05, "latency_ms": lat,
            "gpu_seconds": gpu, "gpu_busy_seconds": gpu * 0.6, "gpu_util_pct": 60.0, "gpu_mem_mb": 3000.0,
            "energy_j": gpu * 70, "devices": [0], "timed": True, "mean_logprob": float(np.log(conf)),
            "conf_seq": conf, "conf_answer": conf, "first_token_entropy": 1 - conf, "thinking": None,
            "reasoning_tokens": None}


def records(n=200, split="test", seed=0, fast_acc=0.6, slow_acc=0.9, benchmark="gsm8k"):
    """Fast model wrong ~40%; its confidence is informative but noisy. Slow: 9x the cost."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        gold = str(i % 50 + 1)
        fast_ok = rng.random() < fast_acc
        slow_ok = rng.random() < (slow_acc if not fast_ok else 0.97)
        conf = min(0.99, max(0.05, (0.85 if fast_ok else 0.55) + rng.gauss(0, 0.15)))
        item = {"question": f"q{i}", "gold": gold, "answer_type": "numeric", "aliases": [], "choices": [],
                "meta": {}}
        f = gen("fast-1.5b", f"Answer: {gold if fast_ok else '0'}", tokens=60, lat=600, gpu=0.6, conf=conf)
        s = gen("slow-7b", f"Answer: {gold if slow_ok else '0'}", tokens=180, lat=4000, gpu=5.4, conf=0.9)
        out.append({"id": f"{benchmark}-{split}-{i}", "benchmark": benchmark, "split": split, "item": item,
                    "prompt": f"q{i}\n\nEnd with 'Answer: <answer>'.", "fast": f, "slow": s,
                    "probe": {"latency_ms": 40.0, "gpu_seconds": 0.04, "gpu_busy_seconds": 0.03,
                              "energy_j": 2.0, "devices": [0], "tokens": 140},
                    "fast_correct": fast_ok, "slow_correct": slow_ok, "fast_lure": None, "slow_lure": None})
    return out


def probe_scores(recs, seed=1, strength=0.35):
    """A probe that separates errors better than confidence does."""
    rng = random.Random(seed)
    return {r["id"]: min(1, max(0, (0.5 + strength if not r["fast_correct"] else 0.5 - strength)
                               + rng.gauss(0, 0.2))) for r in recs}
