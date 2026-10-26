"""Collect real fast/slow generations for a benchmark split (F4). Runs on the GPU machine.

Every item gets the fast answer (with hidden-state features for the probe) and, if asked,
the slow answer, both greedy, at batch 1 so latency and GPU-seconds are what one live
query costs. The three baselines are then computed from these paired runs in
engine/baselines.py, without spending GPU time again: for batch-1 sequential serving,
a cascade's cost on an item is exactly fast cost (+ probe cost) (+ slow cost if escalated).

probe_train collections only need the fast answer and its features, and their timing is
not reported, so they may use batch_size > 1 to go faster.
"""
from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .data import load_split
from .grading import grade, is_lure
from .prompts import build_prompt
from .results import RunDir, key
from .serving import HFModel, default_layers


@dataclass
class ModelPair:
    """Config B: fast and slow are different models. Config A: the same HFModel with
    thinking off (fast) and on (slow)."""
    mode: str  # small_large | single_model_nothink_think
    fast: HFModel
    slow: HFModel | None
    fast_thinking: bool | None = None
    slow_thinking: bool | None = None
    slow_sampling: dict | None = None  # Qwen3 thinking: sample, never greedy (Qwen guidance)

    def describe(self) -> dict:
        return {"mode": self.mode, "fast": self.fast.describe(),
                "slow": self.slow.describe() if self.slow else None,
                "fast_thinking": self.fast_thinking, "slow_thinking": self.slow_thinking,
                "slow_sampling": self.slow_sampling}


def load_pair(mode: str = "small_large", *, fast_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
              slow_model: str = "Qwen/Qwen2.5-7B-Instruct", fast_device: str = "cuda:0",
              slow_device_map: str | None = "sequential", slow_max_memory: dict | None = None,
              slow_device: str = "cuda:1", slow_quant: str = "none", with_slow: bool = True,
              meter=None) -> ModelPair:
    """Defaults for Kaggle's 2x T4 (15 GB each). Qwen2.5-7B in float16 is ~15.2 GB, which
    does not fit one T4 with room for the KV cache, so it is split: a few layers on GPU 0
    next to the 1.5B fast model, the rest on GPU 1. slow_quant='4bit' puts it on GPU 1 alone."""
    from .telemetry import GpuMeter

    meter = meter or GpuMeter()
    if mode == "single_model_nothink_think":
        m = HFModel(slow_model, device=fast_device, meter=meter)
        return ModelPair(mode, m, m, fast_thinking=False, slow_thinking=True,
                         slow_sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    fast = HFModel(fast_model, device=fast_device, meter=meter)
    if not with_slow:
        return ModelPair(mode, fast, None)
    if slow_quant != "none":
        slow = HFModel(slow_model, device=slow_device, quant=slow_quant, meter=meter)
    else:
        slow = HFModel(slow_model, device_map=slow_device_map,
                       max_memory=slow_max_memory or {0: "4GiB", 1: "12GiB"}, meter=meter)
    return ModelPair(mode, fast, slow)


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def environment(meter) -> dict:
    import torch
    import transformers

    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "hardware": meter.hardware(), "platform": platform.platform()}


def _seed(item_id: str) -> int:
    """Per-item seed for sampled (thinking) runs: same item -> same sample, across resumes."""
    import hashlib

    return int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)


def warmup(pair: ModelPair, log=print) -> None:
    """One untimed call per model before any measurement: the first CUDA call pays one-off
    kernel loading / allocation (seen as ~1.3 s TTFT on T4), which is not model cost."""
    if getattr(pair, "_warm", False):
        return
    for m, thinking in ((pair.fast, pair.fast_thinking), (pair.slow, pair.slow_thinking)):
        if m is not None:
            m.generate(["What is 2 + 3?"], max_new_tokens=8, thinking=thinking)
            m.probe_features(*m.generate(["Hi"], max_new_tokens=2, thinking=thinking)[0][1:], [1])
    pair._warm = True
    log("warm-up done (not recorded)")


def _record(item: dict, prompt: str, fast=None, slow=None, probe_cost=None) -> dict:
    r = {"id": item["id"], "benchmark": item["benchmark"], "split": item.get("split"),
         "item": {k: item[k] for k in ("question", "gold", "answer_type", "aliases", "choices", "meta")
                  if k in item},
         "prompt": prompt, "fast": fast.to_dict() if fast else None,
         "slow": slow.to_dict() if slow else None, "probe": probe_cost}
    for name, g in (("fast", fast), ("slow", slow)):
        r[f"{name}_correct"] = grade(item, g.text) if g else None
        r[f"{name}_lure"] = is_lure(item, g.text) if g else None
    return r


def collect(run: RunDir, pair: ModelPair, benchmark: str, split: str, *, n: int | None = None,
            paths: tuple[str, ...] = ("fast", "slow"), hidden: bool = True,
            layers: list[int] | None = None, batch_size: int = 1, max_new_fast: int = 512,
            max_new_slow: int = 768, shard_size: int = 25, shard: tuple[int, int] | None = None,
            deadline: float | None = None, log=print) -> dict:
    if batch_size > 1 and "slow" in paths:
        raise ValueError("batched collection is for fast-only probe_train runs; use batch_size=1")
    items = load_split(benchmark, split)
    items = items[:n] if n else items
    if shard:  # (i, k): this process takes items i, i+k, i+2k, ... (one process per GPU)
        items = items[shard[0]::shard[1]]
    done = {r["id"] for r in run.load_records(benchmark, split)}
    todo = [it for it in items if it["id"] not in done]
    layers = layers or default_layers(pair.fast.n_layers)

    meter = pair.fast.meter
    m = run.manifest()
    if not m.get("models"):
        run.update_manifest(models=pair.describe(), environment=environment(meter),
                            git_commit=_git_commit(), probe_layers=layers,
                            generation={"decoding": "greedy", "max_new_fast": max_new_fast,
                                        "max_new_slow": max_new_slow, "dtype": "float16"})
    elif m["models"]["fast"]["name"] != pair.fast.name or (
            pair.slow and m["models"].get("slow") and m["models"]["slow"]["name"] != pair.slow.name):
        raise ValueError(f"run '{run.path.name}' was collected with other models; use a new run name")
    else:
        layers = m.get("probe_layers", layers)
        if pair.slow and not m["models"].get("slow"):  # first run was fast-only (probe_train)
            run.update_manifest(models=pair.describe())

    info = {"n_target": len(items), "paths": list(paths), "hidden": hidden,
            "batch_size": batch_size, "timed": batch_size == 1,
            "shard": f"{shard[0]}/{shard[1]}" if shard else None}
    log(f"[{key(benchmark, split)}] {len(done)} done, {len(todo)} to go "
        f"(paths={','.join(paths)}, batch={batch_size}, layers={layers})")
    if todo:
        warmup(pair, log)
    t_start, n_new = time.time(), 0

    for s in range(0, len(todo), shard_size):
        if deadline and time.time() > deadline:  # stop cleanly so Kaggle saves output; rerun resumes
            log(f"  deadline reached: stopping {key(benchmark, split)} at {len(done) + n_new}/{len(items)}")
            info["stopped_at_deadline"] = True
            break
        chunk = todo[s:s + shard_size]
        recs, feats = [], []
        prompts = [build_prompt(it) for it in chunk]
        # fast answers
        if batch_size > 1:
            fast_out = []
            for b in range(0, len(chunk), batch_size):
                fast_out += pair.fast.generate(prompts[b:b + batch_size], max_new_tokens=max_new_fast,
                                               thinking=pair.fast_thinking, timed=False)
        else:
            fast_out = [pair.fast.generate([p], max_new_tokens=max_new_fast,
                                           thinking=pair.fast_thinking)[0] for p in prompts] \
                if "fast" in paths else [None] * len(chunk)
        for it, prompt, fo in zip(chunk, prompts, fast_out):
            fast_gen, probe_cost = (fo[0] if fo else None), None
            if fo and hidden:
                pf = pair.fast.probe_features(fo[1], fo[2], layers)
                feats.append(pf.features)
                probe_cost = pf.cost
            slow_gen = None
            if "slow" in paths:
                slow_gen = pair.slow.generate([prompt], max_new_tokens=max_new_slow,
                                              thinking=pair.slow_thinking, sampling=pair.slow_sampling,
                                              seed=_seed(it["id"]),
                                              think_budget=max_new_slow if pair.slow_thinking else None
                                              )[0][0]
            recs.append(_record(it, prompt, fast_gen, slow_gen, probe_cost))
        run.append_shard(benchmark, split, recs, feats if hidden and feats else None)
        n_new += len(recs)
        el = time.time() - t_start
        left = (len(todo) - n_new) * el / n_new
        fa = [r["fast_correct"] for r in recs if r["fast_correct"] is not None]
        sa = [r["slow_correct"] for r in recs if r["slow_correct"] is not None]
        cut = [r for r in recs if r["slow"] and (not r["slow"]["finished"] or r["slow"].get("think_forced"))]
        log(f"  {len(done) + n_new}/{len(items)}  {el / n_new:.1f}s/item  ETA {left / 60:.0f} min"
            + (f"  fast acc {sum(fa) / len(fa):.2f}" if fa else "")
            + (f"  slow acc {sum(sa) / len(sa):.2f}" if sa else "")
            + (f"  slow hit budget {len(cut)}/{len(recs)}" if cut else ""))
        info["n_done"] = len(done) + n_new
        run.update_manifest(collections={key(benchmark, split): info})

    info["n_done"] = len(done) + n_new
    info.setdefault("stopped_at_deadline", False)
    run.update_manifest(collections={key(benchmark, split): info})
    return info
