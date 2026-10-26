"""Real model calls with Transformers (F4). Needs torch + transformers (requirements-ml.txt).

One HFModel wraps one loaded checkpoint. It gives, per call:
  - the answer text (greedy decoding, so runs are repeatable)
  - prompt / completion token counts
  - TTFT and latency, GPU-seconds and energy from engine.telemetry.GpuMeter
  - log-prob confidence: geometric-mean token probability over the whole completion
    (conf_seq) and over the tokens after the last "Answer:" (conf_answer), plus the
    entropy of the first generated token (the fallback signal in the risk table)
  - optionally, hidden states for the probe (O1), taken in one extra forward pass over
    prompt + answer AFTER the answer exists. That pass is timed separately as the
    "probe" path, so probe-based routers pay for it.

T4 GPUs have no bfloat16, so everything runs in float16. Config A (Qwen3, one model)
uses `thinking=False/True`; Config B uses two HFModel instances.

This module is only imported on the GPU machine; nothing in the API imports it.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,
                          LogitsProcessorList, StoppingCriteria, StoppingCriteriaList)
from transformers.generation.streamers import BaseStreamer

from .telemetry import GpuMeter

_ANSWER_RE = re.compile(r"answer\s*[:：]", re.IGNORECASE)
_THINK_END = "</think>"
# Qwen3's documented way to end thinking early: the model then answers from what it has so far
FORCE_CLOSE = ("\n\nConsidering the limited time by the user, I have to give the solution based on "
               "the thinking directly now.\n</think>\n\n")


@dataclass
class Generation:
    model: str
    text: str  # visible answer (thinking removed)
    raw_text: str
    prompt_tokens: int
    completion_tokens: int  # generated tokens, including the end-of-turn token
    finished: bool  # False = stopped at max_new_tokens
    ttft_ms: float | None = None
    latency_ms: float | None = None
    gpu_seconds: float | None = None
    gpu_busy_seconds: float | None = None
    gpu_util_pct: float | None = None
    gpu_mem_mb: float | None = None
    energy_j: float | None = None
    devices: list[int] = field(default_factory=list)
    timed: bool = False  # True only for batch-1 runs; batched runs carry no per-item timing
    mean_logprob: float | None = None
    conf_seq: float | None = None
    conf_answer: float | None = None
    first_token_entropy: float | None = None
    thinking: bool | None = None
    reasoning_tokens: int | None = None  # tokens before </think> (Config A think mode)
    sampled: bool = False  # True: sampled decoding (no log-prob confidence)
    think_forced: bool = False  # thinking hit its budget; </think> was inserted, then the answer generated

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProbeFeatures:
    features: dict  # name -> np.float16 vector, e.g. "last_L20", "mean_L20", "prompt_L20"
    cost: dict  # latency_ms, gpu_seconds, gpu_busy_seconds, energy_j, devices


class _StepLogprobs(LogitsProcessor):
    """Records, per decoding step, the log-prob of the greedy token (= max log-prob) and
    the entropy of the first step. Leaves the scores unchanged. Greedy decoding only."""

    def __init__(self):
        self.max_lp: list[torch.Tensor] = []
        self.first_entropy: torch.Tensor | None = None

    def __call__(self, input_ids, scores):
        lp = torch.log_softmax(scores.float(), dim=-1)
        self.max_lp.append(lp.max(dim=-1).values)
        if self.first_entropy is None:
            self.first_entropy = -(lp.exp() * lp).nan_to_num().sum(dim=-1)
        return scores


class _ThinkBudget(StoppingCriteria):
    """Batch-1 thinking budget: stop when thinking reaches `think_budget` tokens without closing,
    or when the answer after </think> reaches `answer_budget` tokens."""

    def __init__(self, end_id: int, plen: int, think_budget: int, answer_budget: int):
        self.end_id, self.plen = end_id, plen
        self.think_budget, self.answer_budget = think_budget, answer_budget
        self.closed_at = None

    def __call__(self, input_ids, scores, **kwargs):
        n = input_ids.shape[1] - self.plen
        if self.closed_at is None and int(input_ids[0, -1]) == self.end_id:
            self.closed_at = n
        stop = (n >= self.think_budget if self.closed_at is None
                else n - self.closed_at >= self.answer_budget)
        return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)


def _add(m, m2):
    """Fold a second measured call into the first (forced-close continuation)."""
    w = m.wall_s + m2.wall_s
    if m.gpu_util_pct is not None and m2.gpu_util_pct is not None and w > 0:
        m.gpu_util_pct = (m.gpu_util_pct * m.wall_s + m2.gpu_util_pct * m2.wall_s) / w
    m.wall_s = w
    m.gpu_seconds += m2.gpu_seconds
    m.gpu_busy_seconds += m2.gpu_busy_seconds
    if m.energy_j is not None and m2.energy_j is not None:
        m.energy_j += m2.energy_j
    if m2.gpu_mem_mb is not None:
        m.gpu_mem_mb = max(m.gpu_mem_mb or 0, m2.gpu_mem_mb)


class _FirstTokenTimer(BaseStreamer):
    """generate() calls put() once with the prompt, then once per new token."""

    def __init__(self):
        import time

        self._now = time.perf_counter
        self.calls = 0
        self.t_first = None

    def put(self, value):
        self.calls += 1
        if self.calls == 2 and self.t_first is None:
            self.t_first = self._now()

    def end(self):
        pass


def default_layers(n_layers: int, k: int = 7) -> list[int]:
    """k layers spread over the second half-ish of the network, always including the last."""
    if n_layers <= k:
        return list(range(1, n_layers + 1))
    step = n_layers / k
    return sorted({max(1, round(n_layers - i * step)) for i in range(k)})


def split_thinking(raw: str) -> tuple[str, bool]:
    """Return (visible text, thinking_closed). Text before </think> is reasoning."""
    if _THINK_END in raw:
        return raw.split(_THINK_END)[-1].strip(), True
    if "<think>" in raw:  # thinking never finished: no visible answer
        return "", False
    return raw.strip(), True


class HFModel:
    def __init__(self, name: str, *, device: str = "cuda:0", device_map: str | None = None,
                 max_memory: dict | None = None, quant: str = "none",
                 meter: GpuMeter | None = None, dtype=torch.float16):
        self.name = name
        self.meter = meter or GpuMeter()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        kw: dict = {"torch_dtype": dtype}
        if quant in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype)
                if quant == "4bit" else BitsAndBytesConfig(load_in_8bit=True))
        if device_map:
            kw["device_map"] = device_map
            if max_memory:
                kw["max_memory"] = max_memory
        elif quant != "none":
            kw["device_map"] = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(name, **kw)
        if "device_map" not in kw:
            self.model.to(device)
        self.model.eval()
        self.quant = quant
        self.input_device = self.model.get_input_embeddings().weight.device
        self.devices = self._devices()
        self.n_layers = self.model.config.num_hidden_layers
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        self.eos_ids = sorted({e for e in eos + [self.tok.eos_token_id] if e is not None})

    def _devices(self) -> list[int]:
        dm = getattr(self.model, "hf_device_map", None)
        devs = set()
        if dm:
            for v in dm.values():
                if isinstance(v, int):
                    devs.add(v)
                elif isinstance(v, str) and v.startswith("cuda"):
                    devs.add(int(v.split(":")[1]) if ":" in v else 0)
                elif isinstance(v, torch.device) and v.type == "cuda":
                    devs.add(v.index or 0)
        elif self.input_device.type == "cuda":
            devs.add(self.input_device.index or 0)
        return sorted(devs)

    def describe(self) -> dict:
        return {"name": self.name, "dtype": str(self.model.dtype).replace("torch.", ""),
                "quant": self.quant, "devices": self.devices, "n_layers": self.n_layers,
                "hidden_size": self.model.config.hidden_size,
                "device_map": {k: str(v) for k, v in (getattr(self.model, "hf_device_map", None)
                                                      or {}).items()} or None}

    # ---------------------------------------------------------------- prompts
    def chat_text(self, prompt: str, thinking: bool | None = None) -> str:
        msgs = [{"role": "user", "content": prompt}]
        kw = {} if thinking is None else {"enable_thinking": thinking}
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    # ---------------------------------------------------------------- generation
    @torch.inference_mode()
    def generate(self, prompts: list[str], *, max_new_tokens: int = 512,
                 thinking: bool | None = None, timed: bool = True, sampling: dict | None = None,
                 seed: int | None = None, think_budget: int | None = None,
                 answer_budget: int = 512) -> list[tuple[Generation, list[int], list[int]]]:
        """Greedy generation unless `sampling` is given (e.g. Qwen3 thinking mode, where greedy
        decoding can repeat endlessly). Returns, per prompt, (Generation, prompt_ids,
        completion_ids). timed=True needs a single prompt: batch 1 is what a live query costs.
        Sampled runs carry no log-prob confidence (only the greedy fast path needs it).

        think_budget (thinking mode, batch 1): thinking may use at most this many tokens; if it
        has not closed by then, '</think>' is inserted and the model writes its answer (up to
        answer_budget tokens). Both calls are measured and counted. Without this, a long
        reasoning chain that already reached the right answer scores as wrong (F4 smoke)."""
        if timed and len(prompts) != 1:
            raise ValueError("timed generation runs one prompt at a time (batch 1)")
        texts = [self.chat_text(p, thinking) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.input_device) for k, v in enc.items()}
        rec = _StepLogprobs()
        timer = _FirstTokenTimer() if timed else None
        if sampling:
            if seed is not None:
                torch.manual_seed(seed)
            dec = dict(do_sample=True, temperature=sampling.get("temperature", 0.6),
                       top_p=sampling.get("top_p", 0.95), top_k=sampling.get("top_k", 20))
            procs = LogitsProcessorList()
        else:
            dec = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
            procs = LogitsProcessorList([rec])
        plen = enc["input_ids"].shape[1]
        budget = bool(thinking and think_budget and len(prompts) == 1)
        stops = None
        if budget:
            end_id = self.tok.convert_tokens_to_ids(_THINK_END)
            crit = _ThinkBudget(end_id, plen, think_budget, answer_budget)
            stops = StoppingCriteriaList([crit])
            max_new_tokens = think_budget + answer_budget
        with self.meter.measure(self.devices) as m:
            out = self.model.generate(
                **enc, max_new_tokens=max_new_tokens, repetition_penalty=1.0, num_beams=1,
                pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos_ids,
                logits_processor=procs, streamer=timer, stopping_criteria=stops, **dec)
        forced = False
        if budget and crit.closed_at is None and int(out[0, -1]) not in self.eos_ids:
            close = self.tok(FORCE_CLOSE, return_tensors="pt",
                             add_special_tokens=False)["input_ids"].to(out.device)
            ids2 = torch.cat([out, close], dim=1)
            with self.meter.measure(self.devices) as m2:
                out = self.model.generate(
                    input_ids=ids2, attention_mask=torch.ones_like(ids2),
                    max_new_tokens=answer_budget, repetition_penalty=1.0, num_beams=1,
                    pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos_ids,
                    logits_processor=LogitsProcessorList(), **dec)
            _add(m, m2)
            forced = True
        t0_wall = m.wall_s
        gen = out[:, plen:].tolist()
        steps = torch.stack(rec.max_lp, dim=0).float().cpu() if rec.max_lp else None  # (T, B)
        ent = rec.first_entropy.float().cpu() if rec.first_entropy is not None else None

        results = []
        for b, ids in enumerate(gen):
            mask = enc["attention_mask"][b].tolist()
            prompt_ids = [t for t, keep in zip(enc["input_ids"][b].tolist(), mask) if keep]
            n_gen, finished = len(ids), False
            for i, t in enumerate(ids):
                if t in self.eos_ids:
                    n_gen, finished = i + 1, True
                    break
            comp = ids[:n_gen - 1] if finished else ids[:n_gen]
            raw = self.tok.decode(comp, skip_special_tokens=True)
            visible, _ = split_thinking(raw)
            g = Generation(model=self.name, text=visible, raw_text=raw, prompt_tokens=len(prompt_ids),
                           completion_tokens=n_gen, finished=finished, devices=self.devices,
                           thinking=thinking, timed=timed, sampled=bool(sampling),
                           think_forced=forced)
            if steps is not None and comp:
                lps = steps[: len(comp), b].tolist()
                g.mean_logprob = sum(lps) / len(lps)
                g.conf_seq = math.exp(g.mean_logprob)
                g.conf_answer = self._answer_conf(comp, lps)
            if ent is not None:
                g.first_token_entropy = float(ent[b])
            if thinking:
                think_ids = self.tok.convert_tokens_to_ids(_THINK_END)
                g.reasoning_tokens = comp.index(think_ids) if think_ids in comp else len(comp)
            if timed:
                g.latency_ms = t0_wall * 1000
                if timer.t_first is not None:
                    g.ttft_ms = (timer.t_first - m.t_start) * 1000
                g.gpu_seconds, g.gpu_busy_seconds = m.gpu_seconds, m.gpu_busy_seconds
                g.gpu_util_pct, g.gpu_mem_mb, g.energy_j = m.gpu_util_pct, m.gpu_mem_mb, m.energy_j
            results.append((g, prompt_ids, comp))
        return results

    def _answer_conf(self, comp_ids: list[int], lps: list[float]) -> float | None:
        pieces = [self.tok.decode([t], skip_special_tokens=True) for t in comp_ids]
        text = "".join(pieces)
        hits = list(_ANSWER_RE.finditer(text))
        if not hits:
            return None
        start, pos, span = hits[-1].end(), 0, []
        for piece, lp in zip(pieces, lps):
            if pos >= start and piece.strip():
                span.append(lp)
            pos += len(piece)
        return math.exp(sum(span) / len(span)) if span else None

    # ---------------------------------------------------------------- probe features
    @torch.inference_mode()
    def probe_features(self, prompt_ids: list[int], comp_ids: list[int],
                       layers: list[int]) -> ProbeFeatures:
        """Hidden states after the answer exists: last answer token ('last'), mean over the
        answer ('mean'), and last prompt token ('prompt', for the pre-answer ablation)."""
        import numpy as np

        ids = torch.tensor([prompt_ids + comp_ids], device=self.input_device)
        with self.meter.measure(self.devices) as m:
            hs = self.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states
        p = len(prompt_ids)
        feats = {}
        for layer in layers:
            h = hs[layer][0]
            feats[f"prompt_L{layer}"] = h[p - 1]
            feats[f"last_L{layer}"] = h[-1]
            feats[f"mean_L{layer}"] = h[p:].mean(0) if comp_ids else h[p - 1]
        feats = {k: v.float().cpu().numpy().astype(np.float16) for k, v in feats.items()}
        cost = {"latency_ms": m.wall_s * 1000, "gpu_seconds": m.gpu_seconds,
                "gpu_busy_seconds": m.gpu_busy_seconds, "gpu_util_pct": m.gpu_util_pct,
                "energy_j": m.energy_j, "devices": self.devices, "tokens": len(prompt_ids) + len(comp_ids)}
        return ProbeFeatures(feats, cost)
