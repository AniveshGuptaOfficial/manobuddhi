"""Fixed, seeded splits for every benchmark.

Split roles
  probe_train : fast-model runs on these produce the hidden-state probe's training data (O1)
  calib       : learn-then-test certified set, 500 items, never used for training (O3)
                GSM8K: drawn from the TEST source (first 500 test items), because the models
                have seen GSM8K train (F4: fast-model error 15% on train-split calib vs 29% on
                test), and a threshold calibrated on another distribution does not transfer.
  calib_train : GSM8K only - the original 500-item calib from the train pool, kept for the
                train/test shift analysis; not used for thresholds
  dev         : trap sets only; for choosing probe layer / K without touching test
  test        : reported numbers only

Build once with `python data/build_splits.py`; everything else calls load_split().
The manifest records counts, SHA-256 hashes and leakage checks for the paper.
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from . import sources as S
from .answers import norm_text

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed"
SPLITS = DATA_DIR / "splits"

SEED = 42
CALIB_N = 500

BENCHMARKS = {
    # name: (group, splits)
    "gsm8k": ("core", ["probe_train", "calib", "test", "calib_train"]),
    "math": ("core", ["probe_train", "calib", "test"]),
    "strategyqa": ("core", ["probe_train", "calib", "test"]),
    "commonsenseqa": ("core", ["probe_train", "calib", "test"]),
    "traps_hagendorff": ("traps", ["dev", "test"]),
    "traps_gsm_plus": ("traps", ["test"]),
    "traps_gsm_ic": ("traps", ["test"]),
    "popqa": ("rag", ["probe_train", "calib", "test"]),
    "triviaqa": ("rag", ["probe_train", "calib", "test"]),
    "nq_open": ("rag", ["probe_train", "calib", "test"]),
}


def _rng(name: str) -> random.Random:
    """Per-benchmark RNG that does not depend on build order or Python's hash seed."""
    return random.Random(int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:16], 16))


def _dedup(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for it in items:
        k = norm_text(it["question"])
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _pool_split(name, pool, test, exclude, probe_cap=None):
    """Shuffle pool -> calib (500) + probe_train (rest, optionally capped).
    Items whose question appears in ANY eval set (exclude) are dropped first,
    so calib is always exactly CALIB_N."""
    pool = _dedup(pool)
    before = len(pool)
    pool = [p for p in pool if norm_text(p["question"]) not in exclude]
    _rng(name).shuffle(pool)
    calib, probe = pool[:CALIB_N], pool[CALIB_N:]
    if probe_cap:
        probe = probe[:probe_cap]
    return {"probe_train": probe, "calib": calib, "test": test}, before - len(pool)


def _stratified_dev(name, items, per_group):
    groups = defaultdict(list)
    for it in items:
        groups[(it["meta"]["family"], it["meta"]["kind"])].append(it)
    rng, dev, test = _rng(name), [], []
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        dev += g[:per_group]
        test += g[per_group:]
    return {"dev": dev, "test": test}


def build_all() -> dict:
    report = {"seed": SEED, "calib_n": CALIB_N, "benchmarks": {}, "leakage": {}}
    out = {}

    # Phase 1: fix every eval (test/dev) set first.
    gsm_ic_items, gsm_ic_bases = S.gsm_ic(RAW, n=1000, seed=SEED)
    tests = {
        "gsm8k": S.gsm8k(RAW, "test"),
        "math": S.math500(RAW),
        "strategyqa": S.strategyqa(RAW, "test"),
        "commonsenseqa": S.commonsenseqa(RAW, "validation"),
        "nq_open": _dedup(S.nq_open(RAW, "validation")),
    }
    tq = _dedup(S.triviaqa(RAW, "validation"))
    _rng("triviaqa-test").shuffle(tq)
    tests["triviaqa"] = tq[:2000]
    pop = _dedup(S.popqa(RAW))
    _rng("popqa").shuffle(pop)
    tests["popqa"], pop_pool = pop[:3000], pop[3000:]

    out["traps_hagendorff"] = _stratified_dev("traps_hagendorff", S.hagendorff(PROCESSED), per_group=10)
    out["traps_gsm_plus"] = {"test": S.gsm_plus_distractor(RAW)}
    out["traps_gsm_ic"] = {"test": gsm_ic_items}

    exclude = {norm_text(it["question"]) for t in tests.values() for it in t}
    exclude |= {norm_text(it["question"]) for b in out.values() for items in b.values() for it in items}
    exclude |= gsm_ic_bases  # GSM-IC is built on these GSM8K train problems
    # GSM-Plus perturbs GSM8K test problems; keep their seeds out of training too
    exclude |= {norm_text(it["meta"]["seed_question"]) for it in out["traps_gsm_plus"]["test"]}

    # Phase 2: pools -> calib + probe_train, excluding anything evaluated anywhere.
    pools = {
        "gsm8k": (S.gsm8k(RAW, "train"), None),
        "math": (S.math_train(RAW), None),
        "strategyqa": (S.strategyqa(RAW, "train"), None),
        "commonsenseqa": (S.commonsenseqa(RAW, "train"), None),
        "popqa": (pop_pool, None),
        "triviaqa": (S.triviaqa(RAW, "train"), 10000),
        "nq_open": (S.nq_open(RAW, "train"), 10000),
    }
    removed = {}
    for name, (pool, cap) in pools.items():
        out[name], removed[name] = _pool_split(name, pool, tests[name], exclude, probe_cap=cap)
    report["leakage"]["removed_from_training_pools"] = removed

    # GSM8K: calibrate on the test distribution (see docstring). The first CALIB_N test items
    # (source order) become calib - the same items the F4 run r1 reported as test[:500].
    g = out["gsm8k"]
    g["calib_train"] = g["calib"]
    g["calib"], g["test"] = tests["gsm8k"][:CALIB_N], tests["gsm8k"][CALIB_N:]
    report["gsm8k_calib_source"] = "test[:500]; calib_train = former train-pool calib"

    # write + verify
    for name in BENCHMARKS:
        splits = out[name]
        bdir = SPLITS / name
        bdir.mkdir(parents=True, exist_ok=True)
        info, seen = {}, {}
        for split, items in splits.items():
            path = bdir / f"{split}.jsonl"
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for it in items:
                    f.write(json.dumps({**it, "split": split}, ensure_ascii=False) + "\n")
            info[split] = {"n": len(items), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for it in items:
                seen.setdefault(norm_text(it["question"]), set()).add(split)
        overlap = sum(1 for v in seen.values() if len(v) > 1)
        report["benchmarks"][name] = {"group": BENCHMARKS[name][0], "splits": info,
                                      "cross_split_overlap": overlap}

    train_keys = {norm_text(it["question"]) for b in out.values()
                  for s in ("probe_train", "calib", "calib_train") for it in b.get(s, [])}
    eval_keys = {norm_text(it["question"]) for b in out.values()
                 for s in ("test", "dev") for it in b.get(s, [])}
    report["leakage"]["train_or_calib_in_any_eval"] = len(train_keys & eval_keys)

    (SPLITS / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def manifest() -> dict:
    return json.loads((SPLITS / "manifest.json").read_text(encoding="utf-8"))


def load_split(benchmark: str, split: str) -> list[dict]:
    path = SPLITS / benchmark / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run python data/build_splits.py")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]
