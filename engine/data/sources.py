"""Readers for each raw dataset in data/raw/. Every reader returns a list of items:

    {"id", "benchmark", "question", "gold", "answer_type", "aliases", "choices", "meta"}

answer_type: numeric | math_expr | yes_no | multiple_choice | text_aliases | free_text
"""
import csv
import json
import random
from pathlib import Path

from . import answers as A


def _parquet(path: Path, columns=None) -> list[dict]:
    import pandas as pd  # imported lazily so the pure-Python parts work without pandas

    return pd.read_parquet(path, columns=columns).to_dict("records")


def _aslist(x) -> list:
    if x is None:
        return []
    if isinstance(x, (str, bytes)):
        return [x]
    try:
        return [v.item() if hasattr(v, "item") else v for v in list(x)]
    except TypeError:
        return [x]


def _one(pattern_dir: Path, prefix: str) -> Path:
    hits = sorted(pattern_dir.glob(f"{prefix}*.parquet"))
    if not hits:
        raise FileNotFoundError(f"no {prefix}*.parquet in {pattern_dir}")
    return hits[0]


def _item(benchmark, id_, question, gold, answer_type, aliases=None, choices=None, **meta):
    return {
        "id": id_,
        "benchmark": benchmark,
        "question": question,
        "gold": gold,
        "answer_type": answer_type,
        "aliases": aliases or [],
        "choices": choices or [],
        "meta": meta,
    }


# ---------- core benchmarks ----------

def gsm8k(raw: Path, split: str) -> list[dict]:
    rows = _parquet(_one(raw / "gsm8k" / "main", split))
    return [_item("gsm8k", f"gsm8k-{split}-{i}", r["question"], A.gsm8k_gold(r["answer"]), "numeric")
            for i, r in enumerate(rows)]


def math_train(raw: Path) -> list[dict]:
    out = []
    for subj_dir in sorted(p for p in (raw / "math").iterdir() if p.is_dir()):
        for i, r in enumerate(_parquet(_one(subj_dir, "train"))):
            gold = A.last_boxed(r["solution"])
            if gold:
                out.append(_item("math", f"math-train-{subj_dir.name}-{i}", r["problem"], gold,
                                 "math_expr", subject=subj_dir.name, level=r["level"]))
    return out


def math500(raw: Path) -> list[dict]:
    out = []
    with open(raw / "math500" / "test.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out.append(_item("math", "math500-" + r["unique_id"].replace("/", "-").removesuffix(".json"),
                             r["problem"], r["answer"], "math_expr",
                             subject=r["subject"], level=r["level"]))
    return out


def strategyqa(raw: Path, split: str) -> list[dict]:
    rows = json.load(open(raw / "strategyqa" / f"{split}.json", encoding="utf-8"))
    return [_item("strategyqa", f"strategyqa-{r['qid']}", r["question"],
                  "yes" if r["answer"] else "no", "yes_no", term=r.get("term", ""))
            for r in rows]


def commonsenseqa(raw: Path, split: str) -> list[dict]:
    rows = _parquet(_one(raw / "commonsenseqa" / "data", split))
    out = []
    for r in rows:
        ch = r["choices"]
        choices = [{"label": lab, "text": txt} for lab, txt in zip(_aslist(ch["label"]), _aslist(ch["text"]))]
        out.append(_item("commonsenseqa", f"csqa-{r['id']}", r["question"], r["answerKey"],
                         "multiple_choice", choices=choices, concept=r.get("question_concept", "")))
    return out


# ---------- RAG (second embodiment) ----------

def popqa(raw: Path) -> list[dict]:
    csv.field_size_limit(10**8)
    out = []
    with open(raw / "popqa" / "test.tsv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            aliases = json.loads(r["possible_answers"])
            out.append(_item("popqa", f"popqa-{r['id']}", r["question"], aliases[0], "text_aliases",
                             aliases=aliases, prop=r["prop"], s_pop=int(r["s_pop"])))
    return out


def triviaqa(raw: Path, split: str) -> list[dict]:
    rows = _parquet(_one(raw / "triviaqa" / "rc.nocontext", split),
                    columns=["question", "question_id", "answer"])
    out = []
    for r in rows:
        a = r["answer"]
        out.append(_item("triviaqa", f"triviaqa-{r['question_id']}", r["question"], a["value"],
                         "text_aliases", aliases=_aslist(a["aliases"])))
    return out


def nq_open(raw: Path, split: str) -> list[dict]:
    rows = _parquet(_one(raw / "nq_open" / "nq_open", split))
    out = []
    for i, r in enumerate(rows):
        aliases = _aslist(r["answer"])
        out.append(_item("nq_open", f"nq-{split}-{i}", r["question"], aliases[0], "text_aliases",
                         aliases=aliases))
    return out


# ---------- trap / lure items ----------

def hagendorff(processed: Path) -> list[dict]:
    out = []
    with open(processed / "hagendorff_crt.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["scorable"]:
                continue
            numeric = r["family"] in ("crt1", "crt2", "crt3")
            out.append(_item("traps_hagendorff", "crt-" + r["id"], r["question"],
                             A.norm_number(r["correct"]) if numeric else r["correct"],
                             "numeric" if numeric else "free_text",
                             family=r["family"], kind=r["kind"],
                             lure=A.norm_number(r["lure"]) if (numeric and r["lure"]) else r["lure"]))
    return out


def gsm_plus_distractor(raw: Path) -> list[dict]:
    out = []
    with open(raw / "gsm_plus" / "data" / "test-00000-of-00001.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r["perturbation_type"] == "distraction insertion":
                out.append(_item("traps_gsm_plus", f"gsmplus-distract-{i}", r["question"],
                                 A.norm_number(r["answer"]), "numeric",
                                 seed_question=r["seed_question"]))
    return out


def gsm_ic(raw: Path, n: int, seed: int) -> tuple[list[dict], set[str]]:
    """Returns (sample of n items, normalised set of the GSM8K base questions used by GSM-IC)."""
    base = next((raw / "gsm_ic").glob("*/GSM-IC_2step.json")).parent
    rows = []
    for name in ("GSM-IC_2step.json", "GSM-IC_mstep.json"):
        for i, r in enumerate(json.load(open(base / name, encoding="utf-8"))):
            rows.append((name.split("_")[1].removesuffix(".json"), i, r))
    bases = {A.norm_text(r["original_question"]) for _, _, r in rows}
    random.Random(seed).shuffle(rows)
    out = [_item("traps_gsm_ic", f"gsmic-{kind}-{i}", r["new_question"], A.norm_number(r["answer"]),
                 "numeric", n_steps=r["n_steps"], role_label=r["role_label"],
                 number_label=r["number_label"], sentence_label=r["sentence_label"],
                 original_question=r["original_question"])
           for kind, i, r in rows[:n]]
    return out, bases
