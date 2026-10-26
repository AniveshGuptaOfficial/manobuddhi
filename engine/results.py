"""Results folder for GPU runs (F4 onwards). Plain files, so a Kaggle run can be zipped,
downloaded and imported on the PC with scripts/import_results.py.

results/<run>/
  manifest.json                     models, devices, hardware, versions, git commit, collections
  records/<benchmark>__<split>.jsonl  one line per item: prompt, fast/slow generations, grading
  hidden/<benchmark>__<split>/*.npz   probe features per shard (ids + one array per feature)
  probe/probe.npz, probe/report.json  trained probe (numpy only to apply)
  scores/<benchmark>__<split>.jsonl   probe P(fast answer wrong) per item
  baselines.json                      three baselines + references, thresholds, sweeps

Records are appended in shards together with their hidden states, so a run stopped by
Kaggle's 12-hour limit resumes where it left off and never has records without features.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SCHEMA_VERSION = 1


def key(benchmark: str, split: str) -> str:
    return f"{benchmark}__{split}"


class RunDir:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    # ---------------------------------------------------------------- manifest
    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    def manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def update_manifest(self, **fields) -> dict:
        m = self.manifest()
        if not m:
            m = {"schema_version": SCHEMA_VERSION, "run": self.path.name,
                 "created_at": datetime.now(timezone.utc).isoformat(), "collections": {}}
        for k, v in fields.items():
            if k == "collections":
                m.setdefault("collections", {}).update(v)
            else:
                m[k] = v
        m["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        return m

    # ---------------------------------------------------------------- records
    def records_path(self, benchmark: str, split: str) -> Path:
        return self.path / "records" / f"{key(benchmark, split)}.jsonl"

    def load_records(self, benchmark: str, split: str) -> list[dict]:
        p = self.records_path(benchmark, split)
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def collections(self) -> list[tuple[str, str]]:
        d = self.path / "records"
        if not d.exists():
            return []
        return [tuple(p.stem.split("__", 1)) for p in sorted(d.glob("*.jsonl"))]

    def append_shard(self, benchmark: str, split: str, records: list[dict],
                     features: list[dict] | None = None) -> None:
        """Write hidden-state shard first, then the record lines: a crash in between leaves
        an orphan shard (ignored on load) rather than records without features."""
        if not records:
            return
        if features is not None:
            hdir = self.path / "hidden" / key(benchmark, split)
            hdir.mkdir(parents=True, exist_ok=True)
            n = len(list(hdir.glob("*.npz")))
            arrays = {"ids": np.array([r["id"] for r in records])}
            for name in features[0]:
                arrays[name] = np.stack([f[name] for f in features])
            np.savez(hdir / f"{n:05d}.npz", **arrays)
        p = self.records_path(benchmark, split)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- hidden states
    def load_features(self, benchmark: str, split: str, names: list[str] | None = None
                      ) -> tuple[list[str], dict[str, np.ndarray]]:
        """Concatenate all shards; keep only ids that have a record (drops orphan shards
        and duplicates from a resumed shard)."""
        hdir = self.path / "hidden" / key(benchmark, split)
        done = {r["id"] for r in self.load_records(benchmark, split)}
        ids, cols, seen = [], {}, set()
        for f in sorted(hdir.glob("*.npz")) if hdir.exists() else []:
            z = np.load(f)
            keep = [i for i, x in enumerate(z["ids"].tolist()) if x in done and x not in seen]
            if not keep:
                continue
            for i in keep:
                seen.add(z["ids"][i].item())
            ids += [z["ids"][i].item() for i in keep]
            for name in z.files:
                if name == "ids" or (names and name not in names):
                    continue
                cols.setdefault(name, []).append(z[name][keep])
        return ids, {k: np.concatenate(v) for k, v in cols.items()}

    # ---------------------------------------------------------------- probe scores
    def scores_path(self, benchmark: str, split: str) -> Path:
        return self.path / "scores" / f"{key(benchmark, split)}.jsonl"

    def save_scores(self, benchmark: str, split: str, scores: dict[str, float]) -> None:
        p = self.scores_path(benchmark, split)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for i, s in scores.items():
                f.write(json.dumps({"id": i, "probe_p_err": s}) + "\n")

    def load_scores(self, benchmark: str, split: str) -> dict[str, float]:
        p = self.scores_path(benchmark, split)
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as f:
            return {d["id"]: d["probe_p_err"] for d in map(json.loads, f) if d}

    # ---------------------------------------------------------------- baselines
    @property
    def baselines_path(self) -> Path:
        return self.path / "baselines.json"

    def load_baselines(self) -> dict:
        if not self.baselines_path.exists():
            return {}
        return json.loads(self.baselines_path.read_text(encoding="utf-8"))

    def save_baselines(self, data: dict) -> None:
        self.baselines_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def merge_shards(dest: RunDir, shards: list[RunDir], log=print) -> dict:
    """Combine per-GPU shard runs (results/<run>.s0, .s1, ...) into one run. Idempotent: items
    already in dest are skipped, so it can be rerun after a resumed collection."""
    first = next((sh.manifest() for sh in shards if sh.manifest()), None)
    if not first:
        raise ValueError("no shard has a manifest yet")
    models = [sh.manifest().get("models", {}).get("fast", {}).get("name") for sh in shards if sh.manifest()]
    if len(set(models)) > 1:
        raise ValueError(f"shards were collected with different models: {models}")
    counts = {}
    for sh in shards:
        for bench, split in sh.collections():
            have = {r["id"] for r in dest.load_records(bench, split)}
            recs = [r for r in sh.load_records(bench, split) if r["id"] not in have]
            if not recs:
                continue
            ids, cols = sh.load_features(bench, split)
            pos = {x: i for i, x in enumerate(ids)}
            feats = None
            if cols and all(r["id"] in pos for r in recs):
                feats = [{k: v[pos[r["id"]]] for k, v in cols.items()} for r in recs]
            dest.append_shard(bench, split, recs, feats)
            counts[key(bench, split)] = counts.get(key(bench, split), 0) + len(recs)
    colls = {}
    for bench, split in dest.collections():
        k = key(bench, split)
        parts = [sh.manifest().get("collections", {}).get(k) for sh in shards]
        parts = [p for p in parts if p]
        info = dict(parts[0]) if parts else {}
        info.update(n_target=sum(p.get("n_target", 0) for p in parts),
                    n_done=len(dest.load_records(bench, split)), shard=f"merged from {len(parts)} shards")
        colls[k] = info
    dest.update_manifest(**{k: first[k] for k in ("models", "environment", "git_commit", "probe_layers",
                                                   "generation") if k in first},
                         collections=colls,
                         parallel={"shards": [sh.path.name for sh in shards],
                                   "note": "one process per GPU, each with its own model copy; "
                                           "every item is timed on a single GPU"})
    log(f"merged into {dest.path.name}: " + ", ".join(f"{k} +{v}" for k, v in counts.items()) if counts
        else f"merged into {dest.path.name}: nothing new")
    return counts
