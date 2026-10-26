# ManoBuddhi — Dual-Process LLM Inference Router

*Manas* (fast, intuitive mind) + *Buddhi* (deliberate intellect). A small fast model
answers first; a conflict score decides when to escalate to slow reasoning.

| Gap / Objective | What it covers |
|---|---|
| G1 / O1 | Conflict score: hidden-state probe + disagreement across K fast samples |
| G2 / O2 | Measured hardware cost: GPU-seconds, tokens, p95 latency |
| G3 / O3 | Budget-driven threshold inside a learn-then-test certified set |
| G4 / O4 | Fast-path reuse on escalation: KV prefix (Config A), speculative draft (Config B) |
| G5 / O5 | One model no-think -> think vs small -> large |

> **Keep this repository private until the provisional patent is filed (week 11).**

## Layout

```
backend/   FastAPI gateway: API, persistence, metrics, eval jobs
engine/    Router interface, placeholder router, grading, data splits; real router from F4
frontend/  Web UI
data/      download_datasets.py, extract_hagendorff.py; raw/ is git-ignored
scripts/   check_services.py
migrations/ Alembic database migrations
tests/     pytest suite
docker-compose.yml   PostgreSQL + Redis + RQ worker (localhost only)
```

## Setup (Windows cmd)

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
docker compose up -d
python scripts\check_services.py
pytest
uvicorn backend.main:app --reload --port 8000
```

UI: http://localhost:8000 · API docs: http://localhost:8000/docs

## API (F2)

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible chat; response adds route, conflict score, paths |
| GET | `/v1/metrics` | Dashboard metrics from the DB (`?source=playground` or `?eval_run_id=`) |
| GET | `/v1/requests` | Recent routed requests |
| GET | `/v1/benchmarks` | Benchmarks and split sizes |
| POST | `/v1/eval` | Start a benchmark run: `{benchmark, split, n, baseline}` |
| GET | `/v1/eval`, `/v1/eval/{id}` | Run list / status + summary (accuracy, AUROC, GPU-s, p95) |
| GET | `/v1/eval/{id}/results.csv` | Per-item results for the paper |
| GET/PUT | `/v1/settings` | Router settings (validated, saved, reloaded on restart) |
| GET | `/v1/system` | Postgres / Redis / worker status |

Until F5 the router is a **placeholder**: every response has `"simulated": true`.
None of its numbers are research results.

Benchmark jobs run in the Docker `worker` service (RQ needs Linux). After changing
backend or engine code: `docker compose restart worker`. To run jobs inside the API
process instead, set `MB_EVAL_EXECUTOR=inline` in `.env`.

Tests use a separate `manobuddhi_test` database, created automatically.

## Data

```cmd
python data\download_datasets.py
python data\extract_hagendorff.py
python data\build_splits.py
alembic upgrade head
```

Splits (seed 42) live in `data/splits/<benchmark>/<split>.jsonl`; `data/splits/manifest.json`
records counts, SHA-256 hashes and leakage checks. Load them with:

```python
from engine.data import load_split
items = load_split("gsm8k", "test")
```

| Split | Used for |
|---|---|
| probe_train | Fast-model runs -> hidden-state probe training data (O1) |
| calib | Learn-then-test certified set, exactly 500 items (O3) |
| dev | Trap sets only: choose probe layer and K without touching test |
| test | Reported numbers only |

## Build phases

| Phase | Weeks | GPU | Status |
|---|---|---|---|
| F0 Foundations | 1 | No | Done |
| F1 Data layer (splits, DB schema) | 1-2 | No | Done |
| F2 API gateway + placeholder router | 2-3 | No | Done |
| F3 Frontend wired to API | 3-4 | No | Next |
| F4 Model serving + telemetry + baselines | 3-5 | Yes | — |
| F5 Real router engine | 4-9 | Yes | — |
| Patent gate (provisional filing) | 10-11 | — | — |
| F6 Threshold controller + full evaluation | 12-16 | Yes | — |
| F7 Hardening + SUS user test | 16-19 | Light | — |
| F8 Deployment | 20-22 | Yes | — |
