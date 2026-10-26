"""Importing a GPU results folder into the database (needs PostgreSQL)."""
import pytest

from engine import pipeline
from engine.results import RunDir
from tests.conftest import needs_db
from tests.f4_fakes import probe_scores, records

pytestmark = needs_db


@pytest.fixture()
def results_dir(tmp_path):
    run = RunDir(tmp_path / "kaggle-r1")
    run.update_manifest(models={"mode": "small_large", "fast": {"name": "Qwen/Qwen2.5-1.5B-Instruct"},
                                "slow": {"name": "Qwen/Qwen2.5-7B-Instruct"}},
                        git_commit="abc123", environment={"hardware": {"gpus": [{"name": "Tesla T4"}] * 2}})
    calib, test = records(120, "calib", seed=1), records(80, "test", seed=2)
    run.append_shard("gsm8k", "calib", calib)
    run.append_shard("gsm8k", "test", test)
    run.save_scores("gsm8k", "calib", probe_scores(calib, 3))
    run.save_scores("gsm8k", "test", probe_scores(test, 4))
    run.update_manifest(probe={"feature": "last_L20", "trained_on": "gsm8k__probe_train"})
    pipeline.baselines(run, log=lambda *_: None)
    return run


def test_import_creates_real_runs(client, results_dir):
    from backend.db import SessionLocal
    from backend.importer import import_run

    with SessionLocal() as s:
        runs = import_run(s, results_dir.path, log=lambda *_: None)
        assert {r.baseline for r in runs} == {"always_fast", "always_slow", "confidence", "probe_only", "oracle"}
        again = import_run(s, results_dir.path, log=lambda *_: None)  # idempotent
        assert len(again) == len(runs)

    listed = client.get("/v1/eval").json()
    assert len(listed) == 5
    expected = results_dir.load_baselines()["benchmarks"]["gsm8k"]["policies"]
    for r in listed:
        s = r["summary"]
        assert s["simulated"] is False and s["router_backend"] == "hf-transformers"
        assert r["n_items"] == 80 and s["total_requests"] == 80
        # the app's own metrics agree with the paper numbers from baselines.json
        assert s["accuracy"] == pytest.approx(expected[r["baseline"]]["accuracy"])
        assert s["escalation_rate"] == pytest.approx(expected[r["baseline"]]["escalation_rate"])
        assert s["gpu_seconds_used"] / 80 == pytest.approx(expected[r["baseline"]]["gpu_seconds_per_query"])
    slow = next(r for r in listed if r["baseline"] == "always_slow")
    csv = client.get(f"/v1/eval/{slow['id']}/results.csv").text.splitlines()
    assert len(csv) == 81

    with SessionLocal() as s:
        from sqlalchemy import select

        from backend.db_models import PathTelemetry, Request
        probe_run = next(r for r in runs if r.baseline == "probe_only")
        paths = s.scalars(select(PathTelemetry.path).join(Request)
                          .where(Request.eval_run_id == probe_run.id)).all()
        assert "probe" in paths and all(p in ("fast", "probe", "slow") for p in paths)
        assert s.scalar(select(PathTelemetry.energy_j).where(PathTelemetry.energy_j.is_not(None))) is not None

    with SessionLocal() as s:
        runs2 = import_run(s, results_dir.path, replace=True, log=lambda *_: None)
        assert len(runs2) == 5
    assert len(client.get("/v1/eval").json()) == 5
