"""F4 results folder, probe training/applying, pipeline steps, telemetry fallback (CPU only)."""
import json

import numpy as np
import pytest

from engine import pipeline
from engine.probe import LinearProbe
from engine.results import RunDir
from engine.telemetry import GpuMeter
from tests.f4_fakes import records


def _features(recs, seed, signal=True):
    """d=32 features; feature 'last_L4' carries the error signal, 'prompt_L4' is noise."""
    rng = np.random.default_rng(seed)
    out = []
    for r in recs:
        y = 0.0 if r["fast_correct"] else 1.0
        last = rng.normal(0, 1, 32)
        if signal:
            last[:4] += 1.5 * y
        out.append({"last_L4": last.astype(np.float16),
                    "prompt_L4": rng.normal(0, 1, 32).astype(np.float16)})
    return out


def _fill(run, bench, split, recs, seed, shard=40):
    feats = _features(recs, seed)
    for s in range(0, len(recs), shard):
        run.append_shard(bench, split, recs[s:s + shard], feats[s:s + shard])


def test_shards_resume_and_orphans(tmp_path):
    run = RunDir(tmp_path / "r")
    recs = records(50, "probe_train")
    _fill(run, "gsm8k", "probe_train", recs, 0, shard=20)
    # an orphan shard (crash after writing features, before records) must be ignored
    run.append_shard("gsm8k", "probe_train", [], None)
    extra = records(5, "probe_train", seed=9)
    for r in extra:
        r["id"] += "-orphan"
    np.savez(run.path / "hidden" / "gsm8k__probe_train" / "99999.npz",
             ids=np.array([r["id"] for r in extra]), last_L4=np.zeros((5, 32)), prompt_L4=np.zeros((5, 32)))
    ids, f = run.load_features("gsm8k", "probe_train")
    assert ids == [r["id"] for r in recs] and f["last_L4"].shape == (50, 32)
    assert run.collections() == [("gsm8k", "probe_train")]


def test_probe_roundtrip(tmp_path):
    pytest.importorskip("sklearn")
    from engine.probe import train_probe

    recs = records(300, "probe_train")
    feats = _features(recs, 1)
    X = {k: np.stack([f[k] for f in feats]) for k in feats[0]}
    y = np.array([not r["fast_correct"] for r in recs], dtype=int)
    probe, rep = train_probe(X, y, cs=(0.1, 1.0), log=lambda *_: None)
    assert probe.feature == "last_L4" and rep["selected"]["cv_auroc"] > 0.8
    probe.save(tmp_path / "p.npz")
    p2 = LinearProbe.load(tmp_path / "p.npz")
    assert np.allclose(probe.predict(X["last_L4"]), p2.predict(X["last_L4"]))


def test_pipeline_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    run = RunDir(tmp_path / "r")
    run.update_manifest(models={"fast": {"name": "fast-1.5b"}, "slow": {"name": "slow-7b"}})
    _fill(run, "gsm8k", "probe_train", records(400, "probe_train", seed=5), 1)
    _fill(run, "gsm8k", "calib", records(200, "calib", seed=6), 2)
    _fill(run, "gsm8k", "test", records(250, "test", seed=7), 3)
    pipeline.train(run, log=lambda *_: None)
    scored = pipeline.score(run, log=lambda *_: None)
    assert scored == {"gsm8k__calib": 200, "gsm8k__test": 250}
    res = pipeline.baselines(run, log=lambda *_: None)
    assert res["signal_auroc_test"]["probe"] > 0.8
    saved = json.loads(run.baselines_path.read_text())
    assert set(saved["benchmarks"]["gsm8k"]["policies"]) >= {"always_slow", "confidence", "probe_only"}
    assert "always_slow" in pipeline.report(run)


def test_meter_without_nvml_measures_wall_only():
    m = GpuMeter()  # NVML is absent on the PC; on a GPU machine the real path runs
    with m.measure([0]) as r:
        sum(range(10000))
    assert r.wall_s > 0
    if not m.available:
        assert r.nvml is False and r.gpu_seconds == 0.0 and r.energy_j is None
    m.close()


def test_meter_with_fake_nvml():
    class H:
        def __init__(self, i):
            self.i, self.e = i, 1000.0

    class FakeNV:
        handles = [H(0), H(1)]

        def nvmlDeviceGetCount(self): return 2
        def nvmlDeviceGetHandleByIndex(self, i): return self.handles[i]

        def nvmlDeviceGetUtilizationRates(self, h):
            return type("U", (), {"gpu": 50 if h.i == 0 else 100})()

        def nvmlDeviceGetTotalEnergyConsumption(self, h):
            h.e += 500.0  # +0.5 J per read
            return h.e

    import time

    m = GpuMeter(interval_s=0.005, nvml=FakeNV())
    with m.measure([0, 1]) as r:
        time.sleep(0.1)
    m.close()
    assert r.nvml and r.gpu_seconds == pytest.approx(2 * r.wall_s)
    assert r.energy_j == pytest.approx(1.0)  # one read before and after, two GPUs
    assert 50 <= r.gpu_util_pct <= 100
    assert r.gpu_busy_seconds == pytest.approx(r.wall_s * 1.5, rel=0.2)
