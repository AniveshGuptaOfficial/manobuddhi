"""F4 baselines from paired runs: cost accounting, calib-only thresholds, summaries."""
import math

import pytest

from engine import baselines as B
from tests.f4_fakes import probe_scores, records


@pytest.fixture(scope="module")
def data():
    calib, test = records(300, "calib", seed=1), records(400, "test", seed=2)
    return calib, test, probe_scores(calib, 3), probe_scores(test, 4)


def test_cost_accounting_is_sum_of_measured_paths():
    r = records(1)[0]
    slow = B.item_outcome(r, "always_slow", True)
    assert slow["paths"] == ["slow"] and slow["gpu_seconds"] == pytest.approx(5.4)
    casc = B.item_outcome(r, "confidence", True)
    assert casc["paths"] == ["fast", "slow"]
    assert casc["gpu_seconds"] == pytest.approx(0.6 + 5.4)
    assert casc["latency_ms"] == pytest.approx(600 + 4000)
    assert casc["ttft_ms"] == pytest.approx(600 + 4000 * 0.05)  # final answer starts after fast path
    assert casc["tokens"] == (80 + 60) + (80 + 180)
    probe = B.item_outcome(r, "probe_only", False)
    assert probe["paths"] == ["fast", "probe"] and probe["gpu_seconds"] == pytest.approx(0.64)
    assert probe["tokens"] == 140  # probe pass generates nothing


def test_threshold_uses_calib_and_meets_tolerance(data):
    calib, _, pc, _ = data
    for policy, pr in (("confidence", None), ("probe_only", pc)):
        t = B.choose_threshold(calib, policy, pr, tol=0.01)
        assert t["accuracy"] >= t["target_accuracy"] - 1e-9
        assert 0 < t["escalation_rate"] < 1
        assert t["n_calib"] == 300
    loose = B.choose_threshold(calib, "confidence", tol=0.2)
    tight = B.choose_threshold(calib, "confidence", tol=0.0)
    assert loose["escalation_rate"] <= tight["escalation_rate"]


def test_run_baselines(data):
    calib, test, pc, pt = data
    res = B.run_baselines(calib, test, probe_calib=pc, probe_test=pt)
    P = res["policies"]
    assert set(P) == set(B.POLICIES)
    assert P["always_slow"]["escalation_rate"] == 1.0 and P["always_fast"]["escalation_rate"] == 0.0
    assert P["always_slow"]["gpu_seconds_per_query"] == pytest.approx(5.4)
    for p in ("confidence", "probe_only"):
        assert P["always_fast"]["gpu_seconds_per_query"] < P[p]["gpu_seconds_per_query"]
        assert P[p]["gpu_seconds_saving_ci95"][0] <= P[p]["gpu_seconds_saving_vs_always_slow"] \
            <= P[p]["gpu_seconds_saving_ci95"][1]
    # a better error signal routes better: same accuracy target, fewer escalations
    assert res["signal_auroc_test"]["probe"] > res["signal_auroc_test"]["logprob_seq"]
    assert P["probe_only"]["escalation_rate"] < P["confidence"]["escalation_rate"]
    assert P["oracle"]["reference_only"] and not P["confidence"]["reference_only"]
    assert len(res["escalated_ids"]["always_slow"]) == len(test)
    assert res["sweeps_test"]["confidence"][-1]["escalation_rate"] == 0.0
    j = B.jsonable(res)
    assert all(not (isinstance(v, float) and math.isinf(v)) for v in j["thresholds"]["confidence"].values())


def test_without_probe_and_missing_slow(data):
    calib, test, _, _ = data
    res = B.run_baselines(calib, test)
    assert "probe_only" not in res["policies"]
    broken = [dict(test[0], slow=None)] + test[1:]
    with pytest.raises(ValueError):
        B.run_baselines(calib, broken)


def test_missing_signal_escalates():
    r = records(1)[0]
    r["fast"]["conf_seq"] = None
    assert B.escalate(r, "confidence", 0.5) is True
    assert B.escalate(r, "probe_only", 0.5, probe={}) is True
