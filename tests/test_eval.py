"""F2 eval pipeline tests: create run -> route each item -> grade -> summary -> CSV."""
import pytest

from engine.data.splits import SPLITS
from tests.conftest import needs_db

pytestmark = [needs_db, pytest.mark.skipif(not (SPLITS / "manifest.json").exists(),
                                           reason="run python data/build_splits.py first")]


def _run(client, **body):
    r = client.post("/v1/eval", json=body)
    assert r.status_code == 200, r.text
    run = client.get(f"/v1/eval/{r.json()['id']}").json()  # inline executor: already finished
    assert run["status"] == "done", run["summary"]
    return run


def test_benchmarks_listed(client):
    names = {b["name"] for b in client.get("/v1/benchmarks").json()}
    assert {"gsm8k", "math", "traps_hagendorff", "popqa"} <= names


def test_eval_gsm8k_end_to_end(client):
    run = _run(client, benchmark="gsm8k", split="test", n=40)
    s = run["summary"]
    assert run["n_items"] == 40 and s["total_requests"] == 40 and s["simulated"] is True
    assert 0 <= s["accuracy"] <= 1 and s["auroc_conflict"] is not None
    assert run["router_config"]["k_fast_samples"] >= 1
    csv = client.get(f"/v1/eval/{run['id']}/results.csv").text.strip().splitlines()
    assert len(csv) == 41 and csv[0].startswith("item_id,route")


def test_baselines(client):
    slow = _run(client, benchmark="gsm8k", split="test", n=15, baseline="always_slow")["summary"]
    fast = _run(client, benchmark="gsm8k", split="test", n=15, baseline="always_fast")["summary"]
    assert slow["escalation_rate"] == 1.0 and fast["escalation_rate"] == 0.0
    assert slow["gpu_seconds_used"] > fast["gpu_seconds_used"]


def test_trap_items_report_lure_rate(client):
    s = _run(client, benchmark="traps_hagendorff", split="dev", n=30)["summary"]
    assert "fast_lure_rate" in s


@pytest.mark.parametrize("bench", ["math", "strategyqa", "commonsenseqa", "popqa"])
def test_every_answer_type_grades(client, bench):
    s = _run(client, benchmark=bench, split="test", n=10)["summary"]
    assert s["accuracy"] is not None


def test_eval_validation(client):
    assert client.post("/v1/eval", json={"benchmark": "nope"}).status_code == 422
    assert client.post("/v1/eval", json={"benchmark": "gsm8k", "split": "dev"}).status_code == 422
    assert client.get("/v1/eval/999999").status_code == 404
