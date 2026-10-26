"""F2 API tests (need Postgres)."""
from tests.conftest import needs_db

pytestmark = needs_db


def test_health(client):
    assert client.get("/health").json()["service"] == "manobuddhi-router"


def test_chat_openai_shape_and_logged(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "What is 2 + 2?"}]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["object"] == "chat.completion" and b["choices"][0]["message"]["content"]
    assert b["route"] in ("fast", "escalated") and 0 <= b["conflict_score"] <= 1
    assert b["usage"]["total_tokens"] > 0 and b["simulated"] is True
    log = client.get("/v1/requests").json()
    assert log[0]["id"] == b["request_id"] and log[0]["query"] == "What is 2 + 2?"


def test_chat_plain_query_and_overrides(client):
    hard = "If 3 cats catch 3 mice in 3 minutes, how many minutes do 100 cats need for 100 mice? " * 4
    low = client.post("/v1/chat/completions", json={"query": hard, "threshold_mode": "fixed",
                                                     "fixed_threshold": 0.0}).json()
    high = client.post("/v1/chat/completions", json={"query": hard, "threshold_mode": "fixed",
                                                      "fixed_threshold": 1.0}).json()
    assert low["route"] == "escalated" and high["route"] == "fast"
    assert any(p["path"] == "slow" for p in low["paths"])


def test_chat_rejects_empty_and_stream(client):
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 422
    assert client.post("/v1/chat/completions", json={"query": "hi", "stream": True}).status_code == 400


def test_settings_validate_persist_and_reload(client):
    assert client.put("/v1/settings", json={"k_fast_samples": 99}).status_code == 422
    assert client.put("/v1/settings", json={"mode": "bogus"}).status_code == 422
    r = client.put("/v1/settings", json={"k_fast_samples": 5, "threshold_mode": "fixed"})
    assert r.status_code == 200 and r.json()["k_fast_samples"] == 5

    from backend import settings
    from backend.db import SessionLocal

    settings.apply({"k_fast_samples": 3})
    with SessionLocal() as s:
        assert settings.load_latest(s)
    assert settings.snapshot()["k_fast_samples"] == 5


def test_metrics(client):
    assert client.get("/v1/metrics").json() == {"total_requests": 0}
    for q in ["hi", "What is 7 * 8?", "Prove that the square root of 2 is irrational. " * 5]:
        client.post("/v1/chat/completions", json={"query": q})
    m = client.get("/v1/metrics").json()
    assert m["total_requests"] == 3 and 0 <= m["escalation_rate"] <= 1
    assert m["p95_latency_ms"] > 0 and sum(m["latency_histogram"]["counts"]) == 3


def test_system_status(client):
    st = client.get("/v1/system").json()
    assert st["postgres"]["ok"] is True and st["router_backend"] == "placeholder"


def test_request_budget_override_is_used(client):
    b = client.post("/v1/chat/completions", json={"query": "hi", "latency_budget_ms": 700}).json()
    assert b["latency_budget_ms"] == 700
