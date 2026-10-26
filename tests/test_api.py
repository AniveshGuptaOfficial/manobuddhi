"""F0 smoke tests: app imports, core endpoints answer, settings round-trip."""


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["service"] == "manobuddhi-router"


def test_settings_roundtrip(client):
    before = client.get("/v1/settings").json()
    r = client.put("/v1/settings", json={"k_fast_samples": 5})
    assert r.status_code == 200 and r.json()["k_fast_samples"] == 5
    client.put("/v1/settings", json={"k_fast_samples": before["k_fast_samples"]})


def test_chat_returns_route(client):
    r = client.post("/v1/chat/completions", json={"query": "What is 2 + 2?"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] in {"fast", "escalate", "escalated"}
    assert 0.0 <= body["conflict_score"] <= 1.0


def test_metrics(client):
    r = client.get("/v1/metrics")
    assert r.status_code == 200
    assert "escalation_rate" in r.json()
