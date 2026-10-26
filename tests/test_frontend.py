"""F3: the UI is served at / and only calls endpoints that exist."""
import re
from pathlib import Path

from backend.main import app

HTML = (Path(__file__).resolve().parent.parent / "frontend" / "index.html").read_text(encoding="utf-8")


def test_index_served():
    from fastapi.testclient import TestClient

    r = TestClient(app).get("/")
    assert r.status_code == 200 and "ManoBuddhi" in r.text


def test_frontend_calls_only_real_endpoints():
    routes = {r.path for r in app.routes if hasattr(r, "path")}
    used = set(re.findall(r"""['"`](/v1/[a-z/]+)""", HTML))
    used = {u.rstrip("/") for u in used}
    known = {re.sub(r"/\{[^}]+\}.*", "", p) for p in routes}
    missing = {u for u in used if u not in known}
    assert not missing, f"frontend calls endpoints the API does not have: {missing}"
