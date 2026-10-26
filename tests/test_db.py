"""Database checks. Skipped if PostgreSQL is not running (docker compose up -d)."""
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from sqlalchemy import select

from backend.config import PROJECT_ROOT
from backend.db import SessionLocal
from backend.db_models import EvalRun, PathTelemetry, Request
from tests.conftest import needs_db

pytestmark = needs_db


@pytest.fixture(scope="module")
def alembic_cfg():
    cfg = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    command.upgrade(cfg, "head")
    return cfg


def test_models_match_migrations(alembic_cfg):
    command.check(alembic_cfg)  # raises if db_models.py and the migrations disagree


def test_insert_and_read(alembic_cfg, clean_db):
    with SessionLocal() as s:
        run = EvalRun(benchmark="gsm8k", split="test", n_items=1, status="running", progress=0.0,
                      router_config={"mode": "small_large", "k": 3}, baseline="manobuddhi")
        req = Request(source="eval", eval_run=run, item_id="gsm8k-test-0", query="2+2?",
                      mode="small_large", fast_model="f", slow_model="s", route="escalated",
                      conflict_score=0.8, gold="4", correct=True, fast_correct=False)
        req.paths = [PathTelemetry(path="fast", model="f", completion_tokens=5, gpu_seconds=0.1),
                     PathTelemetry(path="slow", model="s", completion_tokens=50, gpu_seconds=0.9)]
        s.add(run)
        s.commit()
        got = s.scalar(select(Request).where(Request.id == req.id))
        assert got.route == "escalated" and len(got.paths) == 2
        assert sum(p.gpu_seconds for p in got.paths) == pytest.approx(1.0)
        s.delete(run)  # cascades to requests and telemetry
        s.commit()
