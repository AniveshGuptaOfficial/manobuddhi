"""Test setup. DB tests run against a separate database, manobuddhi_test, which is
created and migrated automatically and emptied before each test. Your real data in
the `manobuddhi` database is never touched. DB tests are skipped if Postgres is down."""
import pytest

from backend.config import config

BASE_URL = config.database_url
TEST_URL = BASE_URL.rsplit("/", 1)[0] + "/manobuddhi_test"
config.database_url = TEST_URL  # must happen before backend.db is imported
config.eval_executor = "inline"

TABLES = ["path_telemetry", "requests", "eval_runs", "settings_history"]


def _prepare_test_db() -> bool:
    try:
        import psycopg

        admin = BASE_URL.replace("postgresql+psycopg://", "postgresql://").rsplit("/", 1)[0] + "/postgres"
        with psycopg.connect(admin, autocommit=True, connect_timeout=3) as c:
            if not c.execute("select 1 from pg_database where datname='manobuddhi_test'").fetchone():
                c.execute("create database manobuddhi_test")
        from alembic import command
        from alembic.config import Config as AlembicConfig

        from backend.config import PROJECT_ROOT

        cfg = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
        command.upgrade(cfg, "head")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\n[conftest] Postgres not available, DB tests will be skipped: {e}")
        return False


DB_UP = _prepare_test_db()
needs_db = pytest.mark.skipif(not DB_UP, reason="PostgreSQL not reachable (docker compose up -d)")


@pytest.fixture()
def clean_db():
    from sqlalchemy import text

    from backend.db import engine

    with engine.begin() as c:
        c.execute(text(f"truncate {', '.join(TABLES)} restart identity cascade"))
    yield


@pytest.fixture()
def client(clean_db):
    from fastapi.testclient import TestClient

    from backend import settings
    from backend.main import app

    before = settings.snapshot()
    with TestClient(app) as c:
        yield c
    settings.apply(before)
