"""F0 check: can the app reach PostgreSQL, Redis and the downloaded data?

Usage (from the project root, after `docker compose up -d`):
    python scripts/check_services.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import config  # noqa: E402

ok = True


def report(name, fn):
    global ok
    try:
        print(f"[ ok ] {name}: {fn()}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] {name}: {e}")


def pg():
    import psycopg
    dsn = config.database_url.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(dsn, connect_timeout=5) as c:
        return c.execute("select version()").fetchone()[0].split(",")[0]


def rd():
    import redis
    r = redis.Redis.from_url(config.redis_url, socket_connect_timeout=5)
    r.ping()
    return "redis " + r.info()["redis_version"]


def data():
    raw = config.data_dir / "raw"
    need = ["gsm8k", "math", "strategyqa", "commonsenseqa", "gsm_plus", "gsm_ic",
            "popqa", "triviaqa", "nq_open", "azure_traces", "burstgpt"]
    missing = [d for d in need if not (raw / d).is_dir()]
    if missing:
        raise RuntimeError(f"missing {missing}; run python data/download_datasets.py")
    crt = config.data_dir / "processed" / "hagendorff_crt.jsonl"
    if not crt.exists():
        raise RuntimeError("missing hagendorff_crt.jsonl; run python data/extract_hagendorff.py")
    return f"{len(need)} datasets + Hagendorff items present"


report("PostgreSQL", pg)
report("Redis", rd)
report("Datasets", data)
sys.exit(0 if ok else 1)
