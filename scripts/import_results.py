"""Import a results folder from Kaggle into the app database (real numbers, simulated: false).

  python scripts\\import_results.py results\\r1
  python scripts\\import_results.py results\\r1 --replace     (re-import after re-running baselines)

Unzip the Kaggle output into results\\ first. Needs PostgreSQL running (docker compose up -d).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.db import SessionLocal  # noqa: E402
from backend.importer import import_run  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="results/<run> folder containing manifest.json and baselines.json")
    ap.add_argument("--replace", action="store_true", help="delete this run's earlier import first")
    a = ap.parse_args(argv)
    with SessionLocal() as s:
        runs = import_run(s, a.path, replace=a.replace)
    print(f"done: {len(runs)} eval runs. Open http://localhost:8000 -> Benchmark Lab.")


if __name__ == "__main__":
    main()
