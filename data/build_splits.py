"""Build the fixed probe_train / calib / dev / test splits (seed 42).

    python data/build_splits.py

Writes data/splits/<benchmark>/<split>.jsonl and data/splits/manifest.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.data.splits import build_all  # noqa: E402

r = build_all()
print(f"{'benchmark':18} {'probe_train':>11} {'calib':>6} {'dev':>5} {'test':>6} overlap")
for name, b in r["benchmarks"].items():
    s = b["splits"]
    g = lambda k: str(s[k]["n"]) if k in s else "-"  # noqa: E731
    print(f"{name:18} {g('probe_train'):>11} {g('calib'):>6} {g('dev'):>5} {g('test'):>6} "
          f"{b['cross_split_overlap']}")
print("\nleakage:", r["leakage"])
bad = r["leakage"]["train_or_calib_in_any_eval"] or any(
    b["cross_split_overlap"] for b in r["benchmarks"].values())
print("RESULT:", "FAIL - leakage found" if bad else "OK - no leakage")
sys.exit(1 if bad else 0)
