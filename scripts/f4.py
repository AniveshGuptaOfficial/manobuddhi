"""F4 command line: real models, telemetry, GSM8K baselines.

GPU steps (Kaggle):
  python scripts/f4.py collect --run r1 --benchmark gsm8k --split probe_train --n 2000 --paths fast --batch-size 16
  python scripts/f4.py collect --run r1 --benchmark gsm8k --split calib --n 300
  python scripts/f4.py collect --run r1 --benchmark gsm8k --split test  --n 500
  python scripts/f4.py collect --run r1 --benchmark traps_hagendorff --split test --paths fast

CPU steps (anywhere):
  python scripts/f4.py train-probe --run r1
  python scripts/f4.py score --run r1
  python scripts/f4.py baselines --run r1
  python scripts/f4.py report --run r1

Collection resumes automatically: rerun the same command after a timeout.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import pipeline  # noqa: E402
from engine.results import RESULTS_DIR, RunDir  # noqa: E402


def _mem(s: str | None):
    if not s:
        return None
    return {int(k): v for k, v in (p.split(":") for p in s.split(","))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS_DIR))
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="run real models on a split (GPU)")
    c.add_argument("--run", required=True)
    c.add_argument("--benchmark", default="gsm8k")
    c.add_argument("--split", default="test")
    c.add_argument("--n", type=int)
    c.add_argument("--paths", default="fast,slow")
    c.add_argument("--no-hidden", action="store_true")
    c.add_argument("--batch-size", type=int, default=1)
    c.add_argument("--mode", default="small_large", choices=["small_large", "single_model_nothink_think"])
    c.add_argument("--fast-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    c.add_argument("--slow-model", default="Qwen/Qwen2.5-7B-Instruct")
    c.add_argument("--fast-device", default="cuda:0")
    c.add_argument("--slow-quant", default="none", choices=["none", "8bit", "4bit"])
    c.add_argument("--slow-device", default="cuda:1", help="used with --slow-quant")
    c.add_argument("--slow-max-memory", default="0:4GiB,1:12GiB",
                   help="float16 split across GPUs, e.g. 0:4GiB,1:12GiB")
    c.add_argument("--max-new-fast", type=int, default=512)
    c.add_argument("--max-new-slow", type=int, default=768)

    t = sub.add_parser("train-probe", help="train the hidden-state probe on probe_train (CPU)")
    t.add_argument("--run", required=True)
    t.add_argument("--benchmark", default="gsm8k")
    t.add_argument("--split", default="probe_train")

    s = sub.add_parser("score", help="apply the probe to all collected sets (CPU)")
    s.add_argument("--run", required=True)

    b = sub.add_parser("baselines", help="compute the three baselines (CPU)")
    b.add_argument("--run", required=True)
    b.add_argument("--benchmark", default="gsm8k")
    b.add_argument("--calib", default="calib")
    b.add_argument("--test", default="test")
    b.add_argument("--tol", type=float, default=0.01)
    b.add_argument("--conf", default="conf_seq", choices=["conf_seq", "conf_answer"])

    r = sub.add_parser("report", help="print the baseline table")
    r.add_argument("--run", required=True)
    r.add_argument("--benchmark", default="gsm8k")

    a = ap.parse_args(argv)
    run = RunDir(Path(a.results) / a.run)
    if a.cmd == "collect":
        from engine.collect import collect, load_pair

        pair = load_pair(a.mode, fast_model=a.fast_model, slow_model=a.slow_model,
                         fast_device=a.fast_device, slow_quant=a.slow_quant, slow_device=a.slow_device,
                         slow_max_memory=_mem(a.slow_max_memory),
                         with_slow="slow" in a.paths.split(","))
        paths = tuple(a.paths.split(","))
        collect(run, pair, a.benchmark, a.split, n=a.n, paths=paths, hidden=not a.no_hidden,
                batch_size=a.batch_size, max_new_fast=a.max_new_fast, max_new_slow=a.max_new_slow)
    elif a.cmd == "train-probe":
        pipeline.train(run, a.benchmark, a.split)
    elif a.cmd == "score":
        pipeline.score(run)
    elif a.cmd == "baselines":
        pipeline.baselines(run, a.benchmark, a.calib, a.test, tol=a.tol, conf_key=a.conf)
    elif a.cmd == "report":
        print(pipeline.report(run, a.benchmark))


if __name__ == "__main__":
    main()
