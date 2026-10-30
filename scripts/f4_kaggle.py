"""ManoBuddhi F4: real models, GPU telemetry, GSM8K baselines.

Standalone executable script for Kaggle 2x T4 or local GPU runner.
Replaces manual interaction with f4_kaggle.ipynb.

Usage (Kaggle or remote GPU):
    python scripts/f4_kaggle.py --config B --profile smoke
    python scripts/f4_kaggle.py --config B --profile checkpoint
    python scripts/f4_kaggle.py --config A --profile checkpoint

Usage (Local run without git clone / Kaggle secrets):
    python scripts/f4_kaggle.py --local --config B --profile smoke
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROFILES = {
    "smoke": dict(probe_train=64, calib=20, test=20, traps=20),
    "checkpoint": dict(probe_train=2000, calib=300, test=500, traps=240),
    "full": dict(probe_train=4000, calib=500, test=1319, traps=240),
}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="B", choices=["A", "B"],
                    help="'B': Qwen2.5-1.5B -> 7B (two models) | 'A': Qwen3-4B no-think -> think (one model)")
    ap.add_argument("--run", default=None, help="results/<run> name; defaults to r1 (Config B) or a1 (Config A)")
    ap.add_argument("--profile", default="smoke", choices=["smoke", "checkpoint", "full"])
    ap.add_argument("--resume-from", default=None,
                    help="Name of earlier output dataset folder under /kaggle/input (e.g. manobuddhi-f4-r1-output)")
    ap.add_argument("--fast-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--slow-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--slow-quant", default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--slow-max-memory", default="0:4GiB,1:12GiB")
    ap.add_argument("--no-traps", action="store_true", help="Skip Hagendorff trap items")
    ap.add_argument("--local", action="store_true",
                    help="Run locally in current repo instead of cloning to /tmp/manobuddhi with Kaggle secrets")
    ap.add_argument("--out-dir", default=None, help="Output directory for results (default /kaggle/working/results or results/)")
    return ap.parse_args(argv)


def _mem(s: str | None):
    if not s:
        return None
    return {int(k): v for k, v in (p.split(":") for p in s.split(","))}


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    config = args.config
    profile = args.profile
    run_name = args.run or ("a1" if config == "A" else "r1")
    if profile == "smoke" and not run_name.endswith("-smoke"):
        run_name = run_name + "-smoke"

    mode = "single_model_nothink_think" if config == "A" else "small_large"
    fast_model = args.fast_model
    slow_model = "Qwen/Qwen3-4B" if config == "A" else args.slow_model
    slow_quant = args.slow_quant
    slow_max_memory = _mem(args.slow_max_memory)
    run_traps = not args.no_traps
    parallel = (config == "A")

    if config == "A":
        max_new_fast, max_new_slow = 768, 3072
    else:
        max_new_fast, max_new_slow = 512, 768

    n_split = PROFILES[profile]
    print("=== ManoBuddhi F4 Runner ===")
    print(f"Run: {run_name} | Config: {config} | Mode: {mode} | Profile: {profile}")
    print(f"Models: Fast={fast_model}, Slow={slow_model} (quant={slow_quant})")
    print(f"Items per split: {n_split}")

    # Determine environment
    is_kaggle = Path("/kaggle").exists() and not args.local

    if is_kaggle:
        repo_dir = Path("/tmp/manobuddhi")
        out_dir = Path(args.out_dir or "/kaggle/working/results")
        # Check GPU
        subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv"], check=False)

        # Clone repo if not exists
        if not repo_dir.exists():
            try:
                from kaggle_secrets import UserSecretsClient
                tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
            except Exception:
                tok = os.environ.get("GITHUB_TOKEN")

            if not tok:
                raise RuntimeError("GITHUB_TOKEN secret not found. Add GITHUB_TOKEN to Kaggle Secrets.")

            url = "github.com/AniveshGuptaOfficial/manobuddhi.git"
            p = subprocess.run(["git", "clone", "--depth", "1", f"https://x-access-token:{tok}@{url}", str(repo_dir)],
                               capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError("git clone failed: " + p.stderr.split("@")[-1][-300:])
            subprocess.run(["git", "-C", str(repo_dir), "remote", "set-url", "origin", "https://" + url], check=True)

        sys.path.insert(0, str(repo_dir))
        os.chdir(str(repo_dir))
    else:
        repo_dir = Path(__file__).resolve().parents[1]
        out_dir = Path(args.out_dir or (repo_dir / "results"))
        sys.path.insert(0, str(repo_dir))
        os.chdir(str(repo_dir))

    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / run_name

    # Handle resume
    if args.resume_from:
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.exists():
            found = [p for p in kaggle_input.rglob(run_name + "*")
                     if p.is_dir() and p.parent.name == "results" and p.name in (run_name, run_name + ".s0", run_name + ".s1")]
            for src in sorted(found):
                dst = out_dir / src.name
                if not dst.exists():
                    shutil.copytree(src, dst)
                    print(f"Resumed from {src}")
            if not found:
                raise RuntimeError(f"RESUME_FROM is set ({args.resume_from}) but no matching results/{run_name}* under /kaggle/input")

    from engine import pipeline
    from engine.results import RunDir
    run = RunDir(run_path)
    print(f"Results dir: {run.path}")
    print(f"Existing collections: {run.manifest().get('collections', {})}")

    if not parallel:
        # Config B: In-process collection
        from engine.collect import collect, load_pair

        print("Loading models...")
        pair = load_pair(mode, fast_model=fast_model, slow_model=slow_model,
                         slow_quant=slow_quant, slow_max_memory=slow_max_memory)
        print("Model description:", json.dumps(pair.describe(), indent=1)[:1000])

        # 1. Probe train
        print(f"\n--- [1/4] Collecting probe_train (n={n_split['probe_train']}) ---")
        collect(run, pair, "gsm8k", "probe_train", n=n_split["probe_train"],
                paths=("fast",), batch_size=16, max_new_fast=max_new_fast)

        # 2. Calib
        print(f"\n--- [2/4] Collecting calib (n={n_split['calib']}) ---")
        collect(run, pair, "gsm8k", "calib", n=n_split["calib"],
                max_new_fast=max_new_fast, max_new_slow=max_new_slow)

        # 3. Test
        print(f"\n--- [3/4] Collecting test (n={n_split['test']}) ---")
        collect(run, pair, "gsm8k", "test", n=n_split["test"],
                max_new_fast=max_new_fast, max_new_slow=max_new_slow)

        # 4. Traps (optional)
        if run_traps:
            print(f"\n--- [4/4] Collecting traps (n={n_split['traps']}) ---")
            collect(run, pair, "traps_hagendorff", "test", n=n_split["traps"],
                    paths=("fast",), max_new_fast=max_new_fast)
    else:
        # Config A: 2-GPU parallel collection
        print("\n--- Running Config A parallel collection across 2 GPUs ---")
        jobs = [
            f"gsm8k:probe_train:{n_split['probe_train']}:fast:16",
            f"gsm8k:calib:{n_split['calib']}:fast,slow:1",
            f"gsm8k:test:{n_split['test']}:fast,slow:1"
        ]
        if run_traps:
            jobs.append(f"traps_hagendorff:test:{n_split['traps']}:fast:1")

        procs, logs = [], []
        deadline = t0 + 11 * 3600  # 11 hours guard for Kaggle 12-hour session limit
        for i in range(2):
            log_file = out_dir / f"{run_name}.s{i}.log"
            logs.append(log_file)
            cmd = [
                sys.executable, "-u", "scripts/f4.py", "--results", str(out_dir),
                "collect", "--run", f"{run_name}.s{i}",
                "--mode", mode, "--fast-model", fast_model, "--slow-model", slow_model,
                "--fast-device", f"cuda:{i}", "--max-new-fast", str(max_new_fast),
                "--max-new-slow", str(max_new_slow), "--shard", f"{i}/2",
                "--deadline", str(deadline)
            ]
            for j in jobs:
                cmd += ["--job", j]
            procs.append(subprocess.Popen(cmd, stdout=open(log_file, "a"), stderr=subprocess.STDOUT, cwd=str(repo_dir)))

        while any(p.poll() is None for p in procs):
            time.sleep(120)
            for i, log_file in enumerate(logs):
                if log_file.exists():
                    lines = [line for line in log_file.read_text(errors="ignore").replace("\r", "\n").splitlines()
                             if line.strip().startswith(("[", "warm")) or ("/" in line and "s/item" in line)]
                    last = lines[-1].strip()[:150] if lines else "(loading model)"
                    print(f"{time.strftime('%H:%M')} GPU{i}: {last}")

        for i, p in enumerate(procs):
            if p.returncode != 0:
                print(logs[i].read_text(errors="ignore")[-3000:])
                raise RuntimeError(f"GPU{i} process failed (see log above)")

        from engine.results import merge_shards
        merge_shards(run, [RunDir(out_dir / f"{run_name}.s{i}") for i in range(2)])

    # 5. Probe training, scoring, and baselines (CPU)
    print("\n--- [5/5] Training probe, scoring, and computing baselines ---")
    pipeline.train(run)
    pipeline.score(run)
    pipeline.baselines(run)
    print("\nBaseline summary:")
    print(pipeline.report(run, "gsm8k"))

    # Archive output if on Kaggle
    if is_kaggle:
        zip_base = Path("/kaggle/working") / f"f4_{run_name}"
        archive = shutil.make_archive(str(zip_base), "zip", out_dir, run_name)
        print(f"\nPackaged results archive: {archive} ({round(os.path.getsize(archive) / 2**20, 1)} MB)")

    print(f"\nF4 pipeline completed in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
