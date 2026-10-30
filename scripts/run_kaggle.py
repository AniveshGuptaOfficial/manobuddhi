"""ManoBuddhi Kaggle Pipeline Automation CLI.

Enables full command-line automation for running F4 on Kaggle GPU without
manually interacting with Jupyter notebook web UI.

Commands:
    python scripts/run_kaggle.py check
    python scripts/run_kaggle.py push --profile smoke --config B
    python scripts/run_kaggle.py status
    python scripts/run_kaggle.py fetch --run r1-smoke --auto-import
    python scripts/run_kaggle.py run --profile smoke --config B --auto-import

Prerequisites:
    1. Kaggle API credentials:
       - Either ~/.kaggle/kaggle.json (downloaded from https://www.kaggle.com/settings)
       - Or KAGGLE_USERNAME and KAGGLE_KEY set in .env
    2. GITHUB_TOKEN secret added to Kaggle Secrets (read-only access to manobuddhi repo).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Try loading .env variables
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

STAGING_DIR = PROJECT_ROOT / "artifacts" / "kaggle_kernel"
DOWNLOAD_DIR = PROJECT_ROOT / "artifacts" / "kaggle_download"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_KERNEL_TITLE = "ManoBuddhi F4 Runner"
DEFAULT_SLUG = "manobuddhi-f4"


def get_credentials() -> tuple[str | None, str | None, str | None]:
    """Find Kaggle username and key from environment, .env, or ~/.kaggle/kaggle.json."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key, "Environment variables / .env"

    kaggle_json_candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("USERPROFILE", "")) / ".kaggle" / "kaggle.json",
    ]
    for path in kaggle_json_candidates:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                u = data.get("username")
                k = data.get("key")
                if u and k:
                    return u, k, str(path)
            except Exception:
                pass

    return None, None, None


def ensure_authenticated() -> Any:
    """Authenticate with Kaggle API or provide helpful user guidance."""
    user, key, source = get_credentials()
    if not user or not key:
        print("\n" + "=" * 65)
        print("  [!] Kaggle API credentials not found!")
        print("=" * 65)
        print("To run Kaggle pipelines from the command line, provide your API token:")
        print("  1. Visit: https://www.kaggle.com/settings -> Account -> 'Create New Token'")
        print("  2. Place the downloaded 'kaggle.json' file at:")
        print(f"     {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("  OR add these lines to your .env file:")
        print("     KAGGLE_USERNAME=<your_kaggle_username>")
        print("     KAGGLE_KEY=<your_kaggle_api_key>")
        print("=" * 65 + "\n")
        sys.exit(1)

    os.environ["KAGGLE_USERNAME"] = user
    os.environ["KAGGLE_KEY"] = key

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"\n[!] Kaggle authentication failed: {e}")
        print("Please verify your KAGGLE_USERNAME and KAGGLE_KEY.")
        sys.exit(1)


def customize_notebook_settings(nb_path: Path, config: str, run_name: str, profile: str,
                                resume_from: str | None, fast_model: str | None = None,
                                slow_model: str | None = None, slow_quant: str | None = None) -> dict:
    """Load notebook and inject chosen settings into Cell 1 (the settings cell)."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Locate settings cell
    settings_cell = None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", [])
            src_str = "".join(src) if isinstance(src, list) else src
            if "CONFIG =" in src_str and "PROFILE =" in src_str:
                settings_cell = cell
                break

    if not settings_cell:
        raise ValueError("Could not find settings code cell in f4_kaggle.ipynb")

    # Replace values using regex
    code = "".join(settings_cell["source"]) if isinstance(settings_cell["source"], list) else settings_cell["source"]
    code = re.sub(r'CONFIG\s*=\s*["\'][^"\']+["\']', f'CONFIG = "{config}"', code)
    code = re.sub(r'RUN\s*=\s*["\'][^"\']+["\']', f'RUN = "{run_name}"', code)
    code = re.sub(r'PROFILE\s*=\s*["\'][^"\']+["\']', f'PROFILE = "{profile}"', code)
    code = re.sub(r'RESUME_FROM\s*=\s*[^#\n]+', f'RESUME_FROM = {repr(resume_from)}', code)

    if fast_model:
        code = re.sub(r'FAST_MODEL\s*=\s*["\'][^"\']+["\']', f'FAST_MODEL = "{fast_model}"', code)
    if slow_model and config != "A":
        code = re.sub(r'SLOW_MODEL\s*=\s*["\'][^"\']+["\']', f'SLOW_MODEL = "{slow_model}"', code)
    if slow_quant:
        code = re.sub(r'SLOW_QUANT\s*=\s*["\'][^"\']+["\']', f'SLOW_QUANT = "{slow_quant}"', code)

    # Split into lines preserving endings
    settings_cell["source"] = [line + "\n" for line in code.splitlines()]
    return nb


def prepare_kernel(config: str, run_name: str, profile: str, resume_from: str | None,
                   kernel_type: str = "notebook", title: str = DEFAULT_KERNEL_TITLE,
                   slug: str = DEFAULT_SLUG, username: str = "") -> Path:
    """Prepares staging directory with notebook/script and kernel-metadata.json."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    if kernel_type == "notebook":
        src_nb = PROJECT_ROOT / "notebooks" / "f4_kaggle.ipynb"
        if not src_nb.is_file():
            raise FileNotFoundError(f"Source notebook not found: {src_nb}")
        nb_data = customize_notebook_settings(src_nb, config, run_name, profile, resume_from)
        target_code = STAGING_DIR / "f4_kaggle.ipynb"
        with open(target_code, "w", encoding="utf-8") as f:
            json.dump(nb_data, f, indent=1)
        code_file = "f4_kaggle.ipynb"
    else:
        src_script = PROJECT_ROOT / "scripts" / "f4_kaggle.py"
        if not src_script.is_file():
            raise FileNotFoundError(f"Source script not found: {src_script}")
        target_code = STAGING_DIR / "f4_kaggle.py"
        shutil.copy2(src_script, target_code)
        code_file = "f4_kaggle.py"

    dataset_sources = []
    if resume_from:
        dataset_slug = resume_from if "/" in resume_from else f"{username}/{resume_from}"
        dataset_sources.append(dataset_slug)

    metadata = {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": kernel_type,
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": []
    }

    meta_file = STAGING_DIR / "kernel-metadata.json"
    meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return STAGING_DIR


def cmd_check(args):
    """Verify Kaggle authentication and environment setup."""
    user, key, source = get_credentials()
    print("=== Kaggle Automation Diagnostics ===")
    if user and key:
        masked_key = key[:4] + "*" * (len(key) - 8) + key[-4:] if len(key) > 8 else "***"
        print(f"  [+] Credentials: OK (Source: {source})")
        print(f"  [+] Username:    {user}")
        print(f"  [+] API Key:     {masked_key}")
        ensure_authenticated()
        print("  [+] API Connection: Verified successfully!")
    else:
        print("  [!] Credentials: NOT FOUND")
        ensure_authenticated()

    nb_file = PROJECT_ROOT / "notebooks" / "f4_kaggle.ipynb"
    py_file = PROJECT_ROOT / "scripts" / "f4_kaggle.py"
    print(f"  [+] Notebook:    {'Found' if nb_file.exists() else 'Missing'} ({nb_file})")
    print(f"  [+] Python CLI:  {'Found' if py_file.exists() else 'Missing'} ({py_file})")
    print("\nReady to run with:")
    print("  python scripts/run_kaggle.py push --profile smoke --config B")
    print("  python scripts/run_kaggle.py run --profile smoke --config B --auto-import")


def cmd_push(args):
    """Prepare and push kernel to Kaggle."""
    api = ensure_authenticated()
    user, _, _ = get_credentials()

    run_name = args.run or ("a1" if args.config == "A" else "r1")
    if args.profile == "smoke" and not run_name.endswith("-smoke"):
        run_name = run_name + "-smoke"

    print("Preparing kernel payload...")
    print(f"  Config:  {args.config}")
    print(f"  Profile: {args.profile}")
    print(f"  Run:     {run_name}")
    print(f"  Type:    {args.type}")
    if args.resume_from:
        print(f"  Resume:  {args.resume_from}")

    slug = args.slug or DEFAULT_SLUG
    prepare_kernel(
        config=args.config,
        run_name=run_name,
        profile=args.profile,
        resume_from=args.resume_from,
        kernel_type=args.type,
        title=args.title or DEFAULT_KERNEL_TITLE,
        slug=slug,
        username=user,
    )

    print(f"\nPushing kernel '{user}/{slug}' to Kaggle (GPU 2x T4 enabled)...")
    api.kernels_push(str(STAGING_DIR))
    print("[+] Successfully pushed! Kernel URL:")
    print(f"    https://www.kaggle.com/code/{user}/{slug}")
    print("\nNext steps:")
    print(f"  Check status: python scripts/run_kaggle.py status --slug {slug}")
    print(f"  Fetch output: python scripts/run_kaggle.py fetch --slug {slug} --run {run_name} --auto-import")


def cmd_status(args):
    """Check execution status of a Kaggle kernel."""
    api = ensure_authenticated()
    user, _, _ = get_credentials()
    slug = args.slug or DEFAULT_SLUG
    kernel_id = f"{user}/{slug}"

    try:
        resp = api.kernels_status(kernel_id)
        status = str(resp.status).lower() if hasattr(resp, "status") else "unknown"
        failure = getattr(resp, "failure_message", None)
        print(f"\nKernel: {kernel_id}")
        print(f"Status: {status.upper()}")
        if failure:
            print(f"Failure message: {failure}")
        return status, failure
    except Exception as e:
        print(f"[!] Error checking status for {kernel_id}: {e}")
        return "error", str(e)


def cmd_fetch(args):
    """Download and extract results from completed Kaggle kernel."""
    api = ensure_authenticated()
    user, _, _ = get_credentials()
    slug = args.slug or DEFAULT_SLUG
    kernel_id = f"{user}/{slug}"

    run_name = args.run or ("a1" if args.config == "A" else "r1")
    if args.profile == "smoke" and not run_name.endswith("-smoke"):
        run_name = run_name + "-smoke"

    print(f"Downloading output for {kernel_id}...")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    api.kernels_output(kernel_id, path=str(DOWNLOAD_DIR), force=True, quiet=False)

    # Locate zip archive
    zips = list(DOWNLOAD_DIR.glob(f"f4_{run_name}.zip")) or list(DOWNLOAD_DIR.glob("f4_*.zip"))
    if not zips:
        # Check if files were unzipped in output
        manifest = DOWNLOAD_DIR / "manifest.json"
        if manifest.exists():
            target_run_dir = RESULTS_DIR / run_name
            target_run_dir.mkdir(parents=True, exist_ok=True)
            for item in DOWNLOAD_DIR.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_run_dir / item.name)
                elif item.is_dir() and item.name != "results":
                    dest = target_run_dir / item.name
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
            print(f"[+] Direct output copied to {target_run_dir}")
        else:
            print(f"[!] No f4_<run>.zip archive found in downloaded output ({DOWNLOAD_DIR})")
            return
    else:
        archive = zips[0]
        target_run_dir = RESULTS_DIR / run_name
        target_run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {archive.name} -> {target_run_dir}...")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target_run_dir)
        print(f"[+] Extracted to: {target_run_dir}")

    # Optional Auto-Import
    if args.auto_import:
        print("\n--- Importing results into PostgreSQL database ---")
        try:
            from backend.db import SessionLocal
            from backend.importer import import_run
            with SessionLocal() as s:
                runs = import_run(s, str(RESULTS_DIR / run_name), replace=True)
            print(f"[+] Imported {len(runs)} eval runs into database! Available at http://localhost:8000")
        except Exception as e:
            print(f"[!] Import failed (is postgres running via docker compose up -d?): {e}")

    # Print baseline report
    try:
        from engine import pipeline
        from engine.results import RunDir
        rdir = RunDir(RESULTS_DIR / run_name)
        if (rdir.path / "baselines.json").exists():
            print("\n" + "=" * 60)
            print(f"  F4 Baseline Report ({run_name})")
            print("=" * 60)
            print(pipeline.report(rdir, "gsm8k"))
            print("=" * 60)
    except Exception as e:
        print(f"[!] Could not print baseline report: {e}")


def cmd_run(args):
    """End-to-end automated runner: push -> wait -> fetch -> auto-import -> report."""
    print("=== Starting Automated Kaggle F4 Execution ===")
    cmd_push(args)

    poll_interval = args.poll_interval or 30
    t0 = time.time()
    spinner = ["|", "/", "-", "\\"]
    sp_idx = 0

    print(f"\nMonitoring kernel execution (polling every {poll_interval}s)...")
    while True:
        status, failure = cmd_status(args)
        elapsed = int(time.time() - t0)
        mins, secs = divmod(elapsed, 60)

        if "complete" in status:
            print(f"\n[+] Kaggle kernel completed successfully in {mins}m {secs}s!")
            break
        elif "error" in status or "failed" in status or "cancelled" in status:
            print(f"\n[!] Kernel execution terminated with status: {status.upper()}")
            if failure:
                print(f"Failure message: {failure}")
            sys.exit(1)
        else:
            char = spinner[sp_idx % len(spinner)]
            sp_idx += 1
            print(f"\r  [{char}] Status: {status.upper()} (Elapsed: {mins:02d}m {secs:02d}s) - Press Ctrl+C to stop waiting", end="", flush=True)

        time.sleep(poll_interval)

    # Fetch and import
    print("\nFetching results...")
    cmd_fetch(args)


def cmd_list(args):
    """List recent kernels for authenticated Kaggle user."""
    api = ensure_authenticated()
    user, _, _ = get_credentials()
    print(f"Fetching kernels for {user}...")
    kernels = api.kernels_list(user=user)
    if not kernels:
        print("No kernels found.")
        return
    for k in kernels:
        print(f"  - {k.ref:<45} (status: {getattr(k, 'status', 'n/a')})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    # check
    sub.add_parser("check", help="Verify Kaggle credentials and setup")

    # push
    p_push = sub.add_parser("push", help="Configure and push kernel to Kaggle")
    p_push.add_argument("--config", default="B", choices=["A", "B"], help="Config A (Qwen3) or B (Qwen2.5 1.5B->7B)")
    p_push.add_argument("--profile", default="smoke", choices=["smoke", "checkpoint", "full"])
    p_push.add_argument("--run", default=None, help="Run name (defaults to r1 or a1)")
    p_push.add_argument("--resume-from", default=None, help="Dataset folder name to resume from")
    p_push.add_argument("--type", default="notebook", choices=["notebook", "script"], help="Push as notebook or script")
    p_push.add_argument("--title", default=DEFAULT_KERNEL_TITLE)
    p_push.add_argument("--slug", default=DEFAULT_SLUG)

    # status
    p_stat = sub.add_parser("status", help="Check Kaggle kernel execution status")
    p_stat.add_argument("--slug", default=DEFAULT_SLUG)

    # fetch
    p_fetch = sub.add_parser("fetch", help="Download output, extract results, and import into DB")
    p_fetch.add_argument("--slug", default=DEFAULT_SLUG)
    p_fetch.add_argument("--run", default=None)
    p_fetch.add_argument("--config", default="B", choices=["A", "B"])
    p_fetch.add_argument("--profile", default="smoke", choices=["smoke", "checkpoint", "full"])
    p_fetch.add_argument("--auto-import", action="store_true", default=True, help="Import into app database")

    # run (end-to-end)
    p_run = sub.add_parser("run", help="Push, watch, download, and auto-import end-to-end")
    p_run.add_argument("--config", default="B", choices=["A", "B"])
    p_run.add_argument("--profile", default="smoke", choices=["smoke", "checkpoint", "full"])
    p_run.add_argument("--run", default=None)
    p_run.add_argument("--resume-from", default=None)
    p_run.add_argument("--type", default="notebook", choices=["notebook", "script"])
    p_run.add_argument("--title", default=DEFAULT_KERNEL_TITLE)
    p_run.add_argument("--slug", default=DEFAULT_SLUG)
    p_run.add_argument("--poll-interval", type=int, default=30, help="Status polling interval in seconds")
    p_run.add_argument("--auto-import", action="store_true", default=True, help="Import into app database when done")

    # list
    sub.add_parser("list", help="List user's Kaggle kernels")

    args = ap.parse_args(argv)
    if args.cmd == "check":
        cmd_check(args)
    elif args.cmd == "push":
        cmd_push(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "fetch":
        cmd_fetch(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "list":
        cmd_list(args)


if __name__ == "__main__":
    main()
