"""
ManoBuddhi dataset downloader.

Run from the ManoBuddhi folder on your own PC (needs internet):

    pip install -r data/requirements-data.txt
    python data/download_datasets.py            # core + traps + RAG QA + traces (~1-2 GB)
    python data/download_datasets.py --list     # show every dataset key
    python data/download_datasets.py gsm8k math # only some
    python data/download_datasets.py --with-wiki    # also the 13 GB DPR Wikipedia corpus
    python data/download_datasets.py --with-lmsys   # also LMSYS-Chat-1M (gated, needs HF login)
    python data/download_datasets.py azure_2024_code azure_2024_conv   # Azure 2024 one-week traces

Everything lands in data/raw/<key>/. Re-running skips what is already there.
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

RAW = Path(__file__).resolve().parent / "raw"

# key: (kind, source, extra, group, note)
DATASETS = {
    # ---- Core benchmarks (O1, O2, O5)
    "gsm8k":          ("hf", "openai/gsm8k", None, "core", "GSM8K 7,473 train / 1,319 test"),
    "math":           ("hf", "EleutherAI/hendrycks_math", None, "core", "MATH 12.5k, one folder per subject"),
    "math500":        ("hf", "HuggingFaceH4/MATH-500", None, "core", "MATH-500 quick subset"),
    "strategyqa":     ("hf", "ChilleD/StrategyQA", None, "core", "StrategyQA labeled train/test"),
    "strategyqa_orig":("gh", "eladsegal/strategyqa", None, "core", "Original StrategyQA repo (cite)"),
    "commonsenseqa":  ("hf", "tau/commonsense_qa", None, "core", "CommonsenseQA; report on validation"),
    # ---- Trap / lure items (O1, G1)
    "gsm_plus":       ("hf", "qintongli/GSM-Plus", None, "traps", "GSM-Plus perturbations incl. distractors"),
    "gsm_ic":         ("gh", "google-research-datasets/gsm-ic", None, "traps", "GSM-IC irrelevant context"),
    # ---- RAG retrieval gate (second embodiment)
    "popqa":          ("hf", "akariasai/PopQA", None, "rag", "PopQA 14k"),
    "triviaqa":       ("hf", "mandarjoshi/trivia_qa", ["rc.nocontext/*", "*.md"], "rag", "TriviaQA rc.nocontext only (small)"),
    "nq_open":        ("hf", "google-research-datasets/nq_open", None, "rag", "NQ-Open"),
    # ---- Load traces (O3, G3)
    "azure_traces":   ("gh", "Azure/AzurePublicDataset", None, "traces", "Azure public traces (use AzureLLMInference*)"),
    "burstgpt":       ("gh", "HPMLL/BurstGPT", None, "traces", "BurstGPT real-world bursty workload"),
    "azure_2024_code":("url", "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv", None, "azure24", "Azure 2024 code trace, 1 week"),
    "azure_2024_conv":("url", "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_conv_1week.csv", None, "azure24", "Azure 2024 conversation trace, 1 week"),
    # ---- Optional / large
    "dpr_wiki":       ("url", "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz", None, "wiki", "DPR Wikipedia 21M passages (~13 GB)"),
    "lmsys":          ("hf", "lmsys/lmsys-chat-1m", None, "lmsys", "LMSYS-Chat-1M (gated: accept licence + `huggingface-cli login`)"),
}
DEFAULT_GROUPS = {"core", "traps", "rag", "traces"}


def done(dest: Path) -> bool:
    return dest.exists() and any(dest.iterdir())


def get_hf(repo, dest, patterns):
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest), allow_patterns=patterns)


def get_gh(repo, dest):
    for branch in ("main", "master"):
        url = f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"
        r = requests.get(url, timeout=120)
        if r.status_code == 200:
            dest.mkdir(parents=True, exist_ok=True)
            zipfile.ZipFile(io.BytesIO(r.content)).extractall(dest)
            return
    raise RuntimeError(f"could not download {repo} (tried main and master)")


def get_url(url, dest):
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / url.rsplit("/", 1)[-1]
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got/1e9:.2f} / {total/1e9:.2f} GB", end="", flush=True)
    tmp.rename(out)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="dataset keys (default: all core/traps/rag/traces)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--with-wiki", action="store_true")
    ap.add_argument("--with-lmsys", action="store_true")
    ap.add_argument("--with-azure2024", action="store_true")
    a = ap.parse_args()

    if a.list:
        for k, (_, src, _, g, note) in DATASETS.items():
            print(f"{k:16} [{g:6}] {note}\n{'':16} {src}")
        return

    groups = set(DEFAULT_GROUPS)
    if a.with_wiki: groups.add("wiki")
    if a.with_lmsys: groups.add("lmsys")
    if a.with_azure2024: groups.add("azure24")
    keys = a.keys or [k for k, v in DATASETS.items() if v[3] in groups]

    failed = []
    for k in keys:
        if k not in DATASETS:
            print(f"unknown key: {k}"); failed.append(k); continue
        kind, src, extra, _, note = DATASETS[k]
        dest = RAW / k
        if done(dest):
            print(f"[skip] {k} already in {dest}"); continue
        print(f"[get ] {k}: {note}")
        try:
            if kind == "hf": get_hf(src, dest, extra)
            elif kind == "gh": get_gh(src, dest)
            else: get_url(src, dest)
            print(f"[ ok ] {k} -> {dest}")
        except Exception as e:
            print(f"[FAIL] {k}: {e}"); failed.append(k)

    print("\nManual step: Hagendorff et al. CRT-style items -> open "
          "https://doi.org/10.1038/s43588-023-00527-x , follow 'Data availability', "
          "save files into data/raw/hagendorff_crt/")
    if failed:
        print("Failed:", ", ".join(failed)); sys.exit(1)


if __name__ == "__main__":
    main()
