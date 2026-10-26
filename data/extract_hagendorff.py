"""Extract Hagendorff et al. (2023) CRT-style items into data/processed/hagendorff_crt.jsonl.
Reads the list literals only (never executes the original script)."""
import ast, json
from pathlib import Path
here = Path(__file__).resolve().parent
src = (here / "raw/hagendorff_crt/w5vhp-osfstorage-archive/LLMs_responses.py").read_text(encoding="utf-8")
src = src.split("import openai")[0]
SETS = {"crt1": ("crt1", "trap"), "crt_not_hostile": ("crt1", "control"),
        "crt2": ("crt2", "trap"), "crt2_not_hostile": ("crt2", "control"),
        "crt3": ("crt3", "trap"), "crt3_not_hostile": ("crt3", "control"),
        "si": ("semantic_illusion", "trap"), "six_sanity": ("sanity", "control")}
out = here / "processed/hagendorff_crt.jsonl"; out.parent.mkdir(exist_ok=True)
n = 0
with open(out, "w", encoding="utf-8") as f:
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and node.targets[0].id in SETS:
            family, kind = SETS[node.targets[0].id]
            name = node.targets[0].id
            for it in ast.literal_eval(node.value):
                rec = {"id": f"{name}_{it['number']}", "family": family, "kind": kind,
                       "question": it["task"], "correct": str(it["correct"]),
                       "lure": str(it["intuitive"]), "scorable": True, "note": ""}
                if name == "crt_not_hostile":
                    # Original file copies the trap labels. Without the trap, answer = total - stated price.
                    ans = round(float(it["total_cost"]) - float(it["more"]), 2)
                    rec.update(correct=f"${ans}", lure="", note="label recomputed: total - stated price")
                elif name == "crt2_not_hostile":
                    rec.update(lure="")
                elif name == "crt3_not_hostile":
                    rec.update(scorable=False, lure="", note="growth rate not stated; no single correct answer")
                elif name == "si":
                    rec.update(note="free-text correct answer; grade with judge or manual check")
                elif name == "six_sanity":
                    rec.update(scorable=False, note="no gold label in source")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); n += 1
print(f"wrote {n} items -> {out}")
