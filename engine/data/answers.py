"""Gold-answer extraction and normalisation (pure Python, no dependencies)."""
import re

_NUM = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


def norm_text(s: str) -> str:
    """Lowercase + collapse whitespace; used for dedup and leakage checks."""
    return " ".join(str(s).lower().split())


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K answers end with '#### 72'."""
    return norm_number(answer_field.split("####")[-1])


def norm_number(s: str) -> str:
    """'$1,250.00' -> '1250', '0.50' -> '0.5', '-3' -> '-3'. Returns '' if no number."""
    s = str(s).replace(",", "").replace("$", "").strip()
    m = _NUM.search(s)
    if not m:
        return ""
    t = m.group(0)
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    if t in ("-0", ""):
        t = "0"
    return t


def last_boxed(solution: str) -> str | None:
    """Content of the last \\boxed{...} or \\fbox{...}, with nested braces handled."""
    idx = max(solution.rfind("\\boxed"), solution.rfind("\\fbox"))
    if idx < 0:
        return None
    i = solution.find("{", idx)
    if i < 0:
        # form: \boxed 5
        rest = solution[idx:].split(maxsplit=1)
        return rest[1].split("$")[0].strip() if len(rest) > 1 else None
    depth = 0
    for j in range(i, len(solution)):
        if solution[j] == "{":
            depth += 1
        elif solution[j] == "}":
            depth -= 1
            if depth == 0:
                return solution[i + 1 : j].strip()
    return None
