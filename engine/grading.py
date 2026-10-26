"""Answer extraction + grading for every answer_type. Pure Python.

grade() returns True/False, or None when the item cannot be auto-graded (free_text).
"""
import re
import string

from .data.answers import last_boxed, norm_number

_ANSWER_LINE = re.compile(r"answer\s*[:：]\s*(.+)", re.IGNORECASE)


def extract_final(text: str, answer_type: str) -> str:
    text = text or ""
    hits = _ANSWER_LINE.findall(text)
    if hits:
        cand = hits[-1].strip()
    elif answer_type == "math_expr" and last_boxed(text):
        cand = last_boxed(text)
    else:
        cand = text.strip().splitlines()[-1] if text.strip() else ""
    if answer_type == "math_expr" and "\\boxed" in cand:
        cand = last_boxed(cand) or cand
    return cand.strip().rstrip(".").strip()


def _norm_tex(s: str) -> str:
    s = str(s).strip().strip("$")
    for a, b in (("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("^\\circ", ""), ("^{\\circ}", ""),
                 ("\\%", ""), ("%", ""), ("\\$", "")):
        s = s.replace(a, b)
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s).rstrip(".")
    return s


def _squad_norm(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _equal(answer_type: str, pred: str, target: str, choices=None) -> bool:
    if not target:
        return False
    if answer_type == "numeric":
        p, t = norm_number(pred), norm_number(target)
        if not p or not t:
            return False
        try:
            return abs(float(p) - float(t)) < 1e-6
        except ValueError:
            return p == t
    if answer_type == "math_expr":
        p, t = _norm_tex(pred), _norm_tex(target)
        if p == t:
            return True
        try:
            return abs(float(p) - float(t)) < 1e-6
        except ValueError:
            return False
    if answer_type == "yes_no":
        m = re.search(r"\b(yes|no)\b", pred.lower())
        return bool(m) and m.group(1) == target.lower()
    if answer_type == "multiple_choice":
        m = re.search(r"\b([A-E])\b", pred.upper())
        if m:
            return m.group(1) == target.upper()
        # model wrote the option text instead of the letter
        for c in choices or []:
            if c["label"] == target and _squad_norm(c["text"]) == _squad_norm(pred):
                return True
        return False
    if answer_type == "text_aliases":
        np_ = _squad_norm(pred)
        return bool(np_) and any(_squad_norm(a) and _squad_norm(a) in np_ for a in [target])
    return False


def grade(item: dict, text: str) -> bool | None:
    t = item["answer_type"]
    if t == "free_text":
        return None
    pred = extract_final(text, t)
    if t == "text_aliases":
        return any(_equal(t, pred, a) for a in (item.get("aliases") or [item["gold"]]))
    return _equal(t, pred, item["gold"], item.get("choices"))


def is_lure(item: dict, text: str) -> bool | None:
    """Trap items: did the answer match the intuitive (wrong) answer?"""
    lure = (item.get("meta") or {}).get("lure")
    if not lure or item["answer_type"] == "free_text":
        return None
    return _equal(item["answer_type"], extract_final(text, item["answer_type"]), lure)
