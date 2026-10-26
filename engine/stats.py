"""Small statistics helpers (pure Python): AUROC, percentiles."""


def percentile(values, p: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def auroc(scores, labels) -> float | None:
    """P(score of a positive > score of a negative); ties count 1/2. None if one class missing."""
    pairs = [(s, bool(y)) for s, y in zip(scores, labels) if s is not None and y is not None]
    pos = sum(1 for _, y in pairs if y)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return None
    pairs.sort(key=lambda t: t[0])
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1])
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)
