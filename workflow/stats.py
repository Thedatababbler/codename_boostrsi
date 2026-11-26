"""Paired statistics: per-item paired differences and 90% bootstrap intervals."""
from __future__ import annotations

import random
import statistics


def mean_ci(xs, level: float = 0.90, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    xs = list(xs)
    if not xs:
        return 0.0, 0.0, 0.0
    m = statistics.fmean(xs)
    if len(xs) == 1 or len(set(xs)) == 1:
        return m, m, m
    rng = random.Random(seed)
    n = len(xs)
    boots = sorted(statistics.fmean(rng.choices(xs, k=n)) for _ in range(n_boot))
    a = (1 - level) / 2
    return m, boots[int(a * n_boot)], boots[min(n_boot - 1, int((1 - a) * n_boot))]


def paired_diffs(a: dict, b: dict) -> list[float]:
    """a, b: item_id -> value. Differences a - b over shared items, in sorted id order."""
    return [a[k] - b[k] for k in sorted(set(a) & set(b))]


def lower_ci(diffs, level: float = 0.90) -> float:
    return mean_ci(diffs, level)[1]
