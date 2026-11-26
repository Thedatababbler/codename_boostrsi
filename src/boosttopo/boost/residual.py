"""Residual weights, residual report and paired bootstrap confidence bounds (doc sections 6.4, 7.1)."""

from __future__ import annotations

import math
import random
from collections import Counter
from statistics import mean

INVALID_CATS = {"parse_error", "schema_error", "insufficient_funds", "farmland_exceeded", "missing_seeds",
                "not_mature", "missing_fruit", "bad_day", "unknown_crop", "empty"}

U_PRIME = {
    "log": lambda g, cfg: 1.0 / (cfg.get("eps", 0.1) + g),
    "threshold": lambda g, cfg: 1.0 if g < cfg.get("tau", 0.8) else 0.0,
    "exp": lambda g, cfg: cfg.get("alpha", 3.0) * math.exp(-cfg.get("alpha", 3.0) * g),
}


def residual_weights(g_ens: dict[str, float], cfg) -> dict[str, float]:
    if not g_ens:
        return {}
    if cfg.get("uniform_weights"):
        return {i: 1.0 / len(g_ens) for i in g_ens}
    r = {i: U_PRIME[cfg.get("utility", "log")](g, cfg) for i, g in g_ens.items()}
    cap = cfg.get("weight_cap", 5.0) * mean(r.values())
    r = {i: min(v, cap) for i, v in r.items()}
    z = sum(r.values())
    if z <= 0:  # threshold utility with every item above tau
        return {i: 1.0 / len(g_ens) for i in g_ens}
    return {i: v / z for i, v in r.items()}


def high_weight_ids(weights: dict[str, float], mass: float = 0.5) -> list[str]:
    """Smallest set of top-weight items holding at least ``mass`` of the total weight."""
    out, acc = [], 0.0
    for i, w in sorted(weights.items(), key=lambda t: (-t[1], t[0])):
        out.append(i)
        acc += w
        if acc >= mass:
            break
    return out


def residual_report(g: dict[str, float], weights: dict[str, float], cats: dict[str, str]) -> dict:
    """Abstract residual signals (no item ids): concentration, failure-type shares among high-weight items."""
    if not g:
        return {"mean_g": 0.0, "max_w": 0.0, "top_mass_ids": 0, "invalid_share_hi": 0.0, "cat_share_hi": {},
                "cat_share_all": {}}
    hi = high_weight_ids(weights)
    c_hi = Counter(cats.get(i, "") for i in hi)
    c_all = Counter(cats.get(i, "") for i in g)
    n_ent = -sum(w * math.log(w) for w in weights.values() if w > 0) / math.log(max(2, len(weights)))
    return {
        "mean_g": mean(g.values()),
        "max_w": max(weights.values()),
        "norm_entropy": n_ent,
        "n_hi": len(hi),
        "frac_hi": len(hi) / len(g),
        "invalid_share_hi": sum(v for k, v in c_hi.items() if k in INVALID_CATS) / len(hi),
        "invalid_share_all": sum(v for k, v in c_all.items() if k in INVALID_CATS) / len(g),
        "cat_share_hi": {k: v / len(hi) for k, v in c_hi.items()},
        "cat_share_all": {k: v / len(g) for k, v in c_all.items()},
    }


def dominant_failure(report: dict) -> str:
    sh = report.get("cat_share_hi") or {}
    return max(sh, key=sh.get) if sh else ""


def bootstrap_lower(values: list[float], level: float = 0.9, n_boot: int = 2000, seed: int = 0) -> float:
    """One-sided lower confidence bound of the mean by percentile bootstrap."""
    if not values:
        return float("-inf")
    if len(values) == 1:
        return values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int((1 - level) * n_boot)]


def paired(new: dict[str, float], old: dict[str, float]) -> list[float]:
    return [new[i] - old[i] for i in sorted(old)]


def corr(a: list[float], b: list[float]) -> float:
    if len(a) < 2:
        return 0.0
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return 0.0
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)
