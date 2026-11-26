"""Hand-written ranking policy f0 (doc §5.3): rank edits by the *source* of the residual.

- If illegal plans carry most of the residual weight -> repair-type edits first.
- If legal-but-low plans carry most of it -> simulate / improve-type edits first.
- Otherwise (and as tie-break) by the historical mean gain of edits of the same signature.
"""
from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict

from .edits import Fragment, edit_tags
from .model import Block, Step, iter_nodes


def signature(e) -> str:
    if isinstance(e, Fragment):
        return f"frag:{e.name}"
    names = []
    for a in e.args:
        if isinstance(a, (Step, Block)):
            names += sorted({n.name for _, n in iter_nodes(a) if isinstance(n, Step)})
        else:
            names.append(str(a))
    return f"{e.op}:{'/'.join(names)}"


class F0Policy:
    def __init__(self, majority: float = 0.5, salt: int = 0):
        self.history: dict[str, list[float]] = defaultdict(list)
        self.majority, self.salt = majority, salt

    def observe(self, e, gain: float):
        self.history[signature(e)].append(gain)

    def _hist(self, e) -> float:
        h = self.history.get(signature(e))
        return statistics.fmean(h) if h else 0.0

    def _tiebreak(self, e) -> float:
        h = hashlib.sha256(f"{self.salt}:{e}".encode()).digest()
        return int.from_bytes(h[:4], "big") / 2 ** 32

    def score(self, e, obs: dict) -> tuple:
        src = obs.get("residual_source", {})
        tot = sum(src.values()) or 1.0
        tags = edit_tags(e)
        pri = 0
        if src.get("illegal", 0) / tot > self.majority and "repair" in tags:
            pri = 1
        if src.get("legal_low", 0) / tot > self.majority and "improve" in tags:
            pri = 1
        return (pri, self._hist(e), self._tiebreak(e))

    def rank(self, frag, cands, obs: dict) -> list:
        return sorted(cands, key=lambda e: self.score(e, obs), reverse=True)


def residual_source(results, weights: dict | None = None) -> dict:
    """Sum residual weight by where it comes from: illegal vs legal_low vs legal_high plans."""
    out = {"illegal": 0.0, "legal_low": 0.0, "legal_high": 0.0}
    for r in results:
        w = weights.get(r.item_id, 1.0) if weights else max(0.0, 1.0 - r.score)
        key = "illegal" if not r.legal else ("legal_low" if r.score < 0.8 else "legal_high")
        out[key] += w
    return out
