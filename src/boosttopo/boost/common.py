"""Shared run context and helpers for both combination modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from ..graph import Limits, Subgraph
from ..policy import EditStats
from .library import Library
from .residual import bootstrap_lower, corr


@dataclass
class Ctx:
    cfg: object
    ev: object  # Evaluator
    proposer: object
    policy: object
    sampler: object
    log: object  # RunLog
    library: Library
    stats: EditStats = field(default_factory=EditStats)

    @property
    def limits(self) -> Limits:
        lp = self.cfg.get("loop") or {}
        on = bool(lp.get("enabled"))
        return Limits(self.cfg.limits.max_nodes, self.cfg.limits.max_indegree,
                      loop_ks=tuple(int(k) for k in lp.get("ks", ())) if on else (),
                      loop_stops=tuple(lp.get("stops", ())) if on else ())


@dataclass
class FitResult:
    mode: str
    ensemble: list
    history: list
    extra: dict = field(default_factory=dict)


def lower_ci(values: list[float], cfg, seed: int = 0) -> float:
    return bootstrap_lower(values, cfg.get("ci_level", 0.9), cfg.get("n_boot", 2000), seed)


def pick_candidate(cands: list[tuple[float, Subgraph, dict]], ens: dict | None, tol: float) -> tuple[float, Subgraph]:
    """Highest out-of-bag fitness; near-ties broken by the lowest correlation between the candidate's per-item
    gains and the current ensemble's per-item g (doc section 8.4)."""
    top = max(c[0] for c in cands)
    tied = [c for c in cands if c[0] >= top - tol]
    if len(tied) > 1 and ens:
        def rho(c):
            ids = sorted(i for i in c[2] if i in ens)
            return corr([c[2][i] for i in ids], [ens[i] for i in ids])
        tied.sort(key=lambda c: (rho(c), -c[0]))
        return tied[0][0], tied[0][1]
    best = max(cands, key=lambda c: c[0])
    return best[0], best[1]


def mean_or0(xs) -> float:
    xs = list(xs)
    return mean(xs) if xs else 0.0
