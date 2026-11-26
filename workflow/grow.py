"""grow_fragment (doc §6.1): greedy growth of one fragment by edits, shared by both boosting modes.

`fitness(frag)` evaluates the fragment attached to the frozen base workflow; frozen parts come
from cache so only the fragment's cost is spent. `context(frag)` returns the attached workflow
for legality checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .edits import apply_edit
from .model import to_str


@dataclass
class GrowCfg:
    max_steps: int = 4
    top_k: int = 3
    min_gain: float = 0.0


@dataclass
class GrowLog:
    steps: list = field(default_factory=list)   # [{"frag", "cands": [(edit, fit)], "chosen"}]
    evals: int = 0


def grow_fragment(seed, fitness, context, proposer, policy, obs: dict, cfg: GrowCfg,
                  log: GrowLog | None = None, budget: list | None = None):
    """budget: optional single-element list [remaining evaluations], decremented in place."""
    log = log if log is not None else GrowLog()
    frag, fit = seed, fitness(seed)
    for _ in range(cfg.max_steps):
        if budget is not None and budget[0] <= 0:
            break
        cands = proposer.propose(frag, policy, obs, context)
        ranked = policy.rank(frag, cands, obs)[: cfg.top_k]
        if budget is not None:
            ranked = ranked[: max(0, budget[0])]
        if not ranked:
            break
        scored = []
        for e in ranked:
            f = fitness(apply_edit(frag, e))
            scored.append((f, e))
            policy.observe(e, f - fit)
            log.evals += 1
            if budget is not None:
                budget[0] -= 1
        best_fit, best_e = max(scored, key=lambda t: t[0])
        log.steps.append({"frag": to_str(frag), "fit": fit,
                          "cands": [(str(e), f) for f, e in scored], "chosen": str(best_e),
                          "accepted": best_fit - fit > cfg.min_gain})
        if best_fit - fit <= cfg.min_gain:
            break
        frag, fit = apply_edit(frag, best_e), best_fit
    return frag, fit
