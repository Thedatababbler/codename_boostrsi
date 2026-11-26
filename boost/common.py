"""Shared config, residual weights, acceptance test (doc §6.3–6.4)."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from workflow.grow import GrowCfg
from workflow.model import Limits, SKILL_SIGS
from workflow.edits import TOOL_STEPS
from workflow.stats import mean_ci


@dataclass
class BoostCfg:
    rounds: int = 4
    patience: int = 2
    accept_margin: float = 0.0     # lower 90% CI of paired heldout gain must exceed this
    lam: float = 0.002             # cost penalty per 1k tokens in fitness
    grow: GrowCfg = field(default_factory=GrowCfg)
    limits: Limits = field(default_factory=Limits)
    eps: float = 0.1               # residual weight 1/(eps + g)
    cap: float = 5.0               # weights capped at cap * mean
    min_mass: float = 0.10         # don't grow a branch for a label with < 10% of the weight
    check: str = "status"
    run_seed: int = 0
    report_seed: int = 1000        # fresh seeds for reported heldout scores (§6.4)
    # ablations (§8.3)
    no_residual: bool = False      # uniform weights, random target label
    no_tools: bool = False         # drop simulate / crop_stats from the edit set
    ablation_seed: int = 0

    def allowed_steps(self) -> set:
        s = set(SKILL_SIGS) | set(TOOL_STEPS)
        return s - {"simulate", "crop_stats"} if self.no_tools else s

    def allowed_fragments(self) -> set | None:
        if not self.no_tools:
            return None
        return {"repair_loop", "multi_draft"}


def residual_weights(g: dict, cfg: BoostCfg) -> dict:
    """1/(eps + g) (g clipped at 0), capped at cap*mean, normalised to sum 1."""
    if cfg.no_residual:
        return {k: 1.0 / len(g) for k in g} if g else {}
    w = {k: 1.0 / (cfg.eps + max(0.0, v)) for k, v in g.items()}
    if not w:
        return w
    m = statistics.fmean(w.values())
    w = {k: min(v, cfg.cap * m) for k, v in w.items()}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def group_sum(w: dict, label: dict) -> dict:
    out: dict = {}
    for k, v in w.items():
        out[label[k]] = out.get(label[k], 0.0) + v
    return out


def accept(diffs: list, cfg: BoostCfg) -> tuple[bool, dict]:
    m, lo, hi = mean_ci(diffs)
    return lo > cfg.accept_margin, {"mean": m, "lo": lo, "hi": hi, "n": len(diffs)}
