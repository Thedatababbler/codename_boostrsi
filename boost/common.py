"""Shared config, residual weights, acceptance test (doc §6.3–6.4)."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from workflow.grow import GrowCfg
from workflow.model import Limits, SKILL_SIGS
from workflow.edits import EDIT_SKILLS, TOOL_STEPS
from boosttopo.boost.residual import bootstrap_lower

from workflow.stats import mean_ci


@dataclass
class BoostCfg:
    rounds: int = 4
    patience: int = 2
    accept_margin: float = 0.01    # one-sided 90% lower bound of the paired heldout gain must exceed this
    lam: float | str = "auto"      # cost penalty per 1k tokens in fitness; "auto" = calibrate_lam (revision §12)
    grow: GrowCfg = field(default_factory=lambda: GrowCfg(min_gain=0.005))
    limits: Limits = field(default_factory=Limits)
    eps: float = 0.1               # residual weight 1/(eps + g)
    cap: float = 5.0               # weights capped at cap * mean
    min_mass: float = 0.10         # don't grow a branch for a label with < 10% of the weight
    check: str = "status"
    run_seed: int = 0
    fresh_after_reject: bool = True  # after a rejected round, candidate stages draw fresh samples (rep + 1)
    report_seed: int = 1000        # fresh seeds for reported heldout scores (§6.4)
    # ablations (§8.3)
    no_residual: bool = False      # uniform weights, random target label
    no_tools: bool = False         # drop simulate / crop_stats from the edit set
    ablation_seed: int = 0

    def allowed_steps(self) -> set:
        s = set(EDIT_SKILLS) | set(TOOL_STEPS)
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
    """Accept iff the one-sided 90% percentile-bootstrap lower bound (10th percentile, boosttopo's
    bootstrap_lower, 2000 resamples, seed 0) of the mean paired gain exceeds accept_margin. The reported interval
    is the two-sided 90% one."""
    lo1 = bootstrap_lower(list(diffs), 0.9, 2000, 0) if diffs else float("-inf")
    m, lo, hi = mean_ci(diffs)
    return lo1 > cfg.accept_margin, {"mean": m, "lo_one_sided": lo1, "lo": lo, "hi": hi, "n": len(diffs)}


HAND_3 = "seq[analyze, draft, validate, repair]"  # = the multi-agent hand topology analyst -> writer -> repairer


def calibrate_lam(env, train, run_seed: int = 0) -> float:
    """lam such that a 3-node workflow's cost penalty ~= 0.02 g (boosttopo.run.calibrate_lam): mean tokens of
    the hand topology, without y0's tokens, on the first 8 train items."""
    from workflow.evaluate import evaluate
    from workflow.model import parse_wf
    rs = evaluate(parse_wf(HAND_3), train[: min(8, len(train))], env, run_seed)
    c = statistics.fmean(r.cost.tokens - r.cost.y0_tokens for r in rs) if rs else 0.0
    return 0.02 * 1000.0 / c if c > 0 else 0.0


def ck_load(log, mode: str, cfg: "BoostCfg", policy, proposer) -> dict | None:
    """Resume state written by ck_save after the last finished round (None if there is none). Restores the policy's
    gain history, the proposer's tried set and lam in place."""
    ck = log.read(f"checkpoint_{mode}.json") if log is not None else None
    if not ck:
        return None
    policy.history.update({k: list(v) for k, v in ck["policy_history"].items()})
    proposer.tried |= {tuple(x) for x in ck["proposer_tried"]}
    cfg.lam = ck["lam"]
    return ck


def ck_save(log, mode: str, policy, proposer, cfg: "BoostCfg", **state) -> None:
    if log is None:
        return
    log.write(f"checkpoint_{mode}.json", {**state, "lam": cfg.lam, "policy_history": dict(policy.history),
                                          "proposer_tried": sorted(proposer.tried)})


def resolve_lam(cfg: BoostCfg, env, train) -> float:
    if cfg.lam == "auto":
        cfg.lam = calibrate_lam(env, train, cfg.run_seed)
    return float(cfg.lam)
