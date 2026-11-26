"""'No stages' ablation (doc §8.3): same edit set and evaluation budget, but no freezing and no
rounds — greedy search over the whole workflow, each candidate evaluated end to end."""
from __future__ import annotations

import statistics

from workflow.edits import apply_edit
from workflow.evaluate import Env, evaluate
from workflow.model import make_step, seq, to_str
from workflow.policy import F0Policy, residual_source
from workflow.proposer import WorkflowProposer

from .common import BoostCfg


def fit_global(train, heldout, env: Env, cfg: BoostCfg = BoostCfg(), eval_budget: int = 12,
               policy=None, proposer=None):
    policy = policy or F0Policy(salt=cfg.run_seed)
    proposer = proposer or WorkflowProposer(cfg.limits, cfg.allowed_steps(), cfg.allowed_fragments())
    wf = seq(make_step("draft"))

    def fitness(w):
        rs = evaluate(w, train, env, cfg.run_seed)
        return (statistics.fmean(r.g for r in rs)
                - cfg.lam * statistics.fmean(r.cost.tokens for r in rs) / 1000), rs

    fit, rs = fitness(wf)
    used, steps = 0, []
    while used < eval_budget:
        obs = {"residual_source": residual_source(rs)}
        cands = policy.rank(wf, proposer.propose(wf, policy, obs), obs)[: min(cfg.grow.top_k, eval_budget - used)]
        if not cands:
            break
        scored = []
        for e in cands:
            f, r = fitness(apply_edit(wf, e))
            policy.observe(e, f - fit)
            scored.append((f, e, r))
            used += 1
        best_f, best_e, best_r = max(scored, key=lambda t: t[0])
        steps.append({"wf": to_str(wf), "fit": fit, "cands": [(str(e), f) for f, e, _ in scored]})
        if best_f - fit <= cfg.grow.min_gain:
            break
        wf, fit, rs = apply_edit(wf, best_e), best_f, best_r
    return wf, {"mode": "global", "steps": steps, "evals": used}
