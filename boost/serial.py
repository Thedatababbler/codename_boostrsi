"""Serial (GBDT-style) boosting (doc §6.2): each round appends a new stage to the root seq.

The new stage starts from the state left by all earlier (frozen) stages; earlier stages are read
from the memoised prefix states, so evaluating a candidate only pays for the new stage.
"""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor

from workflow.evaluate import Env, prefix_state, run_tail
from workflow.grow import GrowLog, grow_fragment
from workflow.model import Block, empty_seq, make_step, seq, to_str
from workflow.policy import F0Policy, residual_source
from workflow.proposer import WorkflowProposer

from .common import BoostCfg, accept, residual_weights


def attach_tail(wf: Block, frag: Block) -> Block:
    return seq(*wf.children, *frag.children)


def _pmap(env, fn, xs):
    with ThreadPoolExecutor(max_workers=max(1, env.workers)) as ex:
        return list(ex.map(fn, xs))


def tail_results(wf, frag, items, env: Env, run_seed: int):
    pre = _pmap(env, lambda i: prefix_state(wf, i, env, run_seed), items)
    offset = len(wf.children)
    new = _pmap(env, lambda p: run_tail(p[0], frag, p[1], env, offset, run_seed), list(zip(pre, items)))
    return pre, new


def fit_serial(train, heldout, env: Env, cfg: BoostCfg = BoostCfg(), policy=None, proposer=None,
               start: Block | None = None):
    policy = policy or F0Policy(salt=cfg.run_seed)
    proposer = proposer or WorkflowProposer(cfg.limits, cfg.allowed_steps(), cfg.allowed_fragments())
    wf = start or seq(make_step("draft"))                         # stage 0: draft only
    history, stall = [], 0
    budget_used = 0
    for m in range(cfg.rounds):
        base = dict(zip([i.id for i in train],
                        _pmap(env, lambda i: prefix_state(wf, i, env, cfg.run_seed), train)))
        active = [i for i in train if base[i.id].g < 1.0 and base[i.id].score < 1.0]
        if not active:
            break
        offset = len(wf.children)
        weights = residual_weights({i.id: base[i.id].g for i in active}, cfg)
        obs = {"residual_source": residual_source([base[i.id] for i in active], weights)}

        def fitness(frag, active=active, base=base, offset=offset):
            if not frag.children:
                return 0.0
            new = _pmap(env, lambda i: run_tail(base[i.id], frag, i, env, offset, cfg.run_seed), active)
            gain = statistics.fmean(r.g - base[r.item_id].g for r in new)
            return gain - cfg.lam * statistics.fmean(r.cost.tokens for r in new) / 1000

        log = GrowLog()
        frag, fit = grow_fragment(empty_seq(), fitness, lambda f: attach_tail(wf, f), proposer,
                                  policy, obs, cfg.grow, log)
        budget_used += log.evals
        rec = {"round": m, "base": to_str(wf), "frag": to_str(frag), "train_fit": fit,
               "grow": log.steps, "evals": log.evals, "obs": obs}
        if not frag.children:
            rec["accepted"] = False
            history.append(rec)
            stall += 1
            if stall >= cfg.patience:
                break
            continue
        pre, new = tail_results(wf, frag, heldout, env, cfg.run_seed)
        ok, info = accept([n.g - p.g for p, n in zip(pre, new)], cfg)
        rec.update(accepted=ok, heldout=info)
        history.append(rec)
        if ok:
            wf, stall = attach_tail(wf, frag), 0
        else:
            stall += 1
            if stall >= cfg.patience:
                break
    return wf, {"mode": "serial", "rounds": history, "evals": budget_used}
