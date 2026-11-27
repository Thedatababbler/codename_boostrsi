"""Serial (GBDT-style) boosting (doc §6.2): each round appends a new stage to the root seq.

The new stage starts from the state left by all earlier (frozen) stages; earlier stages are read
from the memoised prefix states, so evaluating a candidate only pays for the new stage.
"""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor

from workflow.evaluate import Env, prefix_state, run_tail
from workflow.grow import GrowLog, grow_fragment
from workflow.model import Block, empty_seq, make_step, parse_wf, seq, to_str
from workflow.policy import F0Policy, residual_source
from workflow.proposer import WorkflowProposer

from .common import BoostCfg, accept, ck_load, ck_save, residual_weights, resolve_lam


def attach_tail(wf: Block, frag: Block) -> Block:
    return seq(*wf.children, *frag.children)


def _pmap(env, fn, xs):
    with ThreadPoolExecutor(max_workers=max(1, env.workers)) as ex:
        return list(ex.map(fn, xs))


def tail_results(wf, frag, items, env: Env, run_seed: int, rep: int = 0):
    """Frozen prefix at repetition 0; the new stage at repetition `rep` (fresh samples after a rejection)."""
    pre = _pmap(env, lambda i: prefix_state(wf, i, env, run_seed), items)
    offset = len(wf.children)
    new = _pmap(env, lambda p: run_tail(p[0], frag, p[1], env, offset, run_seed, rep), list(zip(pre, items)))
    return pre, new


def fit_serial(train, heldout, env: Env, cfg: BoostCfg = BoostCfg(), policy=None, proposer=None,
               start: Block | None = None, runlog=None):
    """runlog: a boosttopo RunLog; when given, a checkpoint is written after every round and a rerun resumes
    after the last finished round (the LLM cache makes the replayed work free)."""
    policy = policy or F0Policy(salt=cfg.run_seed)
    proposer = proposer or WorkflowProposer(cfg.limits, cfg.allowed_steps(), cfg.allowed_fragments())
    wf = start or seq(make_step("draft"))                         # stage 0: draft only
    history, stall, budget_used, rejects, m0 = [], 0, 0, 0, 0
    ck = ck_load(runlog, "serial", cfg, policy, proposer)
    if ck:
        wf, history, stall, budget_used, rejects, m0 = (parse_wf(ck["wf"]), ck["history"], ck["stall"],
                                                        ck["budget_used"], ck["rejects"], ck["next_round"])
    resolve_lam(cfg, env, train)
    for m in range(m0, cfg.rounds):
        if ck and ck.get("done"):
            break
        rep = rejects if cfg.fresh_after_reject else 0
        base = dict(zip([i.item_id for i in train],
                        _pmap(env, lambda i: prefix_state(wf, i, env, cfg.run_seed), train)))
        active = [i for i in train if base[i.item_id].g < 1.0]
        if not active:
            break
        offset = len(wf.children)
        weights = residual_weights({i.item_id: base[i.item_id].g for i in active}, cfg)
        obs = {"residual_source": residual_source([base[i.item_id] for i in active], weights)}

        def fitness(frag, active=active, base=base, offset=offset, rep=rep):
            if not frag.children:
                return 0.0
            new = _pmap(env, lambda i: run_tail(base[i.item_id], frag, i, env, offset, cfg.run_seed, rep), active)
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
            ok = False
        else:
            pre, new = tail_results(wf, frag, heldout, env, cfg.run_seed, rep)
            ok, info = accept([n.g - p.g for p, n in zip(pre, new)], cfg)
            rec.update(accepted=ok, heldout=info, rep=rep)
        history.append(rec)
        if ok:
            wf, stall = attach_tail(wf, frag), 0
        else:
            stall, rejects = stall + 1, rejects + 1
        done = stall >= cfg.patience
        ck_save(runlog, "serial", policy, proposer, cfg, wf=to_str(wf), history=history, stall=stall,
                budget_used=budget_used, rejects=rejects, next_round=m + 1, done=done)
        if done:
            break
    return wf, {"mode": "serial", "rounds": history, "evals": budget_used, "lam": cfg.lam}
