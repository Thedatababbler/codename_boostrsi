"""Branch (AdaBoost-style) boosting (doc §6.3): each round adds / extends one case of the
trailing branch block, targeting the label where the residual weight is most concentrated.

Routing is the branch's own check, so at inference each item runs exactly one case.
"""
from __future__ import annotations

import random
import statistics
from concurrent.futures import ThreadPoolExecutor

from workflow.evaluate import Env, ItemResult, prefix_state, run_fragment
from workflow.grow import GrowLog, grow_fragment
from workflow.model import Block, branch, empty_seq, make_step, parse_wf, seq, to_str
from workflow.policy import F0Policy, residual_source
from workflow.proposer import WorkflowProposer
from workflow.tools import CHECKS

from .common import BoostCfg, accept, ck_load, ck_save, group_sum, residual_weights, resolve_lam


def _pmap(env, fn, xs):
    with ThreadPoolExecutor(max_workers=max(1, env.workers)) as ex:
        return list(ex.map(fn, xs))


def normalize_case(frag: Block):
    """A one-block fragment becomes the case itself (keeps nesting depth within limits)."""
    if len(frag.children) == 1 and isinstance(frag.children[0], Block):
        return frag.children[0]
    return frag


def as_seed(case) -> Block:
    return case if case.kind == "seq" else seq(case)


def attach_case(wf: Block, label: str, frag: Block) -> Block:
    b = wf.children[-1]
    cases = b.cases
    if frag.children:
        cases[label] = normalize_case(frag)
    else:
        cases.pop(label, None)
    return seq(*wf.children[:-1], branch(b.params["check"], cases))


class _View:
    """Per-item results for the current branch workflow, reusing the frozen prefix."""

    def __init__(self, wf: Block, items, env: Env, run_seed: int, rep: int = 0):
        """The prefix runs at repetition 0; cases at `rep` (fresh samples after a rejection)."""
        self.wf, self.env, self.seed, self.rep = wf, env, run_seed, rep
        self.b = len(wf.children) - 1
        self.prefix = seq(*wf.children[: self.b])
        self.check = wf.children[-1].params["check"]
        self.pre = dict(zip([i.item_id for i in items],
                            _pmap(env, lambda i: prefix_state(self.prefix, i, env, run_seed), items)))
        self.label = {i.item_id: CHECKS[self.check](self.pre[i.item_id].state, env.ctx(i)) for i in items}
        cases = wf.children[-1].cases
        self.cur = dict(zip([i.item_id for i in items], _pmap(env, lambda i: self.run_case(i, cases.get(self.label[i.item_id])), items)))

    def case_tokens(self, r: ItemResult) -> int:
        """Tokens spent inside the case (0 when the item passed through to the prefix result)."""
        return 0 if r is self.pre[r.item_id] else r.cost.tokens

    def run_case(self, item, case) -> ItemResult:
        pre = self.pre[item.item_id]
        if case is None:
            return pre
        return run_fragment(pre, case, item, self.env, (self.b, self.label[item.item_id]), self.seed, self.rep)


def fit_branch(train, heldout, env: Env, cfg: BoostCfg = BoostCfg(), policy=None, proposer=None, runlog=None):
    policy = policy or F0Policy(salt=cfg.run_seed)
    proposer = proposer or WorkflowProposer(cfg.limits, cfg.allowed_steps(), cfg.allowed_fragments())
    wf = seq(make_step("draft"), branch(cfg.check, {}))          # draft, then an empty branch
    history, stall, budget_used, rejects, m0 = [], 0, 0, 0, 0
    ck = ck_load(runlog, "branch", cfg, policy, proposer)
    if ck:
        wf, history, stall, budget_used, rejects, m0 = (parse_wf(ck["wf"]), ck["history"], ck["stall"],
                                                        ck["budget_used"], ck["rejects"], ck["next_round"])
    resolve_lam(cfg, env, train)
    for m in range(m0, cfg.rounds):
        if ck and ck.get("done"):
            break
        rep = rejects if cfg.fresh_after_reject else 0
        V = _View(wf, train, env, cfg.run_seed, rep)
        g = {k: r.g for k, r in V.cur.items()}
        w = residual_weights(g, cfg)
        mass = group_sum(w, V.label)
        eligible = {l: v for l, v in mass.items() if v >= cfg.min_mass}
        if not eligible:
            break
        if cfg.no_residual:
            target = random.Random(f"{cfg.ablation_seed}:{m}").choice(sorted(eligible))  # resumable
        else:
            target = max(sorted(eligible), key=eligible.get)
        items = [i for i in train if V.label[i.item_id] == target]
        existing = wf.children[-1].cases.get(target)
        seed = as_seed(existing) if existing is not None else empty_seq()
        obs = {"residual_source": residual_source([V.cur[i.item_id] for i in items], w)}

        def fitness(frag, V=V, items=items, w=w, target=target):
            case = normalize_case(frag) if frag.children else None
            new = _pmap(env, lambda i: V.run_case(i, case), items)
            ws = [w[i.item_id] for i in items]
            tot = sum(ws) or 1.0
            gain = sum(wi * (n.g - V.cur[n.item_id].g) for wi, n in zip(ws, new)) / tot
            dtok = statistics.fmean(V.case_tokens(n) - V.case_tokens(V.cur[n.item_id]) for n in new)
            return gain - cfg.lam * dtok / 1000

        log = GrowLog()
        frag, fit = grow_fragment(seed, fitness, lambda f: attach_case(wf, target, f), proposer,
                                  policy, obs, cfg.grow, log)
        budget_used += log.evals
        rec = {"round": m, "base": to_str(wf), "target": target, "mass": mass,
               "frag": to_str(frag), "train_fit": fit, "grow": log.steps, "evals": log.evals}
        if to_str(frag) == to_str(seed):
            rec["accepted"] = False
            history.append(rec)
            stall, rejects = stall + 1, rejects + 1
            done = stall >= cfg.patience
            ck_save(runlog, "branch", policy, proposer, cfg, wf=to_str(wf), history=history, stall=stall,
                    budget_used=budget_used, rejects=rejects, next_round=m + 1, done=done)
            if done:
                break
            continue
        cand = attach_case(wf, target, frag)
        H = _View(wf, heldout, env, cfg.run_seed, rep)
        case = normalize_case(frag) if frag.children else None
        diffs = []
        for i in heldout:
            if H.label[i.item_id] != target:
                diffs.append(0.0)       # other labels run an unchanged case
            else:
                diffs.append(H.run_case(i, case).g - H.cur[i.item_id].g)
        ok, info = accept(diffs, cfg)
        rec.update(accepted=ok, heldout=info, candidate=to_str(cand), rep=rep)
        history.append(rec)
        if ok:
            wf, stall = cand, 0
        else:
            stall, rejects = stall + 1, rejects + 1
        done = stall >= cfg.patience
        ck_save(runlog, "branch", policy, proposer, cfg, wf=to_str(wf), history=history, stall=stall,
                budget_used=budget_used, rejects=rejects, next_round=m + 1, done=done)
        if done:
            break
    return wf, {"mode": "branch", "rounds": history, "evals": budget_used, "lam": cfg.lam}
