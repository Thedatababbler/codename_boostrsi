"""Greedy subgraph growth, shared by both modes (doc section 6.1)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from ..graph import LOOP_EDIT, AnyEdit, Limits, MacroEdit, Subgraph, apply_edit, edit_summary, edit_type, try_apply
from ..policy import EditStats, Obs


@dataclass
class GrowContext:
    h: Subgraph
    items: list
    inputs: dict
    step: int
    obs: Obs
    limits: Limits
    run_id: str = ""


@dataclass
class GrowResult:
    h: Subgraph
    fit: float
    kept: list = field(default_factory=list)  # edits kept, in order
    n_evals: int = 0


def make_obs(h: Subgraph, fit_h: float, cost_pen: float, residual: dict, stats: EditStats, rnd: int,
             step: int) -> Obs:
    sg = {"n_nodes": len(h.nodes), **h.role_counts(), "depth": h.depth(), "fit": fit_h,
          "cost_penalty": cost_pen}
    return Obs(subgraph=sg, residual=residual, edit_stats=stats.view(), round=rnd, step=step)


def is_loop(e) -> bool:
    return getattr(e, "kind", "") == LOOP_EDIT


def one_per_kind(ranked: list, loop_pick: str = "rank", seed: int = 0) -> list:
    """Round 4 Stage B (cfg search.slot_mode = one_per_kind): one loop edit, one macro and one atomic edit per
    step: the best-ranked of each kind, except that with loop_pick = random the loop edit is drawn uniformly
    (seeded) from all loop edits -- f0 scores every loop setting the same, so its order among them is only the
    string tie-break, which would always pick the same (node, k, stop rule)."""
    import random

    loops = [e for e in ranked if is_loop(e)]
    macros = [e for e in ranked if isinstance(e, MacroEdit)]
    atomic = [e for e in ranked if not is_loop(e) and not isinstance(e, MacroEdit)]
    loop = (random.Random(seed).choice(sorted(loops, key=str)) if loop_pick == "random" else loops[0]) \
        if loops else None
    picks = [x for x in (loop, macros[0] if macros else None, atomic[0] if atomic else None) if x is not None]
    return sorted(picks, key=ranked.index)


def reserve_slots(ranked: list, cfg, seed: int = 0) -> list:
    """The top_k ranked edits, except that a reserved slot (macro: cfg search.macro_slot; loop edit: cfg loop
    enabled + reserve_slot) replaces the lowest unreserved pick with the best-ranked edit of that kind when none
    made the top_k. Both switches off: exactly ranked[:top_k]."""
    lp = cfg.get("loop") or {}
    if cfg.search.get("slot_mode") == "one_per_kind":
        return one_per_kind(ranked, lp.get("slot_pick", "rank"), seed)
    k = cfg.search.top_k
    top = list(ranked[:k])
    kinds = []
    if cfg.search.get("macro_slot"):
        kinds.append(lambda e: isinstance(e, MacroEdit))
    if lp.get("enabled") and lp.get("reserve_slot", True):
        kinds.append(lambda e: getattr(e, "kind", "") == LOOP_EDIT)
    fixed = set()
    for is_kind in kinds:
        if any(is_kind(e) for e in top):
            fixed |= {i for i, e in enumerate(top) if is_kind(e)}
            continue
        pick = next((e for e in ranked if is_kind(e)), None)
        free = [i for i in range(len(top)) if i not in fixed]
        if pick is None or not free:
            continue
        top[free[-1]] = pick
        fixed.add(free[-1])
    return top


def grow_subgraph(seed: Subgraph, items, inputs, fitness: Callable, proposer, policy, cfg, *, limits: Limits,
                  residual: dict, stats: EditStats, rnd: int, search_seed: int, log=None, tag: str = "",
                  cost_of: Callable | None = None) -> GrowResult:
    """``fitness(h, items, inputs) -> (fit, cost_penalty)``."""
    proposer.start_search(search_seed)
    h = seed
    fit_h, pen_h = fitness(h, items, inputs)
    kept: list[AnyEdit] = []
    n_evals = 1
    fail = max((residual.get("cat_share_hi") or {"": 1}).items(), key=lambda t: t[1])[0]
    for step in range(cfg.search.max_steps):
        obs = make_obs(h, fit_h, pen_h, residual, stats, rnd, step)
        ctx = GrowContext(h=h, items=items, inputs=inputs, step=step, obs=obs, limits=limits)
        cands = proposer.propose(h, ctx)
        if not cands:
            break
        ranked = policy.rank(h, cands, obs)
        from ..config import sub_seed

        top = reserve_slots(ranked, cfg, seed=sub_seed(search_seed, "slot", step))
        hs = [apply_edit(h, e, limits) for e in top]
        # candidates are independent: evaluate them concurrently (results are logged in rank order)
        with ThreadPoolExecutor(max_workers=max(1, len(top))) as pool:
            fits = list(pool.map(lambda h2: fitness(h2, items, inputs), hs))
        n_evals += len(top)
        scored = []
        for rank, (e, h2, (f2, p2)) in enumerate(zip(top, hs, fits)):
            scored.append((f2, p2, e, rank, h2))
            delta = f2 - fit_h
            stats.add(edit_type(e), fail, delta)
            if log is not None:
                log.append("edit_outcomes.jsonl", {
                    "tag": tag, "obs": obs.to_dict(), "edit": edit_summary(h, e), "rank_by_f": rank,
                    "n_cands": len(cands), "delta_fit": delta, "cost_penalty": p2,
                    "from": h.describe(), "to": h2.describe(), "bag": sorted({it.item_id for it in items})})
        best_fit, best_pen, best_e, _, best_h = max(scored, key=lambda t: (t[0], -t[3]))
        if best_fit - fit_h <= cfg.search.min_gain:
            break
        if isinstance(best_e, MacroEdit) and cfg.macro.get("ablate_on_accept") and log is not None:
            for j in range(1, len(best_e.steps)):
                pre = MacroEdit(best_e.steps[:j], f"{best_e.name}[:{j}]")
                hp = try_apply(h, pre, limits)
                if hp is not None:
                    fp, _ = fitness(hp, items, inputs)
                    n_evals += 1
                    log.append("macro_ablation.jsonl", {"tag": tag, "macro": best_e.name, "prefix": j,
                                                         "delta_fit": fp - fit_h, "full_delta": best_fit - fit_h})
        h, fit_h, pen_h = best_h, best_fit, best_pen
        kept.append(best_e)
    return GrowResult(h=h, fit=fit_h, kept=kept, n_evals=n_evals)
