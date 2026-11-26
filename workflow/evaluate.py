"""Evaluation entry point: run a workflow on items, compute g, cost; frozen-prefix reuse.

g (improvement) = (score - y0) / (1 - y0), where y0 is the cached start-point score (§4.2).
y0's own cost is never charged to any method.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .exec import Cost, Ctx, initial_state, run
from .model import Block, State, seq, make_step, to_str


@dataclass
class ItemResult:
    item_id: str
    score: float
    raw: float
    legal: bool
    g: float
    cost: Cost
    trace: list
    state: State | None = None
    label: str | None = None   # branch label taken at the last top-level branch, if any

    def summary(self) -> dict:
        return {"item": self.item_id, "score": self.score, "raw": self.raw, "legal": self.legal,
                "g": self.g, "label": self.label, "cost": self.cost.as_dict(),
                "trace": [list(map(_j, t)) for t in self.trace]}


def _j(x):
    return list(map(_j, x)) if isinstance(x, tuple) else x


def g_of(score: float, y0: float) -> float:
    if y0 >= 1.0:
        return 1.0 if score >= 1.0 else score - y0
    return (score - y0) / (1.0 - y0)


@dataclass
class Env:
    llm: object
    bench: object
    check_cfg: dict = field(default_factory=lambda: {"status_threshold": 0.8, "status_mode": "score"})
    workers: int = 8
    parallel_fanout: bool = True
    _y0: dict = field(default_factory=dict)
    _states: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def ctx(self, item, run_seed=0, rep=0, path=()) -> Ctx:
        return Ctx(self.llm, self.bench, item, run_seed, rep, path, check_cfg=self.check_cfg,
                   parallel_fanout=self.parallel_fanout)

    def y0(self, item) -> float:
        if item.y0_score is not None:
            return item.y0_score
        with self._lock:
            if item.id in self._y0:
                return self._y0[item.id]
        st = run(seq(make_step("draft")), initial_state(item), self.ctx(item, run_seed="y0"))
        with self._lock:
            self._y0[item.id] = st.best_score
        return st.best_score


def _result(item, st: State, ctx: Ctx, env: Env) -> ItemResult:
    labels = [t[3] for t in ctx.trace if len(t) > 1 and t[1] == "branch"]
    return ItemResult(item.id, st.best_score, st.raw_best, st.best_legal,
                      g_of(st.best_score, env.y0(item)), ctx.cost, list(ctx.trace), st,
                      labels[-1] if labels else None)


def run_item(wf: Block, item, env: Env, run_seed=0, rep=0, start: State | None = None,
             path: tuple = ()) -> ItemResult:
    ctx = env.ctx(item, run_seed, rep, path)
    st = run(wf, start.copy() if start is not None else initial_state(item), ctx)
    return _result(item, st, ctx, env)


def evaluate(wf: Block, items, env: Env, run_seed=0, rep=0) -> list[ItemResult]:
    with ThreadPoolExecutor(max_workers=max(1, env.workers)) as ex:
        return list(ex.map(lambda it: run_item(wf, it, env, run_seed, rep), items))


# ------------------------------------------------------------------ frozen prefixes (§6.1)


def prefix_state(prefix: Block, item, env: Env, run_seed=0, rep=0) -> ItemResult:
    """Memoised result of running an accepted (frozen) workflow; its cost is paid once."""
    key = (to_str(prefix), item.id, run_seed, rep)
    with env._lock:
        if key in env._states:
            return env._states[key]
    r = run_item(prefix, item, env, run_seed, rep)
    with env._lock:
        env._states[key] = r
    return r


def run_fragment(prefix_res: ItemResult, frag: Block, item, env: Env, path: tuple,
                 run_seed=0, rep=0) -> ItemResult:
    """Continue from a frozen prefix's final state, running `frag` at structural `path`.

    Because seeds derive from the structural path, this produces exactly the same samples as
    running the full attached workflow from scratch, but charges only the fragment's cost.
    """
    if prefix_res.state.best_score >= 1.0:   # the full run would have stopped early
        return ItemResult(item.id, prefix_res.score, prefix_res.raw, prefix_res.legal,
                          prefix_res.g, Cost(), [], prefix_res.state, prefix_res.label)
    return run_item(frag, item, env, run_seed, rep, start=prefix_res.state, path=path)


def run_tail(prefix_res: ItemResult, frag: Block, item, env: Env, offset: int,
             run_seed=0, rep=0) -> ItemResult:
    """Serial mode: `frag`'s children are spliced onto the end of the root seq at index `offset`.
    Runs them with the same structural paths the full workflow would use."""
    if prefix_res.state.best_score >= 1.0:
        return ItemResult(item.id, prefix_res.score, prefix_res.raw, prefix_res.legal,
                          prefix_res.g, Cost(), [], prefix_res.state, prefix_res.label)
    ctx = env.ctx(item, run_seed, rep)
    st = prefix_res.state.copy()
    for j, c in enumerate(frag.children):
        st = run(c, st, ctx.child(offset + j))
        if st.best_score >= 1.0:
            break
    return _result(item, st, ctx, env)
