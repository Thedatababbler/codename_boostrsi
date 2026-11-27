"""Evaluation entry point: run a workflow on items from y0, compute g and cost; frozen-prefix reuse.

g = boosttopo.evalx.gap_closure(score, s0) (revision §12), s0 = the score of the item's cached y0. Since every
workflow starts from y0 and its output is the better of y0 and what it produced, g >= 0.

y0 comes from boosttopo's CachedLLM.call_y0 (the benchmark prompt at repeat 0; records under llm.y0_cache_tags are
reused read-only). For splits marked `require_cached_y0` (the test split) a missing y0 aborts: it is never
generated.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from boosttopo.evalx import gap_closure
from boosttopo.roles import y0_request

from .exec import Cost, Ctx, legacy_base, run, start_state
from .model import Block, State, to_str


class MissingY0(RuntimeError):
    pass


@dataclass
class ItemResult:
    item_id: str
    score: float
    legal: bool
    g: float
    cost: Cost
    trace: list
    state: State | None = None
    label: str | None = None   # branch label taken at the last branch, if any
    category: str = ""         # category of the output plan

    @property
    def plan(self) -> str:
        return self.state.best_plan if self.state else ""

    def summary(self) -> dict:
        return {"item": self.item_id, "score": self.score, "legal": self.legal, "g": self.g, "label": self.label,
                "category": self.category, "cost": self.cost.as_dict(),
                "trace": [list(map(_j, t)) for t in self.trace]}


def _j(x):
    return list(map(_j, x)) if isinstance(x, tuple) else x


def y0_cached(llm, item) -> bool:
    """True if call_y0 would be served from the cache (under a y0 tag or the client's own name)."""
    req = y0_request(item, 0)
    for name in list(getattr(llm, "y0_names", [])) + [llm.name]:
        k = llm.key(req, name)
        if (llm.dir / k[:2] / f"{k}.json").exists():
            return True
    return False


@dataclass
class Env:
    llm: object                      # boosttopo.llm.CachedLLM
    bench: object                    # a boosttopo BenchAdapter (FarmAdapter; farm.adapter.ToyFarmAdapter in tests)
    seed_scheme: str = "path"        # path | legacy
    diag_mode: str = "clean"         # clean | original
    start: str = "y0"                # y0 | empty (kept for later; not run)
    workers: int = 8
    parallel_fanout: bool = True
    y0: dict = field(default_factory=dict)
    s0: dict = field(default_factory=dict)
    y0_tokens: dict = field(default_factory=dict)
    _states: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def make_start(self, items, require_cached: bool = False) -> None:
        """y0 for each item (repeat 0 of the benchmark prompt), frozen; aborts on a missing cached y0 if asked."""
        todo = [it for it in items if it.item_id not in self.y0]
        if require_cached:
            missing = [it.item_id for it in todo if not y0_cached(self.llm, it)]
            if missing:
                raise MissingY0(f"{len(missing)} items have no cached y0 (never generated for this split): "
                                f"{missing[:5]}")

        def one(it):
            text, tokens = self.llm.call_y0(y0_request(it, 0))
            return it, text, tokens
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as ex:
            for it, text, tokens in ex.map(one, todo):
                with self._lock:
                    self.y0[it.item_id] = text
                    self.y0_tokens[it.item_id] = tokens
                    self.s0[it.item_id] = self.bench.score(it, text).score

    def g(self, item_id: str, score: float) -> float:
        return gap_closure(score, self.s0[item_id])

    def ctx(self, item, run_seed=0, rep=0, path=(), tokens_before=0) -> Ctx:
        if self.seed_scheme == "legacy":
            rep = legacy_base(rep, run_seed)
        iid = item.item_id
        return Ctx(self.llm, self.bench, item, run_seed, rep, path, g_of=lambda s: self.g(iid, s),
                   seed_scheme=self.seed_scheme, diag_mode=self.diag_mode, tokens_before=tokens_before,
                   parallel_fanout=self.parallel_fanout)

    def initial(self, item) -> State:
        if item.item_id not in self.y0:
            self.make_start([item])
        if self.start == "empty":
            return start_state(item, "", self.bench.score(item, ""))
        y0 = self.y0[item.item_id]
        return start_state(item, y0, self.bench.score(item, y0))


def _result(item, st: State, ctx: Ctx, env: Env) -> ItemResult:
    labels = [t[3] for t in ctx.trace if len(t) > 1 and t[1] == "branch"]
    return ItemResult(item.item_id, st.best_score, st.best_legal, env.g(item.item_id, st.best_score), ctx.cost,
                      list(ctx.trace), st, labels[-1] if labels else None, st.best_cat)


def run_item(wf: Block, item, env: Env, run_seed=0, rep=0, start: State | None = None, path: tuple = (),
             tokens_before: int = 0) -> ItemResult:
    """A fresh run (start=None) is charged y0's tokens; a continuation from a frozen prefix's state is not
    (the prefix paid), but its budget stops see the prefix's tokens through tokens_before."""
    ctx = env.ctx(item, run_seed, rep, path, tokens_before)
    if start is None:
        st = env.initial(item)
        if env.start == "y0":
            ctx.cost.add_y0(env.y0_tokens[item.item_id])
    else:
        st = start.copy()
    st = run(wf, st, ctx)
    return _result(item, st, ctx, env)


def evaluate(wf: Block, items, env: Env, run_seed=0, rep=0) -> list[ItemResult]:
    env.make_start([it for it in items if it.item_id not in env.y0])
    with ThreadPoolExecutor(max_workers=max(1, env.workers)) as ex:
        return list(ex.map(lambda it: run_item(wf, it, env, run_seed, rep), items))


# ------------------------------------------------------------------ frozen prefixes (§6.1)


def prefix_state(prefix: Block, item, env: Env, run_seed=0, rep=0) -> ItemResult:
    """Memoised result of running an accepted (frozen) workflow; its cost is paid once."""
    key = (to_str(prefix), item.item_id, run_seed, rep, env.seed_scheme, env.diag_mode, env.start)
    with env._lock:
        if key in env._states:
            return env._states[key]
    r = run_item(prefix, item, env, run_seed, rep)
    with env._lock:
        env._states[key] = r
    return r


def _unchanged(prefix_res: ItemResult) -> ItemResult:
    return ItemResult(prefix_res.item_id, prefix_res.score, prefix_res.legal, prefix_res.g, Cost(), [],
                      prefix_res.state, prefix_res.label, prefix_res.category)


def run_fragment(prefix_res: ItemResult, frag: Block, item, env: Env, path: tuple,
                 run_seed=0, rep=0) -> ItemResult:
    """Continue from a frozen prefix's final state, running `frag` at structural `path`.

    Because seeds derive from the structural path, this produces exactly the same samples as running the full
    attached workflow from scratch, but charges only the fragment's cost.
    """
    if prefix_res.g >= 1.0:   # the full run would have stopped early
        return _unchanged(prefix_res)
    return run_item(frag, item, env, run_seed, rep, start=prefix_res.state, path=path,
                    tokens_before=prefix_res.cost.tokens)


def run_tail(prefix_res: ItemResult, frag: Block, item, env: Env, offset: int,
             run_seed=0, rep=0) -> ItemResult:
    """Serial mode: `frag`'s children are spliced onto the end of the root seq at index `offset`.
    Runs them with the same structural paths the full workflow would use."""
    if prefix_res.g >= 1.0:
        return _unchanged(prefix_res)
    ctx = env.ctx(item, run_seed, rep, (), prefix_res.cost.tokens)
    st = prefix_res.state.copy()
    for j, c in enumerate(frag.children):
        st = run(c, st, ctx.child(offset + j))
        if ctx.solved(st.best_score):
            break
    return _result(item, st, ctx, env)
