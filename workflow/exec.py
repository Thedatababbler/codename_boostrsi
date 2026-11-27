"""Recursive workflow interpreter and cost accounting (doc §4, revision §12).

Semantics:
- Start: plan = best_plan = y0 (the cached start solution), best_score = s0. y0's tokens are charged to every
  method (Cost.y0_tokens), as in the multi-agent code. `start: empty` (not run yet) starts from an empty plan.
- best_plan is replaced only by a strictly higher score. The output of a run is best_plan, i.e. the better of y0
  and everything the workflow produced (the multi-agent gate).
- seq stops early once g(best) = 1 (multi-agent GBDT: later stages are skipped at g = 1), except inside a gate.
- loop (= graph.run_subgraph's loop attribute): the working plan of the loop's lineage is replaced only when the
  new plan scores >= the current one; the stop rule runs before each repetition (g(best) = 1; stop=legal: the
  lineage best is legal; budget: the item's tokens so far, y0 included, reached the budget); the loop's output
  plan is the best plan of the lineage (which starts with the loop's input).
- gate (= one multi-agent stage): the body's final plan replaces the input only if it scores strictly higher;
  inside a gate best_plan is not updated and seqs do not stop early (a stage runs all its nodes); notes and diag
  written inside are stage-local.
- fanout: branches run on copies, the branch with the highest best score wins (ties -> first branch).

Seeds (Request.rep, which is both the server seed and part of the LLM cache key):
- seed_scheme "path" (all new structures): hash(run seed, item id, structural path incl. loop / fanout indices,
  repetition, step name).
- seed_scheme "legacy" (baselines that must hit the multi-agent cache): the repeat index scheme of boosttopo.
  Base = rep + run_seed * 1e6; fanout branch j -> base + j; loop repetition j (1..k) -> base + j * 7919;
  gate(occ=o) -> base + o * 20011; every step of a stage uses the stage's index.
"""
from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from boosttopo.evalx import Evaluator
from boosttopo.graph import LOOP_REP_STRIDE
from boosttopo.infer import SEED_STRIDE

from .model import Block, State, Step, seq
from .skills import request
from .tools import CHECKS, TOOLS


@dataclass
class Cost:
    llm_calls: int = 0
    new_llm_calls: int = 0        # not served from cache
    tokens: int = 0               # total tokens incl. thinking and y0
    prompt_tokens: int = 0        # when the record has metadata
    completion_tokens: int = 0
    y0_tokens: int = 0
    scorer_calls: int = 0         # plan scores + diagnoses (validate)
    tool_calls: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add_llm(self, tokens: int, meta: dict, cached: bool):
        with self._lock:
            self.llm_calls += 1
            self.new_llm_calls += 0 if cached else 1
            self.tokens += tokens
            self.prompt_tokens += int(meta.get("prompt_tokens") or 0)
            self.completion_tokens += int(meta.get("completion_tokens") or 0)

    def add_y0(self, tokens: int):
        with self._lock:
            self.y0_tokens += tokens
            self.tokens += tokens

    def add_tool(self, name):
        with self._lock:
            self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    def add_scorer(self):
        with self._lock:
            self.scorer_calls += 1

    def absorb(self, o: "Cost"):
        with self._lock:
            for k in ("llm_calls", "new_llm_calls", "tokens", "prompt_tokens", "completion_tokens", "y0_tokens",
                      "scorer_calls"):
                setattr(self, k, getattr(self, k) + getattr(o, k))
            for k, v in o.tool_calls.items():
                self.tool_calls[k] = self.tool_calls.get(k, 0) + v

    def merge(self, o: "Cost") -> "Cost":
        c = Cost()
        c.absorb(self)
        c.absorb(o)
        return c

    def as_dict(self) -> dict:
        return {"llm_calls": self.llm_calls, "new_llm_calls": self.new_llm_calls, "tokens": self.tokens,
                "y0_tokens": self.y0_tokens, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "scorer_calls": self.scorer_calls,
                "tool_calls": sum(self.tool_calls.values()), "tool_calls_by_name": dict(self.tool_calls)}


@dataclass
class Ctx:
    llm: object
    bench: object
    item: object
    run_seed: int = 0
    rep: int = 0                  # path: repetition index; legacy: the current repeat index
    path: tuple = ()              # structural path: child idx / branch label / ("it"|"fo", i)
    cost: Cost = field(default_factory=Cost)
    trace: list = field(default_factory=list)
    g_of: object = None           # score -> g of this item
    seed_scheme: str = "path"
    diag_mode: str = "clean"
    in_gate: bool = False
    tokens_before: int = 0        # tokens spent before this context's cost started (frozen prefix / fanout fork)
    parallel_fanout: bool = True
    _tl: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def child(self, key) -> "Ctx":
        return replace(self, path=self.path + (key,))

    def seed(self, step: Step) -> int:
        if self.seed_scheme == "legacy":
            return self.rep
        h = hashlib.sha256(repr((self.run_seed, self.item.item_id, self.path, self.rep, step.name)).encode()).digest()
        return int.from_bytes(h[:4], "big") & 0x7FFFFFFF

    def spent(self) -> int:
        return self.tokens_before + self.cost.tokens

    def solved(self, score: float) -> bool:
        return self.g_of(score) >= 1.0

    def log(self, *ev):
        with self._tl:
            self.trace.append((self.path,) + ev)


def legacy_base(rep: int, run_seed: int) -> int:
    return rep + int(run_seed) * SEED_STRIDE


def _set_plan(st: State, plan: str, sr) -> State:
    return replace(st, plan=plan, plan_score=sr.score, plan_legal=sr.valid, plan_cat=sr.category)


def run(node, st: State, ctx: Ctx) -> State:
    if isinstance(node, Step):
        return run_step(node, st, ctx)
    p = node.params
    if node.kind == "seq":
        for i, c in enumerate(node.children):
            st = run(c, st, ctx.child(i))
            if not ctx.in_gate and ctx.solved(st.best_score):
                break                                     # reached g = 1: skip the rest
        return st
    if node.kind == "gate":
        rep = ctx.rep + p.get("occ", 0) * Evaluator.DUP_STRIDE if ctx.seed_scheme == "legacy" else ctx.rep
        inner = replace(ctx, in_gate=True, rep=rep)
        out = run(seq(*node.children), st, inner)
        adopted = out.plan_score > st.plan_score
        ctx.log("gate", adopted)
        res = replace(st, notes=dict(st.notes), trace=st.trace)   # notes / diag are stage-local
        if adopted:
            res = replace(res, plan=out.plan, plan_score=out.plan_score, plan_legal=out.plan_legal,
                          plan_cat=out.plan_cat)
            if not ctx.in_gate and out.plan_score > res.best_score:
                res = replace(res, best_plan=out.plan, best_score=out.plan_score, best_legal=out.plan_legal,
                              best_cat=out.plan_cat)
        return res
    if node.kind == "loop":
        body = seq(*node.children)
        cur, cur_s, cur_l, cur_c = st.plan, st.plan_score, st.plan_legal, st.plan_cat
        lb, lb_s, lb_l, lb_c = cur, cur_s, cur_l, cur_c          # best of the lineage
        base = ctx.rep
        it = 0
        for i in range(p["k"]):
            if ctx.solved(max(lb_s, st.best_score)):
                break
            if p["stop"] == "legal" and lb_l:
                break
            if "budget" in p and ctx.spent() >= p["budget"]:
                break
            sub = ctx.child(("it", i))
            if ctx.seed_scheme == "legacy":
                sub = replace(sub, rep=base + (i + 1) * LOOP_REP_STRIDE)
            st = run(body, replace(st, plan=cur, plan_score=cur_s, plan_legal=cur_l, plan_cat=cur_c), sub)
            it = i + 1
            if st.plan_score >= cur_s:
                cur, cur_s, cur_l, cur_c = st.plan, st.plan_score, st.plan_legal, st.plan_cat
            if st.plan_score > lb_s:
                lb, lb_s, lb_l, lb_c = st.plan, st.plan_score, st.plan_legal, st.plan_cat
        ctx.log("loop", it)
        return replace(st, plan=lb, plan_score=lb_s, plan_legal=lb_l, plan_cat=lb_c)
    if node.kind == "branch":
        label = CHECKS[p["check"]](st, ctx)
        cases = node.cases
        ctx.log("branch", p["check"], label, label in cases)
        if label not in cases:
            return st                                     # no case for this label: pass through
        return run(cases[label], st, ctx.child(label))
    if node.kind == "fanout":
        b = seq(*node.children)
        n = p["n"]
        spent = ctx.spent()

        def one(j):
            sub = replace(ctx.child(("fo", j)), cost=Cost(), trace=[], tokens_before=spent, _tl=threading.Lock())
            if ctx.seed_scheme == "legacy":
                sub.rep = ctx.rep + j
            return run(b, st.copy(), sub), sub

        if ctx.parallel_fanout and n > 1:
            with ThreadPoolExecutor(max_workers=n) as ex:
                outs = list(ex.map(one, range(n)))
        else:
            outs = [one(j) for j in range(n)]
        for _, sub in outs:                                # merge in branch order (deterministic trace)
            ctx.cost.absorb(sub.cost)
            with ctx._tl:
                ctx.trace.extend(sub.trace)
        key = (lambda s: s.plan_score) if ctx.in_gate else (lambda s: s.best_score)
        return max((s for s, _ in outs), key=key)          # ties -> first branch
    raise ValueError(node.kind)


def write(st: State, field_name: str, value) -> State:
    if field_name.startswith("notes."):
        notes = dict(st.notes)
        notes[field_name[6:]] = value
        return replace(st, notes=notes)
    return replace(st, **{field_name: value})


def _cached(llm, req) -> bool:
    try:
        k = llm.key(req)
        return (llm.dir / k[:2] / f"{k}.json").exists()
    except AttributeError:
        return False


def run_step(step: Step, st: State, ctx: Ctx) -> State:
    if step.kind == "llm":
        req = request(step.name, st, ctx.item, ctx.seed(step))
        cached = _cached(ctx.llm, req)
        if hasattr(ctx.llm, "call_meta"):
            out, tokens, meta = ctx.llm.call_meta(req)
        else:
            (out, tokens), meta = ctx.llm.call(req), {}
        ctx.cost.add_llm(tokens, meta, cached)
        ctx.log("llm", step.name, req.rep, tokens)
    elif step.kind == "tool":
        out = TOOLS[step.name](st, ctx)                   # deterministic, no tokens
        ctx.cost.add_tool(step.name)
        if step.name == "validate":
            ctx.cost.add_scorer()                         # a diagnosis is a verifier query
    else:
        out = CHECKS[step.name](st, ctx)
        ctx.cost.add_tool(step.name)
    st = write(st, step.writes, out)
    if step.writes == "plan":
        sr = ctx.bench.score(ctx.item, st.plan)
        ctx.cost.add_scorer()
        st = _set_plan(st, st.plan, sr)
        if not ctx.in_gate and sr.score > st.best_score:
            st = replace(st, best_plan=st.plan, best_score=sr.score, best_legal=sr.valid, best_cat=sr.category)
    return st


def start_state(item, plan: str, sr) -> State:
    return State(problem=item.prompt, plan=plan, best_plan=plan, best_score=sr.score, plan_score=sr.score,
                 plan_legal=sr.valid, plan_cat=sr.category, best_legal=sr.valid, best_cat=sr.category)
