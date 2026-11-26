"""Recursive workflow interpreter, cost accounting (doc §4).

Semantics (§4.1):
- Progress is kept: loop iterations continue from the latest `plan`, not `best_plan`
  (use the explicit `reset_to_best` tool to restart from the best one).
- Every execution is a fresh sample: the seed is derived from (run seed, the step's structural
  path in the tree incl. loop/fanout indices, repetition index); the cache key contains it.
- Cost is what actually ran: tokens incl. thinking; scorer and tool calls counted separately;
  untaken branches and skipped loop iterations cost nothing.

The output of a run is always `best_plan` — the equivalent of the first version's gate.
"""
from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from .model import Block, State, Step, seq
from .skills import extract_output, render
from .tools import CHECKS, TOOLS


@dataclass
class Cost:
    llm_calls: int = 0
    new_llm_calls: int = 0        # not served from cache
    prompt_tokens: int = 0
    completion_tokens: int = 0    # includes thinking
    scorer_calls: int = 0
    tool_calls: dict = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add_llm(self, usage):
        with self._lock:
            self.llm_calls += 1
            self.new_llm_calls += 0 if usage.cached else 1
            self.prompt_tokens += usage.prompt_tokens
            self.completion_tokens += usage.completion_tokens

    def add_tool(self, name):
        with self._lock:
            self.tool_calls[name] = self.tool_calls.get(name, 0) + 1

    def add_scorer(self):
        with self._lock:
            self.scorer_calls += 1

    def merge(self, o: "Cost") -> "Cost":
        tc = dict(self.tool_calls)
        for k, v in o.tool_calls.items():
            tc[k] = tc.get(k, 0) + v
        return Cost(self.llm_calls + o.llm_calls, self.new_llm_calls + o.new_llm_calls,
                    self.prompt_tokens + o.prompt_tokens, self.completion_tokens + o.completion_tokens,
                    self.scorer_calls + o.scorer_calls, tc)

    def as_dict(self) -> dict:
        return {"llm_calls": self.llm_calls, "new_llm_calls": self.new_llm_calls,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "tokens": self.tokens, "scorer_calls": self.scorer_calls,
                "tool_calls": sum(self.tool_calls.values()), "tool_calls_by_name": dict(self.tool_calls)}


@dataclass
class Ctx:
    llm: object
    bench: object
    item: object
    run_seed: int = 0
    rep: int = 0
    path: tuple = ()              # structural path: child idx / branch label / ("it"|"fo", i)
    cost: Cost = field(default_factory=Cost)
    trace: list = field(default_factory=list)
    check_cfg: dict = field(default_factory=dict)
    parallel_fanout: bool = True
    _tl: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def child(self, key) -> "Ctx":
        return replace(self, path=self.path + (key,))

    def at(self, i: int, tag: str = "it") -> "Ctx":
        return self.child((tag, i))

    def seed(self, step: Step) -> int:
        h = hashlib.sha256(repr((self.run_seed, getattr(self.item, "id", None), self.path, self.rep, step.name)).encode()).digest()
        return int.from_bytes(h[:4], "big") & 0x7FFFFFFF

    def log(self, *ev):
        with self._tl:
            self.trace.append((self.path,) + ev)


def run(node, st: State, ctx: Ctx) -> State:
    if isinstance(node, Step):
        return run_step(node, st, ctx)
    p = node.params
    if node.kind == "seq":
        for i, c in enumerate(node.children):
            st = run(c, st, ctx.child(i))
            if st.best_score >= 1.0:
                break                                     # reached the reference: stop early
        return st
    if node.kind == "loop":
        it = 0
        for i in range(p["k"]):
            st = run(seq(*node.children), st, ctx.at(i, "it"))
            it = i + 1
            if st.best_score >= 1.0:
                break
            if p["stop"] == "legal" and st.plan_legal:
                break
        ctx.log("loop", it)
        return st
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
        if ctx.parallel_fanout and n > 1:
            with ThreadPoolExecutor(max_workers=n) as ex:
                outs = list(ex.map(lambda j: run(b, st.copy(), ctx.at(j, "fo")), range(n)))
        else:
            outs = [run(b, st.copy(), ctx.at(j, "fo")) for j in range(n)]
        return max(outs, key=lambda s: s.best_score)      # ties -> first branch
    raise ValueError(node.kind)


def write(st: State, field_name: str, value) -> State:
    if field_name.startswith("notes."):
        notes = dict(st.notes)
        notes[field_name[6:]] = value
        return replace(st, notes=notes)
    return replace(st, **{field_name: value})


def run_step(step: Step, st: State, ctx: Ctx) -> State:
    if step.kind == "llm":
        text, usage = ctx.llm.generate(render(step.name, st), seed=ctx.seed(step))
        ctx.cost.add_llm(usage)
        out = extract_output(step.name, text)
    elif step.kind == "tool":
        out = TOOLS[step.name](st, ctx)                   # deterministic, no tokens
        ctx.cost.add_tool(step.name)
        if out is None:                                   # e.g. reset_to_best with no best yet
            return st
    else:
        out = CHECKS[step.name](st, ctx)
        ctx.cost.add_tool(step.name)
    st = write(st, step.writes, out)
    if step.writes == "plan":
        res = ctx.bench.score(ctx.item, st.plan)
        ctx.cost.add_scorer()
        st = replace(st, plan_score=res.score, plan_legal=res.legal)
        better = (st.best_plan is None and (res.legal or res.score > 0)) or res.score > st.best_score \
            or (res.score == st.best_score and res.legal and not st.best_legal)
        if better:
            st = replace(st, best_plan=st.plan, best_score=res.score, raw_best=res.raw,
                         best_legal=res.legal)
    return st


def initial_state(item) -> State:
    return State(problem=item.problem)
