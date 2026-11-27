"""Workflow representation (doc §2): State, Step, Block, legality checks, worst-case calls.

A workflow is a tree of Steps and control Blocks. All steps read/write one shared State.
Blocks are immutable and hashable so they can be used as cache / dedup keys.
"""
from __future__ import annotations

import copy as _copy
import re
from dataclasses import dataclass, field, replace
from typing import Iterator, Union

# --------------------------------------------------------------------------- state


@dataclass
class State:
    problem: str                      # problem text (read-only)
    plan: str | None = None           # current plan
    best_plan: str | None = None      # highest-scoring plan so far
    best_score: float = 0.0
    diag: str | None = None           # latest diagnosis from a check / tool
    notes: dict = field(default_factory=dict)  # tool outputs & analyses, by name
    # --- executor-maintained (deviation, see docs/IMPLEMENTATION_NOTES.md) ---
    plan_score: float = 0.0           # score of the *current* plan
    plan_legal: bool = False          # legality of the *current* plan
    plan_cat: str = ""                # ScoreResult.category of the current plan (status / error_type checks)
    best_legal: bool = False
    best_cat: str = ""
    trace: list = field(default_factory=list)  # execution trace events

    def copy(self) -> "State":
        return replace(self, notes=dict(self.notes), trace=list(self.trace))


# Fields readable from the start. `notes` is always readable (missing keys render empty, §3.4).
# Revision §12: every workflow starts from y0 (plan = best_plan = y0), so plan is readable from the start.
# With `start: empty` the plan is the empty string, which the writer prompt also accepts.
INITIAL_FIELDS = frozenset({"problem", "notes", "best_score", "plan", "best_plan"})


def field_root(f: str) -> str:
    return f.split(".", 1)[0]


# --------------------------------------------------------------------------- tree nodes


@dataclass(frozen=True)
class Step:
    kind: str                 # "llm" | "tool" | "check"
    name: str                 # skill / tool / check name (§3)
    reads: tuple = ()         # state fields read; "notes.sim" style allowed
    writes: str = ""          # field written back ("plan", "diag", "notes.sim", ...)

    def __str__(self) -> str:
        return self.name


def _freeze(params: dict) -> tuple:
    return tuple(sorted(params.items()))


@dataclass(frozen=True)
class Block:
    kind: str                 # "seq" | "loop" | "branch" | "fanout" | "gate"
    children: tuple = ()
    # stored frozen as sorted (key, value) pairs; use .params for a dict view.
    # branch: {"check": name, "labels": tuple-of-labels parallel to children}
    frozen_params: tuple = ()

    @property
    def params(self) -> dict:
        return dict(self.frozen_params)

    def with_children(self, children) -> "Block":
        return replace(self, children=tuple(children))

    def with_params(self, **kw) -> "Block":
        p = self.params
        p.update(kw)
        return replace(self, frozen_params=_freeze(p))

    # branch helpers
    @property
    def cases(self) -> dict:
        assert self.kind == "branch"
        return dict(zip(self.params["labels"], self.children))

    def __str__(self) -> str:
        return to_str(self)


Node = Union[Step, Block]
Workflow = Block  # root is a seq


# --------------------------------------------------------------------------- constructors


def seq(*children) -> Block:
    return Block("seq", tuple(children))


def loop(*children, k: int = 3, stop: str = "legal", budget: int | None = None) -> Block:
    p = {"k": k, "stop": stop}
    if budget is not None:
        p["budget"] = budget          # stop before a repetition once the item's tokens (incl. y0) reach it
    return Block("loop", tuple(children), _freeze(p))


def gate(*children, occ: int = 0) -> Block:
    """One gated stage (the multi-agent GBDT member): its output replaces the input plan only if it scores
    strictly higher; notes written inside are stage-local. occ = occurrence index among same-structure members
    (legacy seeds only)."""
    return Block("gate", tuple(children), _freeze({"occ": occ}) if occ else ())


def fanout(*children, n: int = 3) -> Block:
    return Block("fanout", tuple(children), _freeze({"n": n}))


def branch(check: str = "status", cases: dict | None = None) -> Block:
    cases = cases or {}
    labels = tuple(cases.keys())
    return Block("branch", tuple(cases[l] for l in labels), _freeze({"check": check, "labels": labels}))


def empty_seq() -> Block:
    return seq()


def body(node: Block) -> Block:
    """Loop / fanout bodies execute their children as a seq."""
    return seq(*node.children)


# --------------------------------------------------------------------------- registries
# Static I/O signatures of skills / tools / checks. The behaviour lives in skills.py / tools.py;
# this module only needs signatures for legality checks, so it stays import-light.

SKILL_SIGS = {
    "draft":    (("problem", "plan", "notes"), "plan"),           # writer role on the current plan
    "repair":   (("problem", "plan", "diag", "notes"), "plan"),   # repairer role
    "improve":  (("problem", "plan", "notes"), "plan"),           # new skill
    "analyze":  (("problem", "notes"), "notes.analysis"),         # analyst role
    "critique": (("problem", "plan", "notes"), "notes.critique"), # critic role (legacy ensembles; not in edits)
}
TOOL_SIGS = {
    "validate":   (("plan",), "diag"),
    "simulate":   (("plan",), "notes.sim"),
    "crop_stats": (("problem",), "notes.crops"),
}
# check name -> (fields it needs, possible labels)
CHECK_SIGS = {
    "status":     (("plan",), ("illegal", "legal_low", "legal_high")),
    "error_type": (("plan",), ("funds", "farmland", "not_mature", "inventory", "other", "none")),
}


def make_step(name: str) -> Step:
    if name in SKILL_SIGS:
        r, w = SKILL_SIGS[name]
        return Step("llm", name, r, w)
    if name in TOOL_SIGS:
        r, w = TOOL_SIGS[name]
        return Step("tool", name, r, w)
    if name in CHECK_SIGS:
        r, _ = CHECK_SIGS[name]
        return Step("check", name, r, f"notes.check_{name}")
    raise KeyError(f"unknown step {name!r}")


# --------------------------------------------------------------------------- limits & analysis


@dataclass(frozen=True)
class Limits:
    max_depth: int = 3        # block nesting depth, root seq = 1
    max_llm_steps: int = 6    # static count of llm steps
    max_calls: int = 12       # worst-case llm calls (revision §12)
    k_values: tuple = (2, 3, 5)
    n_values: tuple = (2, 3, 5)


def iter_nodes(node: Node, path: tuple = ()) -> Iterator[tuple[tuple, Node]]:
    """Pre-order traversal yielding (index-path, node)."""
    yield path, node
    if isinstance(node, Block):
        for i, c in enumerate(node.children):
            yield from iter_nodes(c, path + (i,))


def get(node: Node, path: tuple) -> Node:
    for i in path:
        node = node.children[i]
    return node


def put(root: Block, path: tuple, new: Node | None) -> Block:
    """Return root with node at path replaced by `new` (None deletes it)."""
    if not path:
        assert isinstance(new, Block)
        return new
    parent = get(root, path[:-1])
    ch = list(parent.children)
    i = path[-1]
    if new is None:
        del ch[i]
        if parent.kind == "branch":
            labels = list(parent.params["labels"])
            del labels[i]
            newp = parent.with_children(ch).with_params(labels=tuple(labels))
            return put(root, path[:-1], newp)
    else:
        ch[i] = new
    return put(root, path[:-1], parent.with_children(ch))


def insert_at(root: Block, path: tuple, new: Node) -> Block:
    """Insert `new` so that it ends up at `path` (parent must be seq/loop/fanout)."""
    parent = get(root, path[:-1])
    assert isinstance(parent, Block) and parent.kind != "branch"
    ch = list(parent.children)
    ch.insert(path[-1], new)
    return put(root, path[:-1], parent.with_children(ch))


def depth(node: Node) -> int:
    if isinstance(node, Step):
        return 0
    return 1 + max((depth(c) for c in node.children), default=0)


def llm_step_count(node: Node) -> int:
    return sum(1 for _, n in iter_nodes(node) if isinstance(n, Step) and n.kind == "llm")


def worst_case_calls(node: Node) -> int:
    """Loops count k times, fanouts n times, branches their most expensive case (§2.4)."""
    if isinstance(node, Step):
        return 1 if node.kind == "llm" else 0
    if node.kind in ("seq", "gate"):
        return sum(worst_case_calls(c) for c in node.children)
    inner = sum(worst_case_calls(c) for c in node.children)
    if node.kind == "loop":
        return node.params["k"] * inner
    if node.kind == "fanout":
        return node.params["n"] * inner
    if node.kind == "branch":
        return max((worst_case_calls(c) for c in node.children), default=0)
    raise ValueError(node.kind)


def writes_plan(node: Node) -> bool:
    return any(isinstance(n, Step) and n.writes == "plan" for _, n in iter_nodes(node))


def _defined_after(node: Node, defined: frozenset, errors: list, path: tuple) -> frozenset:
    """Definite-assignment analysis. Appends reads-before-write violations to `errors`."""
    if isinstance(node, Step):
        for f in node.reads:
            if field_root(f) not in defined and f not in defined:
                errors.append(f"{path}: step {node.name} reads {f!r} before it is written")
        w = node.writes
        out = defined | {w, field_root(w)} if not w.startswith("notes.") else defined | {w}
        if w == "plan":
            out = out | {"best_plan"}  # executor updates best_plan whenever plan is written
        return frozenset(out)
    if node.kind in ("seq", "loop", "fanout", "gate"):
        d = defined
        for i, c in enumerate(node.children):
            d = _defined_after(c, d, errors, path + (i,))
        if node.kind == "loop":
            return defined  # the stop check runs before every repetition, so a loop may run zero times
        if node.kind == "gate":
            return defined  # diag and notes written inside a gate are stage-local
        return d  # a fanout body always runs (n >= 1)
    if node.kind == "branch":
        check = node.params["check"]
        if check not in CHECK_SIGS:
            errors.append(f"{path}: unknown check {check!r}")
            return defined
        need, labels = CHECK_SIGS[check]
        for f in need:
            if f not in defined:
                errors.append(f"{path}: check {check} needs {f!r} which has no source")
        outs = [_defined_after(c, defined, errors, path + (i,)) for i, c in enumerate(node.children)]
        if set(node.params["labels"]) != set(labels):
            outs.append(defined)  # pass-through path for uncovered labels
        res = outs[0]
        for o in outs[1:]:
            res = res & o
        return frozenset(res)
    raise ValueError(node.kind)


def validate(wf: Block, limits: Limits = Limits(), require_plan: bool = True,
             defined: frozenset = INITIAL_FIELDS) -> list[str]:
    """Return a list of violations; empty list means the workflow is legal (§2.4)."""
    errs: list[str] = []
    if not isinstance(wf, Block) or wf.kind != "seq":
        errs.append("root must be a seq block")
        return errs
    for path, n in iter_nodes(wf):
        if isinstance(n, Step):
            sigs = {"llm": SKILL_SIGS, "tool": TOOL_SIGS, "check": CHECK_SIGS}.get(n.kind)
            if sigs is None or n.name not in sigs:
                errs.append(f"{path}: unknown {n.kind} step {n.name!r}")
            continue
        p = n.params
        if n.kind not in ("seq", "loop", "branch", "fanout", "gate"):
            errs.append(f"{path}: unknown block kind {n.kind!r}")
        if n.kind == "loop":
            if not isinstance(p.get("k"), int) or p["k"] < 1:
                errs.append(f"{path}: loop k must be a positive int")
            if p.get("stop") not in ("legal", "best"):
                errs.append(f"{path}: loop stop must be 'legal' or 'best'")
            if "budget" in p and (not isinstance(p["budget"], int) or p["budget"] < 1):
                errs.append(f"{path}: loop budget must be a positive int (tokens)")
        if n.kind == "fanout" and (not isinstance(p.get("n"), int) or p["n"] < 1):
            errs.append(f"{path}: fanout n must be a positive int")
        if n.kind in ("loop", "fanout", "gate") and not writes_plan(n):
            errs.append(f"{path}: {n.kind} body must contain a step that writes plan")
        if n.kind == "branch":
            labels = p.get("labels", ())
            if len(labels) != len(n.children) or len(set(labels)) != len(labels):
                errs.append(f"{path}: branch labels must be unique and match children")
            known = CHECK_SIGS.get(p.get("check"), ((), ()))[1]
            for l in labels:
                if l not in known:
                    errs.append(f"{path}: label {l!r} not produced by check {p.get('check')!r}")
            for c in n.children:
                if isinstance(c, Step):
                    errs.append(f"{path}: branch cases must be blocks")
    if depth(wf) > limits.max_depth:
        errs.append(f"nesting depth {depth(wf)} > {limits.max_depth}")
    if llm_step_count(wf) > limits.max_llm_steps:
        errs.append(f"llm steps {llm_step_count(wf)} > {limits.max_llm_steps}")
    if worst_case_calls(wf) > limits.max_calls:
        errs.append(f"worst-case calls {worst_case_calls(wf)} > {limits.max_calls}")
    if require_plan and not writes_plan(wf):
        errs.append("workflow has no step that writes plan")
    _defined_after(wf, defined, errs, ())
    return errs


def is_valid(wf: Block, limits: Limits = Limits(), **kw) -> bool:
    return not validate(wf, limits, **kw)


# --------------------------------------------------------------------------- text form
# seq[crop_stats, draft, branch(status){illegal: loop(k=3, stop=legal)[validate, repair]}]


def to_str(node: Node) -> str:
    if isinstance(node, Step):
        return node.name
    if node.kind == "branch":
        p = node.params
        inner = ", ".join(f"{l}: {to_str(c)}" for l, c in zip(p["labels"], node.children))
        return f"branch({p['check']}){{{inner}}}"
    inner = ", ".join(to_str(c) for c in node.children)
    args = ", ".join(f"{k}={v}" for k, v in node.frozen_params)
    return f"{node.kind}({args})[{inner}]" if args else f"{node.kind}[{inner}]"


_TOK = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9]*|[\[\]\(\)\{\}:,=]|\d+)")


def parse(text: str) -> Node:
    """Parse the textual form produced by to_str (also accepts `a -> b` sugar at top level)."""
    if "->" in text or "→" in text:
        parts = [p.strip() for p in re.split(r"->|→", text)]
        return seq(*[parse(p) for p in parts])
    toks = _TOK.findall(text)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def eat(t=None):
        nonlocal pos
        tok = toks[pos]
        if t is not None and tok != t:
            raise SyntaxError(f"expected {t!r}, got {tok!r} in {text!r}")
        pos += 1
        return tok

    def node():
        name = eat()
        if name not in ("seq", "loop", "fanout", "branch", "gate"):
            return make_step(name)
        params: dict = {}
        if peek() == "(":
            eat("(")
            if name == "branch":
                params["check"] = eat()
            else:
                while peek() != ")":
                    k = eat(); eat("=")
                    v = eat()
                    params[k] = int(v) if v.isdigit() else v
                    if peek() == ",":
                        eat(",")
            eat(")")
        if name == "branch":
            eat("{")
            cases = {}
            while peek() != "}":
                lab = eat(); eat(":")
                c = node()
                cases[lab] = c if isinstance(c, Block) else seq(c)
                if peek() == ",":
                    eat(",")
            eat("}")
            return branch(params.get("check", "status"), cases)
        eat("[")
        ch = []
        while peek() != "]":
            ch.append(node())
            if peek() == ",":
                eat(",")
        eat("]")
        if name == "seq":
            return seq(*ch)
        if name == "loop":
            return loop(*ch, k=params.get("k", 3), stop=params.get("stop", "legal"), budget=params.get("budget"))
        if name == "gate":
            return gate(*ch, occ=params.get("occ", 0))
        return fanout(*ch, n=params.get("n", 3))

    out = node()
    if pos != len(toks):
        raise SyntaxError(f"trailing tokens in {text!r}: {toks[pos:]}")
    return out


def parse_wf(text: str) -> Block:
    n = parse(text)
    return n if isinstance(n, Block) and n.kind == "seq" else seq(n)


def deepcopy(x):
    return _copy.deepcopy(x)
