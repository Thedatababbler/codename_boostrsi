"""Edit set (doc §5): atomic edits, fragments, legal_edits, apply_edit.

Paths are index paths in the tree: (2, 0) = first child of the root's third child.
Every edit returned by legal_edits maps a legal workflow to a legal workflow: candidates are
generated structurally and then filtered through model.validate (on the *attached* workflow when
a fragment is being grown in the context of a frozen prefix).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .model import (CHECK_SIGS, SKILL_SIGS, Block, Limits, Step, branch, fanout, get, insert_at,
                    iter_nodes, loop, make_step, parse, put, seq, to_str, validate)

CONTROL_OPS = ("wrap_loop", "wrap_fanout", "add_branch", "set_param")
OTHER_OPS = ("insert_step", "remove", "replace_skill", "unwrap")
TOOL_STEPS = ("validate", "simulate", "crop_stats")
EDIT_SKILLS = ("draft", "repair", "improve", "analyze")  # critique exists only to express multi-agent ensembles


@dataclass(frozen=True)
class Edit:
    op: str
    path: tuple
    args: tuple = ()

    @property
    def category(self) -> str:
        return "control" if self.op in CONTROL_OPS else "other"

    def __str__(self) -> str:
        a = ", ".join(to_str(x) if isinstance(x, (Step, Block)) else repr(x) for x in self.args)
        return f"{self.op}({self.path}{', ' if a else ''}{a})"


@dataclass(frozen=True)
class Fragment:
    name: str
    path: tuple

    category = "fragment"
    op = "fragment"

    def __str__(self) -> str:
        return f"{self.name}@{self.path}"


# ------------------------------------------------------------------ fragment library (§5.2)

FRAGMENTS = {
    "repair_loop": parse("loop(k=3, stop=legal)[validate, repair]"),
    "sim_improve": parse("seq[simulate, improve]"),
    "split_by_status": parse("branch(status){illegal: loop(k=3, stop=legal)[validate, repair], "
                             "legal_low: seq[simulate, improve]}"),
    # stats_first / multi_draft are transforms of an existing draft step, see apply_edit
    "stats_first": None,
    "multi_draft": None,
}
# Sub-flows add_branch may hang under a label.
BRANCH_CHILDREN = (
    parse("loop(k=3, stop=legal)[validate, repair]"),
    parse("seq[simulate, improve]"),
    parse("seq[validate, repair]"),
    parse("seq[improve]"),
)

# What problem each edit targets (used by the f0 policy, §5.3).
REPAIR_NAMES = {"repair", "validate", "repair_loop", "error_type"}
IMPROVE_NAMES = {"simulate", "improve", "sim_improve"}


def edit_tags(e) -> set:
    names = set()
    if isinstance(e, Fragment):
        names.add(e.name)
        if e.name == "split_by_status":
            names |= {"repair_loop", "sim_improve"}
    else:
        for a in e.args:
            if isinstance(a, (Step, Block)):
                names |= {n.name for _, n in iter_nodes(a) if isinstance(n, Step)}
                if isinstance(a, Block) and a.kind == "branch":
                    names.add(a.params["check"])
            elif isinstance(a, str):
                names.add(a)
    tags = set()
    if names & REPAIR_NAMES:
        tags.add("repair")
    if names & IMPROVE_NAMES:
        tags.add("improve")
    return tags


# ------------------------------------------------------------------ apply


def _splice(root: Block, path: tuple, nodes: list) -> Block:
    parent = get(root, path[:-1])
    if parent.kind == "branch":
        return put(root, path, seq(*nodes))
    ch = list(parent.children)
    ch[path[-1]: path[-1] + 1] = nodes
    return put(root, path[:-1], parent.with_children(ch))


def apply_edit(wf: Block, e) -> Block:
    if isinstance(e, Fragment):
        if e.name == "stats_first":
            return insert_at(wf, e.path, make_step("crop_stats"))
        if e.name == "multi_draft":
            return put(wf, e.path, fanout(get(wf, e.path), n=3))
        return insert_at(wf, e.path, FRAGMENTS[e.name])
    op, p, a = e.op, e.path, e.args
    if op == "insert_step":
        return insert_at(wf, p, a[0])
    if op == "remove":
        return put(wf, p, None)
    if op == "replace_skill":
        return put(wf, p, make_step(a[0]))
    if op == "wrap_loop":
        return put(wf, p, loop(get(wf, p), k=a[0], stop=a[1]))
    if op == "wrap_fanout":
        return put(wf, p, fanout(get(wf, p), n=a[0]))
    if op == "add_branch":
        check, label, child = a
        node = get(wf, p) if _exists(wf, p) else None
        if isinstance(node, Block) and node.kind == "branch" and node.params["check"] == check:
            cases = node.cases
            cases[label] = child
            return put(wf, p, branch(check, cases))
        return insert_at(wf, p, branch(check, {label: child}))
    if op == "set_param":
        return put(wf, p, get(wf, p).with_params(**{a[0]: a[1]}))
    if op == "unwrap":
        return _splice(wf, p, list(get(wf, p).children))
    raise ValueError(op)


def _exists(wf, p) -> bool:
    try:
        get(wf, p)
        return True
    except (IndexError, AttributeError):
        return False


# ------------------------------------------------------------------ enumerate


def raw_edits(wf: Block, limits: Limits, allowed_steps=None, allowed_fragments=None) -> list:
    allowed_steps = set(allowed_steps or (set(EDIT_SKILLS) | set(TOOL_STEPS)))
    allowed_fragments = set(FRAGMENTS) if allowed_fragments is None else set(allowed_fragments)
    out: list = []
    nodes = list(iter_nodes(wf))
    insert_pos = []
    for p, n in nodes:
        if isinstance(n, Block) and n.kind != "branch":
            insert_pos += [p + (i,) for i in range(len(n.children) + 1)]
    for pos in insert_pos:
        for name in sorted(allowed_steps):
            out.append(Edit("insert_step", pos, (make_step(name),)))
        for f in ("repair_loop", "sim_improve", "split_by_status"):
            if f in allowed_fragments:
                out.append(Fragment(f, pos))
        for check, (_, labels) in CHECK_SIGS.items():
            for lab in labels:
                for ch in BRANCH_CHILDREN:
                    out.append(Edit("add_branch", pos, (check, lab, ch)))
    for p, n in nodes:
        if not p:
            continue
        parent = get(wf, p[:-1])
        out.append(Edit("remove", p))
        if isinstance(n, Step) and n.kind == "llm":
            for s in sorted(set(SKILL_SIGS) & allowed_steps - {n.name}):
                out.append(Edit("replace_skill", p, (s,)))
            if n.name == "draft":
                if "stats_first" in allowed_fragments:
                    out.append(Fragment("stats_first", p))
                if "multi_draft" in allowed_fragments and parent.kind != "fanout":
                    out.append(Fragment("multi_draft", p))
        if parent.kind != "branch" or isinstance(n, Block):
            for k in limits.k_values:
                for stop in ("legal", "best"):
                    out.append(Edit("wrap_loop", p, (k, stop)))
            for m in limits.n_values:
                out.append(Edit("wrap_fanout", p, (m,)))
        if isinstance(n, Block):
            if n.kind == "loop":
                for k in limits.k_values:
                    if k != n.params["k"]:
                        out.append(Edit("set_param", p, ("k", k)))
                out.append(Edit("set_param", p, ("stop", "best" if n.params["stop"] == "legal" else "legal")))
                out.append(Edit("unwrap", p))
            if n.kind == "fanout":
                for m in limits.n_values:
                    if m != n.params["n"]:
                        out.append(Edit("set_param", p, ("n", m)))
                out.append(Edit("unwrap", p))
            if n.kind == "branch":
                check = n.params["check"]
                for lab in CHECK_SIGS[check][1]:
                    if lab not in n.params["labels"]:
                        for ch in BRANCH_CHILDREN:
                            out.append(Edit("add_branch", p, (check, lab, ch)))
    return out


def _uses_only(node: Block, allowed_steps) -> bool:
    return all(n.name in allowed_steps for _, n in iter_nodes(node) if isinstance(n, Step))


def legal_edits(wf: Block, limits: Limits = Limits(), context: Callable[[Block], Block] | None = None,
                allowed_steps=None, allowed_fragments=None) -> list:
    """All edits whose result is a legal workflow. `context(frag)` maps the object being edited
    to the full workflow that is validated (identity when editing a whole workflow)."""
    context = context or (lambda x: x)
    allowed = set(allowed_steps or (set(EDIT_SKILLS) | set(TOOL_STEPS)))
    seen, out = {to_str(wf)}, []
    for e in raw_edits(wf, limits, allowed, allowed_fragments):
        try:
            new = apply_edit(wf, e)
        except (IndexError, AssertionError, KeyError, TypeError):
            continue
        key = to_str(new)
        if key in seen:
            continue
        if not _uses_only(new, allowed | {n.name for _, n in iter_nodes(wf) if isinstance(n, Step)}):
            continue
        if validate(context(new), limits):
            continue
        seen.add(key)
        out.append(e)
    return out
