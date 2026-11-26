"""Subgraphs, atomic and macro edits, and the gated executor (doc section 5)."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Union

from ..roles import READERS, ROLES, WRITERS, is_writer, role_prompt

IN = "IN"
EDIT_KINDS = ("add_node", "add_reader", "remove_node", "add_edge", "remove_edge", "change_role")
# Loop attribute (round 4, off by default): a writer-type node with k > 1 repeats its role up to k times on the
# best plan of its own lineage. The edit that sets it is not in EDIT_KINDS, so the random-subspace proposer
# samples exactly as before when loops are off.
LOOP_EDIT = "set_loop"
LOOP_STOPS = ("until_legal", "keep_best")  # stop once a legal plan exists | run all k, stop early only at g = 1
LOOP_REP_STRIDE = 7919  # repetition j (1..k) uses repeat index rep + j * stride (= the repair-loop baseline)


@dataclass(frozen=True)
class Node:
    nid: str
    role: str
    k: int = 1  # 1 = no loop
    stop: str = ""  # one of LOOP_STOPS when k > 1

    def loop_label(self) -> str:
        return f"*{self.k}{self.stop}" if self.k > 1 else ""


@dataclass(frozen=True)
class Limits:
    max_nodes: int = 3
    max_indegree: int = 2
    loop_ks: tuple = ()  # allowed k values for the loop edit; () = loop edits are never proposed
    loop_stops: tuple = ()


@dataclass(frozen=True)
class Subgraph:
    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...]
    out: str

    # ---- structure helpers
    def role(self, nid: str) -> str:
        return next(n.role for n in self.nodes if n.nid == nid)

    def ids(self) -> list[str]:
        return [n.nid for n in self.nodes]

    def preds(self, nid: str) -> list[str]:
        return [u for (u, v) in self.edges if v == nid]

    def succs(self, nid: str) -> list[str]:
        return [v for (u, v) in self.edges if u == nid]

    def _canon(self) -> tuple:
        ids = self.ids()
        best = None
        for perm in itertools.permutations(range(len(ids))):
            m = {ids[i]: f"n{perm[i]}" for i in range(len(ids))}
            m[IN] = IN
            # a node without a loop keeps the 2-tuple, so loop-free subgraphs keep their earlier keys
            roles = tuple(sorted((m[n.nid], n.role) + ((n.k, n.stop) if n.k > 1 else ()) for n in self.nodes))
            edges = tuple(sorted((m[u], m[v]) for u, v in self.edges))
            cand = (roles, edges, m[self.out])
            if best is None or cand < best:
                best = cand
        return best

    def key(self) -> str:
        return hashlib.sha256(json.dumps(self._canon()).encode()).hexdigest()[:16]

    def canonical(self) -> "Subgraph":
        roles, edges, out = self._canon()
        return Subgraph(tuple(Node(*x) for x in roles), edges, out)

    def to_dict(self) -> dict:
        return {"nodes": [[n.nid, n.role] + ([n.k, n.stop] if n.k > 1 else []) for n in self.nodes],
                "edges": [list(e) for e in self.edges], "out": self.out}

    @staticmethod
    def from_dict(d: dict) -> "Subgraph":
        return Subgraph(tuple(Node(*x) for x in d["nodes"]), tuple(tuple(e) for e in d["edges"]), d["out"])

    def node(self, nid: str) -> Node:
        return next(n for n in self.nodes if n.nid == nid)

    def loops(self) -> list[tuple[str, int, str]]:
        return [(n.role, n.k, n.stop) for n in self.nodes if n.k > 1]

    @staticmethod
    def from_describe(text: str) -> "Subgraph":
        """Inverse of describe(): '[IN->n0:writer, n0:writer->n1:repairer] out=n1:repairer'."""
        body, _, out = text.rpartition(" out=")
        nodes, edges = {}, []
        for e in body.strip()[1:-1].split(", "):
            u, v = e.split("->")
            for lab in (u, v):
                if lab != IN:
                    nid, role = lab.split(":")
                    role, star, loop = role.partition("*")
                    if star:
                        k = int("".join(c for c in loop if c.isdigit()))
                        nodes[nid] = Node(nid, role, k, loop[len(str(k)):])
                    else:
                        nodes[nid] = Node(nid, role)
            edges.append((u.split(":")[0], v.split(":")[0]))
        return Subgraph(tuple(nodes[n] for n in sorted(nodes)), tuple(edges), out.split(":")[0])

    def describe(self) -> str:
        """Human-readable form, e.g. 'IN->n0:writer; n0->n1:repairer; IN->n1 | out=n1'."""
        c = self.canonical()
        lab = {n.nid: f"{n.nid}:{n.role}{n.loop_label()}" for n in c.nodes}
        lab[IN] = IN
        es = ", ".join(f"{lab[u]}->{lab[v]}" for u, v in c.edges)
        return f"[{es}] out={lab[c.out]}"

    def role_counts(self) -> dict:
        d = {r: 0 for r in ROLES}
        for n in self.nodes:
            d[n.role] += 1
        return d

    def depth(self) -> int:
        order = topo_order(self)
        dep = {IN: 0}
        for n in order:
            dep[n.nid] = 1 + max(dep[u] for u in self.preds(n.nid))
        return max(dep[n.nid] for n in self.nodes)


def single_writer() -> Subgraph:
    return Subgraph((Node("n0", "writer"),), ((IN, "n0"),), "n0")


def topo_order(h: Subgraph) -> list[Node]:
    indeg = {n.nid: 0 for n in h.nodes}
    for u, v in h.edges:
        if u != IN:
            indeg[v] += 1
    ready = sorted([nid for nid, d in indeg.items() if d == 0])
    order = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for v in sorted(h.succs(nid)):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
        ready.sort()
    if len(order) != len(h.nodes):
        raise ValueError("cycle")
    by = {n.nid: n for n in h.nodes}
    return [by[i] for i in order]


def is_valid(h: Subgraph, limits: Limits) -> bool:
    ids = h.ids()
    if not (1 <= len(ids) <= limits.max_nodes) or len(set(ids)) != len(ids) or IN in ids:
        return False
    if any(n.role not in ROLES for n in h.nodes):
        return False
    for n in h.nodes:  # loops: writer-type nodes only, with a known stop rule
        if n.k < 1 or (n.k > 1 and (not is_writer(n.role) or n.stop not in LOOP_STOPS)) or (n.k == 1 and n.stop):
            return False
    if h.out not in ids or not is_writer(h.role(h.out)):
        return False
    if len(set(h.edges)) != len(h.edges):
        return False
    for u, v in h.edges:
        if v not in ids or (u != IN and u not in ids) or u == v:
            return False
    for nid in ids:
        if not (1 <= len(h.preds(nid)) <= limits.max_indegree):
            return False
    try:
        topo_order(h)
    except ValueError:
        return False
    # every node must reach the output (no dead nodes); out must be reachable from IN (implied
    # by indegree>=1 and acyclicity)
    reach = {h.out}
    changed = True
    while changed:
        changed = False
        for u, v in h.edges:
            if v in reach and u != IN and u not in reach:
                reach.add(u)
                changed = True
    return reach == set(ids)


@dataclass(frozen=True)
class Edit:
    kind: str
    args: tuple  # sorted (key, value) pairs; use .a for dict view

    @staticmethod
    def make(kind: str, **args) -> "Edit":
        return Edit(kind, tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in args.items())))

    @property
    def a(self) -> dict:
        return dict(self.args)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "args": {k: list(v) if isinstance(v, tuple) else v for k, v in self.args}}

    def __str__(self) -> str:
        return f"{self.kind}({', '.join(f'{k}={v}' for k, v in self.args)})"


@dataclass(frozen=True)
class MacroEdit:
    steps: tuple[Edit, ...]
    name: str = ""

    @property
    def kind(self) -> str:
        return "macro"

    def to_dict(self) -> dict:
        return {"kind": "macro", "name": self.name, "steps": [s.to_dict() for s in self.steps]}

    def __str__(self) -> str:
        return f"macro:{self.name}[{'; '.join(map(str, self.steps))}]"


AnyEdit = Union[Edit, MacroEdit]


class IllegalEdit(ValueError):
    pass


def _next_id(h: Subgraph) -> str:
    used = {n.nid for n in h.nodes}
    i = 0
    while f"n{i}" in used:
        i += 1
    return f"n{i}"


def _apply_atomic(h: Subgraph, e: Edit) -> Subgraph:
    a = e.a
    nodes, edges, out = list(h.nodes), list(h.edges), h.out
    ids = h.ids()
    if e.kind == "add_node":
        if a["role"] not in WRITERS:
            raise IllegalEdit("add_node needs a writer role")
        nid = a["nid"]
        if nid in ids:
            raise IllegalEdit("id exists")
        nodes.append(Node(nid, a["role"]))
        edges += [(s, nid) for s in a["srcs"]]
        out = nid
    elif e.kind == "add_reader":
        if a["role"] not in READERS:
            raise IllegalEdit("add_reader needs a reader role")
        nid = a["nid"]
        if nid in ids or a["dst"] not in ids:
            raise IllegalEdit("bad ids")
        nodes.append(Node(nid, a["role"]))
        edges += [(a["src"], nid), (nid, a["dst"])]
    elif e.kind == "remove_node":
        x = a["nid"]
        if x not in ids or len(ids) == 1:
            raise IllegalEdit("cannot remove")
        ps, ss = h.preds(x), h.succs(x)
        nodes = [n for n in nodes if n.nid != x]
        edges = [(u, v) for u, v in edges if x not in (u, v)]
        for p in ps:
            for s in ss:
                if (p, s) not in edges:
                    edges.append((p, s))
        if x == out:
            wp = [p for p in ps if p != IN and is_writer(h.role(p))]
            if not wp:
                raise IllegalEdit("no writer predecessor to become output")
            order = [n.nid for n in topo_order(h)]
            out = max(wp, key=order.index)
    elif e.kind == "add_edge":
        if (a["u"], a["v"]) in edges:
            raise IllegalEdit("edge exists")
        edges.append((a["u"], a["v"]))
    elif e.kind == "remove_edge":
        if (a["u"], a["v"]) not in edges:
            raise IllegalEdit("no such edge")
        edges.remove((a["u"], a["v"]))
    elif e.kind == "change_role":
        x, r = a["nid"], a["role"]
        if x not in ids or h.role(x) == r or is_writer(h.role(x)) != is_writer(r):
            raise IllegalEdit("bad role change")
        nodes = [dataclasses.replace(n, role=r) if n.nid == x else n for n in nodes]
    elif e.kind == LOOP_EDIT:
        x, k, stop = a["nid"], int(a["k"]), a.get("stop", "")
        if x not in ids or not is_writer(h.role(x)):
            raise IllegalEdit("loops only on writer-type nodes")
        old = h.node(x)
        new = Node(x, old.role, k, stop if k > 1 else "")
        if new == old:
            raise IllegalEdit("loop unchanged")
        nodes = [new if n.nid == x else n for n in nodes]
    else:
        raise IllegalEdit(f"unknown edit {e.kind}")
    return Subgraph(tuple(nodes), tuple(edges), out)


def apply_edit(h: Subgraph, e: AnyEdit, limits: Limits | None = None) -> Subgraph:
    """Apply an atomic or macro edit; raises IllegalEdit if any step (or the result) is illegal."""
    limits = limits or Limits()
    steps = e.steps if isinstance(e, MacroEdit) else (e,)
    cur = h
    for s in steps:
        cur = _apply_atomic(cur, s)
    if not is_valid(cur, limits):
        raise IllegalEdit("result violates limits")
    return cur


def try_apply(h: Subgraph, e: AnyEdit, limits: Limits) -> Subgraph | None:
    try:
        return apply_edit(h, e, limits)
    except (IllegalEdit, KeyError, StopIteration, ValueError):
        return None


def legal_edits(h: Subgraph, limits: Limits) -> list[Edit]:
    ids = h.ids()
    order = [n.nid for n in topo_order(h)]
    new = _next_id(h)
    cands: list[Edit] = []
    # add_node: a new writer-type output reading the current output (and optionally one more source)
    for r in WRITERS:
        cands.append(Edit.make("add_node", role=r, nid=new, srcs=(h.out,)))
        for x in [IN] + [i for i in ids if i != h.out]:
            cands.append(Edit.make("add_node", role=r, nid=new, srcs=(h.out, x)))
    # add_reader: src in IN or a writer node before dst; dst a writer node
    for r in READERS:
        for dst in ids:
            if not is_writer(h.role(dst)):
                continue
            for src in [IN] + [i for i in order[: order.index(dst)] if is_writer(h.role(i))]:
                cands.append(Edit.make("add_reader", role=r, nid=new, src=src, dst=dst))
    for x in ids:
        cands.append(Edit.make("remove_node", nid=x))
    for u in [IN] + ids:
        for v in ids:
            if u != v and (u, v) not in h.edges:
                cands.append(Edit.make("add_edge", u=u, v=v))
    for u, v in h.edges:
        cands.append(Edit.make("remove_edge", u=u, v=v))
    for n in h.nodes:
        for r in (WRITERS if is_writer(n.role) else READERS):
            if r != n.role:
                cands.append(Edit.make("change_role", nid=n.nid, role=r))
    if limits.loop_ks:  # set / change / clear the loop of a writer-type node
        for n in h.nodes:
            if not is_writer(n.role):
                continue
            for k in limits.loop_ks:
                for stop in limits.loop_stops:
                    cands.append(Edit.make(LOOP_EDIT, nid=n.nid, k=int(k), stop=stop))
            if n.k > 1:
                cands.append(Edit.make(LOOP_EDIT, nid=n.nid, k=1, stop=""))
    seen, out = set(), []
    for e in cands:
        res = try_apply(h, e, limits)
        if res is None or res.key() == h.key():
            continue
        k = (res.key(), e.kind)
        if k in seen:  # different edits, same result: keep one per kind
            continue
        seen.add(k)
        out.append(e)
    return out


MACROS = ("critique_and_rewrite", "analyze_first", "repair_tail")


def expand_macro(name: str, h: Subgraph, limits: Limits) -> list[MacroEdit]:
    new = _next_id(h)
    h_tmp = Subgraph(h.nodes + (Node(new, "writer"),), h.edges, h.out)
    new2 = _next_id(h_tmp)
    outs: list[MacroEdit] = []
    if name == "critique_and_rewrite":
        for src in (IN, h.out):
            outs.append(MacroEdit((Edit.make("add_node", role="writer", nid=new, srcs=(h.out,)),
                                   Edit.make("add_reader", role="critic", nid=new2, src=src, dst=new)), name))
    elif name == "analyze_first":
        outs.append(MacroEdit((Edit.make("add_reader", role="analyst", nid=new, src=IN, dst=h.out),), name))
    elif name == "repair_tail":
        outs.append(MacroEdit((Edit.make("add_node", role="repairer", nid=new, srcs=(h.out,)),
                               Edit.make("add_edge", u=IN, v=new)), name))
    else:
        raise KeyError(name)
    return [m for m in outs if try_apply(h, m, limits) is not None]


def edit_roles(h: Subgraph, e: AnyEdit) -> set[str]:
    """Roles an edit touches (added, removed or changed nodes)."""
    steps = e.steps if isinstance(e, MacroEdit) else (e,)
    roles = set()
    for s in steps:
        a = s.a
        if "role" in a:
            roles.add(a["role"])
        if s.kind in ("remove_node", "change_role") and a.get("nid") in h.ids():
            roles.add(h.role(a["nid"]))
    return roles


def edit_type(e: AnyEdit) -> str:
    return f"macro:{e.name}" if isinstance(e, MacroEdit) else e.kind


def edit_summary(h: Subgraph, e: AnyEdit) -> dict:
    return {"type": edit_type(e), "roles": sorted(edit_roles(h, e)), "is_macro": isinstance(e, MacroEdit),
            "edit": e.to_dict()}


# ---------------------------------------------------------------- execution


@dataclass
class StageOutput:
    solution: str
    score: float
    raw_score: float
    cost: float
    diag: str = ""
    category: str = ""
    trace: list = field(default_factory=list)
    scorer_calls: int = -1  # verifier queries of this stage; -1 = recorded before it was counted (use static)


def static_scorer_calls(h: Subgraph) -> int:
    """Scorer calls of a loop-free stage: one diagnosis per repairer node + one to gate the output.
    (The stage input is always an already-scored solution, so its score is not counted again.)"""
    return h.role_counts().get("repairer", 0) + 1


def run_subgraph(h: Subgraph, item, y_in: str, llm, bench, rep: int = 0, s_in=None, g_of=None) -> StageOutput:
    """Run nodes in topological order; gate the output (adopt only if strictly better).

    A node with a loop (k > 1) runs its role up to k times. Repetition j (1..k) works on the best-scoring plan of
    the node's lineage so far (the latest plan on ties; the lineage starts with the node's current input), with
    repeat index rep + j * LOOP_REP_STRIDE; a repairer gets that plan's diagnosis each time. Before every
    repetition it stops if g(best) = 1 ('keep_best') or if the best plan is legal ('until_legal'). The node's
    output is the best plan of its lineage. Each repetition costs its tokens and two scorer calls for a repairer
    (diagnosis + score of the new plan) or one for a writer. g_of(score) -> g; default: g = 1 iff score >= 1.
    """
    g_of = g_of or (lambda sc: 1.0 if sc >= 1.0 else 0.0)
    msgs = {IN: y_in}
    order = [n.nid for n in topo_order(h)]
    rank = {IN: -1, **{nid: i for i, nid in enumerate(order)}}
    cost = 0
    calls = 1  # the output gate
    trace = []
    for node in topo_order(h):
        srcs = sorted(h.preds(node.nid), key=rank.get)
        writer_srcs = [u for u in srcs if u == IN or is_writer(h.role(u))]
        cur_src = writer_srcs[-1] if writer_srcs else IN
        y_cur = msgs[cur_src]
        upstream = []
        for u in srcs:
            if u == cur_src:
                continue
            urole = "writer" if u == IN else h.role(u)
            upstream.append((urole, msgs[u]))
        if node.k <= 1:
            diag = None
            if node.role == "repairer":
                diag = bench.score(item, y_cur).diag
                calls += 1
            req = role_prompt(node.role, item, y_cur, upstream, diag, rep=rep)
            out, tokens = llm.call(req)
            msgs[node.nid] = out
            cost += tokens
            trace.append({"nid": node.nid, "role": node.role, "tokens": tokens})
            continue
        cur, cur_sr = y_cur, bench.score(item, y_cur)  # the input's score is already known upstream: not counted
        best, best_sr = cur, cur_sr
        for j in range(1, node.k + 1):
            if g_of(best_sr.score) >= 1.0 or (node.stop == "until_legal" and best_sr.valid):
                break
            diag = cur_sr.diag if node.role == "repairer" else None
            out, tokens = llm.call(role_prompt(node.role, item, cur, upstream, diag,
                                               rep=rep + j * LOOP_REP_STRIDE))
            cost += tokens
            calls += 2 if node.role == "repairer" else 1
            sr = bench.score(item, out)
            trace.append({"nid": node.nid, "role": node.role, "tokens": tokens, "rep_j": j, "score": sr.score})
            if sr.score >= cur_sr.score:
                cur, cur_sr = out, sr
            if sr.score > best_sr.score:
                best, best_sr = out, sr
        msgs[node.nid] = best
    y_out = msgs[h.out]
    r_in = s_in if s_in is not None else bench.score(item, y_in)
    r_out = bench.score(item, y_out)
    if r_out.score > r_in.score:
        return StageOutput(y_out, r_out.score, r_out.score, cost, r_out.diag, r_out.category, trace, calls)
    return StageOutput(y_in, r_in.score, r_out.score, cost, r_in.diag, r_in.category, trace, calls)
