"""Multi-agent methods written as workflows, for cache replay (seed_scheme legacy, diag_mode original).

- writer_once      = boosttopo infer(adaboost, [writer_role])             -> seq[draft]
- best_of_n_writer = boosttopo best_of_n_subgraph(writer_role, n)          -> seq[fanout(n)[draft]]
- repair_loop(B)   = boosttopo infer._repair_loop_one (token budget B)    -> seq[draft, loop(k, best, budget)[validate, repair]]
- a GBDT ensemble  = boosttopo infer(gbdt, members)                       -> seq[gate[member 0], gate[member 1], ...]

A member (graph.Subgraph) is translated node by node in topological order:
writer -> draft; repairer -> validate, repair; analyst -> analyze; critic -> critique; a loop node (k, stop) ->
loop(k, stop)[draft] or loop(k, stop)[validate, repair] (until_legal -> legal, keep_best -> best).

A member can be expressed only if it has one plan lineage that the shared state can carry:
1. the writer-type nodes form a chain IN -> w1 -> ... -> out (each writer's plan input is the previous writer,
   the first one's is IN), so no writer reads a second plan ("ALTERNATIVE PLAN FROM ...");
2. every reader (analyst / critic) feeds exactly one writer, and that writer is the first writer after the
   reader in the chain; a critic reads the plan that writer reads (an analyst's prompt has no plan);
3. a reader's notes are seen only by its own writer: no later writer of the same member may follow it,
   since state notes persist until the end of the gate.
Anything else raises NotExpressible with the reason.
"""
from __future__ import annotations

import math

from boosttopo.graph import IN, Subgraph, topo_order

from .model import Block, fanout, gate, loop, make_step, seq

STEP_OF = {"writer": ("draft",), "repairer": ("validate", "repair"), "analyst": ("analyze",), "critic": ("critique",)}
STOP_OF = {"until_legal": "legal", "keep_best": "best"}
WRITERS = ("writer", "repairer")


class NotExpressible(ValueError):
    pass


def member_steps(h: Subgraph) -> list:
    h = h.canonical()  # node ids as in describe()
    order = topo_order(h)
    rank = {IN: -1, **{n.nid: i for i, n in enumerate(order)}}
    role = {n.nid: n.role for n in order}
    writers = [n.nid for n in order if n.role in WRITERS]
    if not writers or writers[-1] != h.out:
        raise NotExpressible(f"{h.describe()}: the output is not the last writer")
    prev = IN
    owner: dict[str, str] = {}
    for w in writers:
        srcs = sorted(h.preds(w), key=rank.get)
        wsrc = [u for u in srcs if u == IN or role.get(u) in WRITERS]
        if wsrc != [prev]:
            raise NotExpressible(f"{h.describe()}: writer {w} reads plans {wsrc}, not just {prev} "
                                 "(a second plan cannot be carried in the state)")
        for u in srcs:
            if u not in wsrc:
                if u in owner:
                    raise NotExpressible(f"{h.describe()}: reader {u} feeds two writers")
                owner[u] = w
        prev = w
    out: list = []
    pending_readers: dict[str, list] = {}
    for n in order:
        if n.role in WRITERS:
            continue
        dsts = [v for v in h.succs(n.nid)]
        if len(dsts) != 1 or owner.get(n.nid) != dsts[0]:
            raise NotExpressible(f"{h.describe()}: reader {n.nid} must feed exactly one writer")
        dst = dsts[0]
        plan_src = [u for u in h.preds(dst) if u == IN or role.get(u) in WRITERS][0]
        if n.role == "critic" and h.preds(n.nid) != [plan_src]:
            raise NotExpressible(f"{h.describe()}: critic {n.nid} reads {h.preds(n.nid)}, its writer reads {plan_src}")
        if dst != writers[-1]:
            raise NotExpressible(f"{h.describe()}: reader {n.nid}'s notes would also reach later writers")
        pending_readers.setdefault(dst, []).append(n)
    for w in writers:
        node = h.node(w)
        for r in sorted(pending_readers.get(w, []), key=lambda n: rank[n.nid]):
            out += [make_step(s) for s in STEP_OF[r.role]]
        steps = [make_step(s) for s in STEP_OF[node.role]]
        if node.k > 1:
            out.append(loop(*steps, k=node.k, stop=STOP_OF[node.stop]))
        else:
            out += steps
    return out


def member_workflow(h: Subgraph, occ: int = 0) -> Block:
    return gate(*member_steps(h), occ=occ)


def ensemble_workflow(members: list[Subgraph], fresh_duplicate_stages: bool = True) -> Block:
    """GBDT inference chain: each member gated on the previous output, skipped once g = 1."""
    out = []
    for i, h in enumerate(members):
        occ = sum(1 for e in members[:i] if e.key() == h.key()) if fresh_duplicate_stages else 0
        out.append(member_workflow(h, occ))
    return seq(*out)


def writer_once() -> Block:
    return seq(make_step("draft"))


def best_of_n_writer(n: int) -> Block:
    return seq(fanout(make_step("draft"), n=n))


def repair_loop(token_budget: float, max_repairs: int = 40) -> Block:
    """Integer budget b with `tokens < b` <=> `tokens < token_budget` for integer token counts."""
    b = int(math.ceil(token_budget)) if token_budget != int(token_budget) else int(token_budget)
    return seq(make_step("draft"), loop(make_step("validate"), make_step("repair"), k=max_repairs, stop="best",
                                        budget=b))


def repair_loop_k(k: int) -> Block:
    """Stage A's repair loop: at most k repairs, no token cap, stops early at g = 1."""
    return seq(make_step("draft"), loop(make_step("validate"), make_step("repair"), k=k, stop="best"))
