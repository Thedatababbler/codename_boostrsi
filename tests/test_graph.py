from boosttopo.graph import (IN, EDIT_KINDS, Edit, Limits, MacroEdit, Node, Subgraph, apply_edit, expand_macro,
                             is_valid, legal_edits, run_subgraph, single_writer, topo_order, try_apply)
from boosttopo.llm import FakeLLM
from boosttopo.bench.toy import ToyBench, toy_rule, toy_y0

L = Limits()


def test_legal_edits_cover_six_kinds_and_are_valid():
    h = single_writer()
    seen = set()
    frontier = [h]
    for _ in range(2):
        nxt = []
        for g in frontier:
            for e in legal_edits(g, L):
                h2 = apply_edit(g, e, L)
                assert is_valid(h2, L)
                assert len(h2.nodes) <= 3
                seen.add(e.kind)
                nxt.append(h2)
        frontier = nxt[:30]
    assert seen == set(EDIT_KINDS)


def test_illegal_edits_rejected():
    h = single_writer()
    assert try_apply(h, Edit.make("remove_node", nid="n0"), L) is None  # cannot remove the only node
    assert try_apply(h, Edit.make("change_role", nid="n0", role="critic"), L) is None  # output must be a writer
    assert try_apply(h, Edit.make("remove_edge", u=IN, v="n0"), L) is None  # unreachable output
    three = apply_edit(h, expand_macro("critique_and_rewrite", h, L)[0], L)
    assert len(three.nodes) == 3
    assert all(try_apply(three, e, L) is None for e in
               [Edit.make("add_node", role="writer", nid="n9", srcs=(three.out,))])  # node limit
    cyc = Subgraph((Node("a", "writer"), Node("b", "writer")), ((IN, "a"), ("a", "b"), ("b", "a")), "b")
    assert not is_valid(cyc, L)
    dead = Subgraph((Node("a", "writer"), Node("c", "critic")), ((IN, "a"), (IN, "c")), "a")
    assert not is_valid(dead, L)
    deg = Subgraph((Node("a", "analyst"), Node("c", "critic"), Node("w", "writer")),
                   ((IN, "a"), (IN, "c"), ("a", "w"), ("c", "w"), (IN, "w")), "w")
    assert not is_valid(deg, L)


def test_canonical_key_ignores_ids():
    a = Subgraph((Node("x", "analyst"), Node("y", "writer")), ((IN, "x"), (IN, "y"), ("x", "y")), "y")
    b = Subgraph((Node("n1", "writer"), Node("n0", "analyst")), (("n0", "n1"), (IN, "n0"), (IN, "n1")), "n1")
    assert a.key() == b.key()


def test_macros_expand():
    h = single_writer()
    for name in ("critique_and_rewrite", "analyze_first", "repair_tail"):
        ms = expand_macro(name, h, L)
        assert ms and all(isinstance(m, MacroEdit) for m in ms)
        for m in ms:
            assert is_valid(apply_edit(h, m, L), L)


def test_execution_order_is_topological_and_gated():
    h = apply_edit(single_writer(), expand_macro("repair_tail", single_writer(), L)[0], L)
    h = apply_edit(h, Edit.make("add_reader", role="analyst", nid="n2", src=IN, dst="n0"), L)
    order = [n.nid for n in topo_order(h)]
    pos = {n: i for i, n in enumerate(order)}
    for u, v in h.edges:
        if u != IN:
            assert pos[u] < pos[v]
    bench = ToyBench()
    it = bench.make_items("toy", 1, [1])[0]  # broken item
    calls = []
    fake = FakeLLM(lambda r: (calls.append(r.role), toy_rule(r))[1])
    y0 = toy_y0(it)
    out = run_subgraph(h, it, y0, fake, bench)
    assert calls == [h.role(n) for n in order]
    assert out.score >= bench.score(it, y0).score
    # a subgraph that makes things worse is gated off
    worse = FakeLLM(lambda r: "[-5,0,0,0,0,0]")
    it2 = bench.make_items("toy", 1, [0])[0]
    o2 = run_subgraph(single_writer(), it2, toy_y0(it2), worse, bench)
    assert o2.solution == toy_y0(it2) and o2.raw_score == 0.0
