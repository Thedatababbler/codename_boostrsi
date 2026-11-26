"""Round 4: loop attribute, fresh samples for duplicate members, macro slot, call metadata, thinking strip."""
import itertools
import math

import pytest

from boosttopo.config import load_config
from boosttopo.evalx import Evaluator
from boosttopo.graph import (IN, LOOP_EDIT, LOOP_REP_STRIDE, Edit, Limits, MacroEdit, Node, Subgraph, apply_edit,
                             legal_edits)
from boosttopo.infer import SEED_STRIDE, hand_topology, infer, repair_loop, writer_role
from boosttopo.llm import CachedLLM, FakeLLM, OpenAICompatLLM, strip_thinking
from boosttopo.search.grow import reserve_slots

from conftest import toy_cfg

LOOP_LIMITS = Limits(3, 2, loop_ks=(2, 4, 8), loop_stops=("until_legal", "keep_best"))


def writer_then_loop(k, stop="keep_best"):
    return Subgraph((Node("n0", "writer"), Node("n1", "repairer", k, stop)), ((IN, "n0"), ("n0", "n1")), "n1")


# ---------------------------------------------------------------- representation
def test_loop_free_keys_unchanged():
    """Keys recorded with the pre-round-4 code (cache keys of every earlier stage depend on them)."""
    assert hand_topology().key() == "4dc5ca9c8fa85377"
    assert writer_role().key() == "b373ce514d8c8cf8"
    h = Subgraph.from_dict({"nodes": [["n0", "repairer"], ["n1", "repairer"], ["n2", "repairer"]],
                            "edges": [["IN", "n0"], ["n0", "n1"], ["IN", "n1"], ["n1", "n2"], ["IN", "n2"]],
                            "out": "n2"})
    assert h.key() == "0451a9ec0d9be63f"
    assert h.to_dict()["nodes"][0] == ["n0", "repairer"]  # no loop fields written for loop-free nodes


def test_loop_roundtrip_and_distinct_keys():
    h = writer_then_loop(4, "until_legal")
    assert Subgraph.from_dict(h.to_dict()) == h
    assert Subgraph.from_describe(h.describe()).key() == h.key()
    assert "repairer*4until_legal" in h.describe()
    keys = {writer_then_loop(k, s).key() for k in (2, 4) for s in ("until_legal", "keep_best")}
    keys.add(Subgraph((Node("n0", "writer"), Node("n1", "repairer")), ((IN, "n0"), ("n0", "n1")), "n1").key())
    assert len(keys) == 5


def test_loop_edit_only_when_enabled():
    h = hand_topology()
    assert not any(e.kind == LOOP_EDIT for e in legal_edits(h, Limits()))
    edits = [e for e in legal_edits(h, LOOP_LIMITS) if e.kind == LOOP_EDIT]
    # two writer-type nodes x 3 k x 2 stop rules; never on the analyst
    assert len(edits) == 12 and {e.a["nid"] for e in edits} == {"n1", "n2"}
    h2 = apply_edit(h, edits[0], LOOP_LIMITS)
    clear = [e for e in legal_edits(h2, LOOP_LIMITS) if e.kind == LOOP_EDIT and e.a["k"] == 1]
    assert len(clear) == 1 and apply_edit(h2, clear[0], LOOP_LIMITS).key() == h.key()
    with pytest.raises(Exception):
        apply_edit(h, Edit.make(LOOP_EDIT, nid="n0", k=2, stop="keep_best"), LOOP_LIMITS)  # analyst
    # change_role keeps the loop
    h3 = apply_edit(writer_then_loop(2), Edit.make("change_role", nid="n1", role="writer"), LOOP_LIMITS)
    assert h3.node("n1") == Node("n1", "writer", 2, "keep_best")


def test_random_subspace_kinds_unchanged():
    from boosttopo.graph import EDIT_KINDS
    assert EDIT_KINDS == ("add_node", "add_reader", "remove_node", "add_edge", "remove_edge", "change_role")


# ---------------------------------------------------------------- execution
def _ev(tmp_path, **over):
    cfg = toy_cfg(tmp_path, **over)
    from boosttopo.run import build, make_splits
    bench, llm, ev, log = build(cfg, None)
    sp = make_splits(cfg, bench)
    ev.make_start(sp["heldout"] + sp["G1"])
    return cfg, llm, ev, sp


@pytest.mark.parametrize("k", [2, 4, 8])
def test_keep_best_loop_reproduces_repair_loop(tmp_path, k):
    """[IN->writer, writer->repairer*k keep_best] = the repair-loop baseline with k repairs (no token cap):
    same plans, scores, tokens and scorer calls per (item, repeat)."""
    cfg, llm, ev, sp = _ev(tmp_path, eval={"n_repeats": 2})
    ev.rep_seed = 1 * SEED_STRIDE
    items = sp["heldout"] + sp["G1"]
    a = infer("gbdt", [writer_then_loop(k)], items, ev, 2)
    b = repair_loop(items, ev, budget=math.inf, n_repeats=2, max_repairs=k)
    key = lambda r: (r["item_id"], r["rep"])  # noqa: E731
    ra, rb = sorted(a["rows"], key=key), sorted(b["rows"], key=key)
    for x, y in zip(ra, rb):
        assert (x["g"], x["score"], x["tokens"], x["scorer_calls"], x["valid"]) == \
               (y["g"], y["score"], y["tokens"], y["scorer_calls"], y["valid"]), (x, y)
    assert any(y["n_repairs"] > 0 for y in rb)


def test_until_legal_stops_at_first_legal_and_counts_every_repetition(tmp_path):
    cfg, llm, ev, sp = _ev(tmp_path)
    it = sp["heldout"][0]
    calls = []
    orig = llm.inner.rule

    def rule(req):  # repairer output legal only from repetition 3 on
        calls.append(req.rep)
        if req.role == "repairer":
            j = req.rep // LOOP_REP_STRIDE
            return "not json" if j < 3 else orig(req)
        return orig(req)

    llm.inner.rule = rule
    h = Subgraph((Node("n0", "repairer", 8, "until_legal"),), ((IN, "n0"),), "n0")
    from boosttopo.graph import run_subgraph
    y_bad = "not json"
    so = run_subgraph(h, it, y_bad, llm, ev.bench, rep=0)
    reps = [r for r in calls]
    assert len(reps) == 3  # stopped after the first legal plan
    assert so.scorer_calls == 1 + 2 * 3
    assert so.cost == sum(t["tokens"] for t in so.trace) and len(so.trace) == 3
    calls.clear()
    so2 = run_subgraph(Subgraph((Node("n0", "repairer", 8, "keep_best"),), ((IN, "n0"),), "n0"),
                       it, y_bad, llm, ev.bench, rep=0)
    assert len(so2.trace) == 8 or so2.score >= 1.0  # keep_best runs all k unless the score reaches 1
    assert so2.scorer_calls == 1 + 2 * len(so2.trace)


def test_loop_free_stage_unchanged(tmp_path):
    """No loop: same requests, tokens and scorer calls as before (repairer diagnosis + gate)."""
    cfg, llm, ev, sp = _ev(tmp_path)
    res = infer("adaboost", [hand_topology()], sp["heldout"], ev, 1)
    assert res["scorer_calls_per_item"] == 1 + 2


# ---------------------------------------------------------------- duplicate members
def test_duplicate_member_draws_fresh_samples(tmp_path):
    h = Subgraph((Node("n0", "repairer"),), ((IN, "n0"),), "n0")
    for dedup, expect_dup in ((False, True), (True, False)):
        cfg, llm, ev, sp = _ev(tmp_path / str(dedup), fresh_duplicate_stages=dedup)
        assert ev.dedup_stages is dedup
        reps = []
        orig = llm.inner.rule
        llm.inner.rule = lambda req, _o=orig: (reps.append(req.rep), _o(req))[1]
        res = infer("adaboost", [h, h], sp["heldout"], ev, 1)
        dup = sum(r["dup_stage_runs"] for r in res["rows"])
        assert (dup > 0) is expect_dup
        if dedup:
            assert Evaluator.DUP_STRIDE in reps and 0 in reps
        else:
            assert Evaluator.DUP_STRIDE not in reps


def test_non_duplicates_keep_occurrence_zero(tmp_path):
    cfg, llm, ev, sp = _ev(tmp_path)
    a, b = hand_topology(), writer_role()
    assert [ev.occurrence(x, [a, b, a][:i]) for i, x in enumerate([a, b, a])] == [0, 0, 1]
    ev.dedup_stages = False
    assert ev.occurrence(a, [a, a]) == 0


def test_repeat_offsets_never_collide():
    """Every combination of the repeat-index shifts maps to a distinct index (within one run seed)."""
    seen = {}
    for r, rej, occ, j, fresh in itertools.product(range(4), range(12), range(6), range(0, 9), (0, 1)):
        v = r + rej * 1000 + occ * Evaluator.DUP_STRIDE + j * LOOP_REP_STRIDE + fresh * Evaluator.FRESH_OFFSET
        assert v < SEED_STRIDE
        assert seen.setdefault(v, (r, rej, occ, j, fresh)) == (r, rej, occ, j, fresh)


# ---------------------------------------------------------------- search plumbing
def test_reserve_slots():
    cfg = load_config(None, {"search": {"top_k": 3}})
    macro = MacroEdit((Edit.make("add_reader", role="analyst", nid="n9", src=IN, dst="n0"),), "analyze_first")
    loop = Edit.make(LOOP_EDIT, nid="n0", k=2, stop="keep_best")
    ranked = [Edit.make("remove_node", nid=f"n{i}") for i in range(5)] + [loop, macro]
    assert reserve_slots(ranked, cfg) == ranked[:3]  # both switches off
    cfg.search.macro_slot = True
    assert reserve_slots(ranked, cfg) == ranked[:2] + [macro]
    cfg.loop.enabled = True
    top = reserve_slots(ranked, cfg)
    assert top[0] == ranked[0] and set(top[1:]) == {macro, loop}


def test_atomic_proposer_emits_macros_only_with_switch(tmp_path):
    from boosttopo.search.proposers import make_proposer

    class C:
        limits = Limits()
    h = writer_role()
    off = make_proposer(load_config(None, {}))
    on = make_proposer(load_config(None, {"search": {"macro_slot": True}}))
    assert not any(isinstance(e, MacroEdit) for e in off.propose(h, C))
    assert any(isinstance(e, MacroEdit) for e in on.propose(h, C))


def test_loop_limits_follow_config(tmp_path):
    from boosttopo.boost.common import Ctx
    off = Ctx(load_config(None, {}), None, None, None, None, None, None).limits
    on = Ctx(load_config(None, {"loop": {"enabled": True}}), None, None, None, None, None, None).limits
    assert off.loop_ks == () and on.loop_ks == (2, 4, 8) and on.loop_stops == ("until_legal", "keep_best")


# ---------------------------------------------------------------- LLM layer
def test_cache_key_has_model_and_thinking(tmp_path):
    def mk(model, think):
        inner = OpenAICompatLLM("http://x", model, max_tokens=4096,
                                extra_body={"chat_template_kwargs": {"enable_thinking": think}})
        return CachedLLM(inner, tmp_path)
    from boosttopo.llm import Request
    req = Request(prompt="p", rep=0)
    keys = {mk(m, t).key(req) for m in ("qwen3-14b", "other") for t in (False, True)}
    assert len(keys) == 4
    assert mk("qwen3-14b", False).name == "qwen3-14b"  # earlier non-thinking keys unchanged


def test_strip_thinking_and_meta(tmp_path):
    assert strip_thinking("plain") == ("plain", 0)
    assert strip_thinking("<think>\nabc\n</think>\n\n{\"a\": 1}") == ('{"a": 1}', len("<think>\nabc\n</think>"))
    assert strip_thinking("<think> truncated reasoning")[0] == ""
    from boosttopo.llm import Request
    llm = CachedLLM(FakeLLM(lambda r: "<think>x</think>ans"), tmp_path)
    for _ in range(2):  # miss, then hit: same stripped answer, raw text kept in the record
        text, tok, meta = llm.call_meta(Request(prompt="q"))
        assert text == "ans" and meta["thinking_chars"] > 0
    rec = next((tmp_path / "llm").rglob("*.json")).read_text()
    assert "<think>" in rec


# ---------------------------------------------------------------- round 4 Stage A/B plumbing
def test_one_per_kind_slots():
    cfg = load_config(None, {"search": {"slot_mode": "one_per_kind"}, "loop": {"enabled": True}})
    macro = MacroEdit((Edit.make("add_reader", role="analyst", nid="n9", src=IN, dst="n0"),), "analyze_first")
    loops = [Edit.make(LOOP_EDIT, nid="n0", k=k, stop=s) for k in (2, 4, 8) for s in ("keep_best", "until_legal")]
    atom = [Edit.make("remove_node", nid=f"n{i}") for i in range(3)]
    ranked = [macro] + loops + atom
    top = reserve_slots(ranked, cfg)
    assert top == [macro, loops[0], atom[0]]
    cfg.loop.slot_pick = "random"
    picks = {str(next(e for e in reserve_slots(ranked, cfg, seed=s) if e.kind == LOOP_EDIT)) for s in range(40)}
    assert len(picks) > 3  # the loop slot explores settings instead of always the string-first one
    assert reserve_slots(ranked, cfg, seed=5) == reserve_slots(ranked, cfg, seed=5)
    assert reserve_slots(atom, cfg) == [atom[0]]  # missing kinds are simply absent


def test_cache_tag_and_y0_reuse(tmp_path):
    from boosttopo.llm import Request

    calls = []
    inner = FakeLLM(lambda r: (calls.append(r.prompt), f"out:{r.prompt}")[1])
    inner.max_tokens, inner.extra_body = 30000, {"chat_template_kwargs": {"enable_thinking": True}}
    old = CachedLLM(inner, tmp_path)
    old.call(Request(prompt="y0-a"))
    new = CachedLLM(inner, tmp_path, tag="kvfp8", y0_tags=(None,))
    assert new.name != old.name and new.y0_names == [old.name]
    n = len(calls)
    assert new.call_y0(Request(prompt="y0-a"))[0] == "out:y0-a" and len(calls) == n  # reused, no call
    new.call(Request(prompt="y0-a"))  # a non-y0 call with the same prompt is regenerated under the new tag
    assert len(calls) == n + 1
    new.call_y0(Request(prompt="y0-b"))  # no old record: generated (and cached) under the new name
    assert len(calls) == n + 2 and new.call(Request(prompt="y0-b"))[0] == "out:y0-b" and len(calls) == n + 2
