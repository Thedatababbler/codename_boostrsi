"""Acceptance (a)/(b) on boosttopo's own toy setup: the multi-agent methods run live (original code), then
(b) the original code replays them from the LLM cache alone, and (a) the same methods written as workflows
(seed_scheme legacy, diag_mode original) replay them with zero new calls and match plan, score, g and tokens
item by item. The real-benchmark version of this check is scripts/replay_check.py (needs the stored cache)."""
import hashlib
import math
import threading

import pytest

from boosttopo.bench.toy import ToyBench, toy_rule
from boosttopo.evalx import Evaluator
from boosttopo.graph import IN, Node, Subgraph
from boosttopo.infer import SEED_STRIDE, best_of_n_subgraph, hand_topology, infer, repair_loop, writer_role
from boosttopo.llm import CachedLLM, FakeLLM
from workflow import legacy
from workflow.evaluate import Env, MissingY0, run_item
from workflow.llm import CacheMiss, CacheOnly
from workflow.model import to_str, validate, Limits


def noisy_rule(req):
    """toy_rule, but rep-dependent: a writer sometimes fails to move, and whitespace depends on rep, so a seed
    mismatch changes the text, the tokens of later prompts and often the score."""
    out = toy_rule(req)
    h = int(hashlib.sha256(f"{req.rep}|{req.prompt}".encode()).hexdigest()[:8], 16)
    if req.role == "writer" and h % 3 == 0:
        out = req.y_cur or out
    return out + " " * (req.rep % 5) if req.role in ("writer", "repairer", "y0") else out


class Recorder:
    def __init__(self, b):
        self.b, self.last, self._l = b, {}, threading.Lock()

    def score(self, it, sol):
        with self._l:
            self.last[it.item_id] = sol
        return self.b.score(it, sol)

    def __getattr__(self, k):
        return getattr(self.b, k)


ITEMS = ToyBench().make_items("toy", 1, list(range(30, 42)))
SEED = 2  # a non-zero run seed: the legacy base repeat index is rep + seed * 1e6
ENSEMBLE = [
    Subgraph((Node("n0", "writer"), Node("n1", "analyst")), ((IN, "n0"), (IN, "n1"), ("n1", "n0")), "n0"),
    Subgraph((Node("n0", "writer", 3, "keep_best"),), ((IN, "n0"),), "n0"),
    Subgraph((Node("n0", "repairer", 2, "until_legal"),), ((IN, "n0"),), "n0"),
    Subgraph((Node("n0", "writer", 3, "keep_best"),), ((IN, "n0"),), "n0"),  # duplicate: occ = 1
]


def ev_for(bench, llm, d):
    ev = Evaluator(bench, llm, d, n_repeats=1, max_workers=4)
    ev.rep_seed = SEED * SEED_STRIDE
    ev.dedup_stages = True
    ev.make_start(ITEMS)
    return ev


def methods(ev_budget):
    return {
        "writer_once": (lambda items, ev: infer("adaboost", [writer_role()], items, ev, 1), legacy.writer_once()),
        "best_of_n_writer": (lambda items, ev: best_of_n_subgraph(writer_role(), items, ev, 5),
                             legacy.best_of_n_writer(5)),
        "repair_loop": (lambda items, ev: repair_loop(items, ev, ev_budget, 1), legacy.repair_loop(ev_budget)),
        "repair_loop5": (lambda items, ev: repair_loop(items, ev, math.inf, 1, max_repairs=5), legacy.repair_loop_k(5)),
        "ours": (lambda items, ev: infer("gbdt", ENSEMBLE, items, ev, 1), legacy.ensemble_workflow(ENSEMBLE)),
    }


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache")
    bench = Recorder(ToyBench())
    llm = CachedLLM(FakeLLM(noisy_rule), d)
    ev = ev_for(bench, llm, d)
    budget = 2.5 * (sum(ev.llm.call_y0(__import__("boosttopo.roles", fromlist=["y0_request"]).y0_request(it, 0))[1]
                        for it in ITEMS) / len(ITEMS)) + 0.5   # a non-integer budget that bites mid-loop
    stored, plans = {}, {}
    for name, (orig, _) in methods(budget).items():
        bench.last.clear()
        out = orig(ITEMS, ev)
        stored[name] = {r["item_id"]: r for r in out["rows"]}
        plans[name] = dict(bench.last)
    return d, budget, stored, plans


def test_original_code_replays_from_llm_cache_alone(live, tmp_path):
    d, budget, stored, plans = live
    inner = CacheOnly(FakeLLM(noisy_rule))
    bench = Recorder(ToyBench())
    llm = CachedLLM(inner, d)
    ev = ev_for(bench, llm, tmp_path)            # empty eval cache
    for name, (orig, _) in methods(budget).items():
        bench.last.clear()
        rows = {r["item_id"]: r for r in orig(ITEMS, ev)["rows"]}
        for iid, w in stored[name].items():
            assert {k: rows[iid][k] for k in ("score", "g", "tokens")} == {k: w[k] for k in ("score", "g", "tokens")}
        assert bench.last == plans[name]
    assert inner.misses == 0 and llm.budget.new_calls == 0


def test_workflows_match_original_rows_and_plans(live):
    d, budget, stored, plans = live
    inner = CacheOnly(FakeLLM(noisy_rule))
    llm = CachedLLM(inner, d)
    env = Env(llm, ToyBench(), seed_scheme="legacy", diag_mode="original", workers=1)
    env.make_start(ITEMS, require_cached=True)
    for name, (_, wf) in methods(budget).items():
        assert validate(wf, Limits(max_calls=100, max_llm_steps=100)) == [], name
        for it in ITEMS:
            r = run_item(wf, it, env, run_seed=SEED)
            w = stored[name][it.item_id]
            assert (r.score, r.g, r.cost.tokens, float(r.legal)) == (w["score"], w["g"], w["tokens"], w["valid"]), \
                (name, it.item_id, to_str(wf))
            assert r.plan == plans[name][it.item_id], (name, it.item_id)
    assert inner.misses == 0 and llm.budget.new_calls == 0


def test_the_budget_and_duplicates_matter(live):
    """Guard against a vacuous pass: the repair loop's budget cuts loops short on some items, and the duplicate
    member's occurrence shift draws different samples than its first copy."""
    d, budget, stored, _ = live
    assert any(stored["repair_loop"][i]["tokens"] < stored["repair_loop5"][i]["tokens"] for i in stored["repair_loop"])
    env = Env(CachedLLM(CacheOnly(FakeLLM(noisy_rule)), d), ToyBench(), seed_scheme="legacy", diag_mode="original")
    no_occ = legacy.ensemble_workflow(ENSEMBLE, fresh_duplicate_stages=False)
    misses = 0
    for it in ITEMS:
        try:
            run_item(no_occ, it, env, run_seed=SEED)
        except CacheMiss:
            misses += 1
    assert misses > 0


def test_not_expressible_is_reported():
    with pytest.raises(legacy.NotExpressible, match="later writers"):
        legacy.member_workflow(hand_topology())   # the analyst's notes would reach the repairer
    two_plans = Subgraph((Node("n0", "writer"), Node("n1", "writer")), ((IN, "n0"), ("n0", "n1"), (IN, "n1")), "n1")
    with pytest.raises(legacy.NotExpressible, match="second plan"):
        legacy.member_workflow(two_plans)


def test_missing_test_y0_aborts(tmp_path):
    env = Env(CachedLLM(CacheOnly(FakeLLM(noisy_rule)), tmp_path), ToyBench())
    with pytest.raises(MissingY0):
        env.make_start(ITEMS[:2], require_cached=True)
