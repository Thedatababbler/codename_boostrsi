"""Regression tests added in the trustworthiness round: best-of-N, f0 rule 1, split hygiene, scorer counts."""
import json

import pytest

from boosttopo.bench.toy import toy_rule
from boosttopo.config import load_config
from boosttopo.evalx import Evaluator
from boosttopo.graph import IN, Edit, Node, Subgraph, single_writer
from boosttopo.infer import SEED_STRIDE, best_of_n, hand_topology, infer, stage_scorer_calls
from boosttopo.llm import build_llm
from boosttopo.policy import F0Policy, Obs
from boosttopo.run import eval_splits, fit, main, make_splits

from conftest import toy_cfg


# ---------------------------------------------------------------- best-of-N
def _perfect_on_rep(good_rep: int):
    """y0 sample `good_rep` is the hidden target (score 1); every other y0 sample is the usual toy y0."""
    def rule(req):
        if req.role == "y0" and req.rep == good_rep:
            return json.dumps(req.item.payload["target"])
        return toy_rule(req)
    return rule


def _ev(tmp_path, rule, cfg=None):
    cfg = cfg or toy_cfg(tmp_path)
    from boosttopo.bench.base import make_bench

    bench = make_bench(cfg)
    llm = build_llm(cfg, str(tmp_path / "cache"), fake_rule=rule)
    return cfg, bench, llm, Evaluator(bench, llm, str(tmp_path / "cache"), n_repeats=1, max_workers=4)


def test_best_of_n_uses_n_distinct_samples_and_picks_best(tmp_path):
    cfg, bench, llm, ev = _ev(tmp_path, _perfect_on_rep(3))
    items = make_splits(cfg, bench)["heldout"]
    ev.make_start(items)
    calls_before = llm.budget.new_calls
    res = best_of_n(items, ev, 4)
    assert res["n_samples"] == 4
    # 3 new samples per item (sample 0 is the cached y0), never collapsed to 1
    assert llm.budget.new_calls - calls_before == 3 * len(items)
    assert res["scorer_calls_per_item"] == 4
    y0_tok = llm.call(__import__("boosttopo.roles", fromlist=["y0_request"]).y0_request(items[0], 0))[1]
    assert res["rows"][0]["tokens"] == 4 * y0_tok
    assert res["mean_score"] == pytest.approx(1.0)  # sample rep 3 is perfect on every item
    assert res["mean_g"] == pytest.approx(1.0)
    # with N=3 the perfect sample is not drawn: no gain over y0, but still >= 0
    res3 = best_of_n(items, ev, 3)
    assert res3["mean_g"] == 0.0 and res3["n_samples"] == 3


def test_best_of_n_seed_shifts_extra_samples_but_keeps_y0(tmp_path):
    cfg, bench, llm, ev = _ev(tmp_path, _perfect_on_rep(SEED_STRIDE + 1))
    items = make_splits(cfg, bench)["heldout"][:3]
    ev.make_start(items)
    assert best_of_n(items, ev, 2, seed=0)["mean_g"] == 0.0
    assert best_of_n(items, ev, 2, seed=1)["mean_g"] == pytest.approx(1.0)


def test_best_of_n_same_on_heldout_and_test_splits(tmp_path):
    """No split-dependent path: the same items give the same best-of-N whatever split they are called from."""
    cfg, bench, llm, ev = _ev(tmp_path, _perfect_on_rep(2))
    sp = make_splits(cfg, bench)
    for name in ("heldout", "G1"):
        ev.make_start(sp[name])
        assert best_of_n(sp[name], ev, 3)["mean_g"] == pytest.approx(1.0), name


# ---------------------------------------------------------------- scorer-call accounting
def test_scorer_calls_per_item(toy):
    cfg, bench, llm, ev, log, sp = toy()
    items = sp["heldout"]
    assert stage_scorer_calls(hand_topology()) == 2  # repairer diagnosis + output gate
    assert infer("adaboost", [], items, ev, 1)["scorer_calls_per_item"] == 1
    assert infer("adaboost", [hand_topology()], items, ev, 1)["scorer_calls_per_item"] == 3
    h2 = Subgraph((Node("n0", "repairer"),), ((IN, "n0"),), "n0")
    assert infer("adaboost", [hand_topology(), h2], items, ev, 1)["scorer_calls_per_item"] == 5


# ---------------------------------------------------------------- f0 rule 1
def _obs(invalid_share=1.0):
    return Obs(subgraph={"fit": 0.0, "cost_penalty": 0.0}, residual={"invalid_share_hi": invalid_share},
               edit_stats={}, round=0, step=0)


def test_f0_rule1_promotes_only_edits_that_add_a_repairer():
    pol = F0Policy()
    h = Subgraph((Node("n0", "writer"), Node("n1", "repairer")), ((IN, "n0"), ("n0", "n1")), "n1")
    add_rep = Edit.make("add_node", role="repairer", nid="n2", srcs=("n1",))
    away = Edit.make("change_role", nid="n1", role="writer")  # turns the repairer into a writer
    toward = Edit.make("change_role", nid="n0", role="repairer")
    other = Edit.make("add_node", role="writer", nid="n2", srcs=("n1",))
    obs = _obs()
    assert pol._score(h, add_rep, obs) >= 100.0
    assert pol._score(h, toward, obs) >= 100.0
    assert pol._score(h, away, obs) < 100.0
    assert pol._score(h, other, obs) < 100.0
    ranked = pol.rank(h, [away, other, add_rep], obs)
    assert ranked[0] == add_rep
    # rule 1 is off when invalid plans do not dominate
    assert pol._score(h, add_rep, _obs(0.0)) < 100.0


# ---------------------------------------------------------------- split hygiene
def test_config_split_maps_replace_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("bench:\n  dev_tests:\n    dev_d2: {difficulty: 2, seeds: [3000, 3002]}\n")
    cfg = load_config(p)
    assert list(cfg.bench.dev_tests) == ["dev_d2"]  # no default split merged in (the old G2_d5 bug)
    p.write_text("bench:\n  tests:\n    G1: {difficulty: 2, seeds: [3000, 3002]}\n")
    with pytest.raises(ValueError):
        load_config(p)


def test_final_test_splits_only_with_flag(tmp_path):
    cfg = toy_cfg(tmp_path)
    from boosttopo.bench.base import make_bench

    bench = make_bench(cfg)
    sp = make_splits(cfg, bench)
    finals = set(cfg.bench.final_tests)
    assert finals and not finals & set(sp)
    assert not finals & set(eval_splits(cfg, final_test=False))
    sp_f = make_splits(cfg, bench, final_test=True)
    assert finals <= set(sp_f)
    assert set(eval_splits(cfg, final_test=True)) == finals


def test_run_without_flag_never_touches_final_items(tmp_path, monkeypatch):
    """End-to-end on the toy: neither the LLM nor the scorer ever sees a final-test item without --final-test."""
    import boosttopo.run as runmod
    from boosttopo.bench import toy

    seen = set()
    orig = toy.ToyBench.score

    def spy(self, item, sol):
        seen.add(item.item_id)
        return orig(self, item, sol)

    monkeypatch.setattr(toy.ToyBench, "score", spy)
    cfgp = tmp_path / "toy.yaml"
    cfg = toy_cfg(tmp_path, rounds=1, run_id="hyg")
    cfgp.write_text(json.dumps(cfg.to_dict()))
    monkeypatch.setattr(runmod, "__file__", str(tmp_path / "src" / "boosttopo" / "run.py"))
    main(["--config", str(cfgp), "--baselines"])
    from boosttopo.bench.base import make_bench

    final_ids = {it.item_id for k, its in make_splits(cfg, make_bench(cfg), final_test=True).items()
                 if k in cfg.bench.final_tests for it in its}
    assert seen and not seen & final_ids
    rows = [json.loads(l) for l in (tmp_path / "runs" / "hyg" / "per_item.jsonl").read_text().splitlines()]
    assert {r["split"] for r in rows} == {"heldout", "G1"}
    assert {"ours", "single_writer", "hand_topology", "best_of_n_matched", "writer_once", "best_of_n_writer",
            "repair_loop"} == {r["method"] for r in rows}
    main(["--config", str(cfgp), "--baselines", "--final-test"])
    fin = json.loads((tmp_path / "runs" / "hyg" / "final_final_test.json").read_text())
    assert set(fin["splits"]) == set(cfg.bench.final_tests)


# ---------------------------------------------------------------- writer-role baselines (round 3)
def test_writer_role_baselines(toy):
    from boosttopo.infer import best_of_n_subgraph, repair_loop, writer_role

    cfg, bench, llm, ev, log, sp = toy()
    items = sp["heldout"]
    once = infer("adaboost", [writer_role()], items, ev, 1)
    assert once["scorer_calls_per_item"] == 2
    bon = best_of_n_subgraph(writer_role(), items, ev, 3)
    assert bon["scorer_calls_per_item"] == 4 and bon["mean_g"] >= once["mean_g"] - 1e-12
    # repair loop: the writer sample is the writer_once sample; with a budget of 0 no repair is made
    rl0 = repair_loop(items, ev, 0.0, 1)
    assert rl0["mean_repairs"] == 0 and rl0["mean_g"] == pytest.approx(once["mean_g"])
    rl = repair_loop(items, ev, once["tokens_per_item"] * 3, 1)
    assert rl["mean_repairs"] > 0
    assert rl["mean_g"] >= once["mean_g"] - 1e-12  # toy repairer only fixes negatives, never hurts
    assert all(r["tokens"] >= once["tokens_per_item"] * 3 or r["g"] >= 1.0 or r["n_repairs"] == 40 for r in rl["rows"])


def test_fresh_heldout_eval_uses_new_samples(toy, tmp_path):
    for mode in ("gbdt", "adaboost"):
        seen = {}
        for flag in (False, True):
            cfg, bench, llm, ev, log, sp = toy(run_dir=str(tmp_path / f"fr_{mode}_{flag}"), mode=mode,
                                               fresh_heldout_eval=flag)
            reps = set()
            orig = ev._one

            def spy(h, item, y_in, rep, occ=0, _o=orig, _reps=reps):
                _reps.add(rep + ev.rep_base)
                return _o(h, item, y_in, rep, occ)

            ev._one = spy
            fr = fit(cfg, sp, ev, log)
            seen[flag] = (reps, [h.key() for h in fr.ensemble], [x["accepted"] for x in fr.history])
        assert not any(r >= ev.FRESH_OFFSET for r in seen[False][0])
        assert any(r >= ev.FRESH_OFFSET for r in seen[True][0])
        # the switch never changes accept decisions on the (deterministic) toy
        assert seen[False][1:] == seen[True][1:]


def test_methods_subset_merges_into_stored_results(tmp_path):
    cfgp = tmp_path / "toy.yaml"
    cfg = toy_cfg(tmp_path, rounds=1, run_id="merge")
    cfgp.write_text(json.dumps(cfg.to_dict()))
    import boosttopo.run as runmod
    old = runmod.__file__
    runmod.__file__ = str(tmp_path / "src" / "boosttopo" / "run.py")
    try:
        main(["--config", str(cfgp)])  # ours only
        d = tmp_path / "runs" / "merge"
        before = json.loads((d / "final.json").read_text())
        main(["--config", str(cfgp), "--methods", "writer_once", "--splits", "heldout"])
        after = json.loads((d / "final.json").read_text())
        rows = [json.loads(l) for l in (d / "per_item.jsonl").read_text().splitlines()]
    finally:
        runmod.__file__ = old
    assert set(after["splits"]["heldout"]) == {"ours", "writer_once"}
    assert set(after["splits"]["G1"]) == {"ours"}
    assert after["splits"]["G1"]["ours"]["mean_g"] == before["splits"]["G1"]["ours"]["mean_g"]
    assert {(r["split"], r["method"]) for r in rows} == {("heldout", "ours"), ("G1", "ours"), ("heldout", "writer_once")}
    assert (d / "config_test.json").exists()
