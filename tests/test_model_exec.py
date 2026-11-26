"""W1: representation + executor."""
import os
import tempfile
import unittest

from farm.bench import gen_items
from workflow.evaluate import evaluate, run_item
from workflow.model import (Limits, parse_wf, to_str, validate, worst_case_calls)
from workflow import baselines

from helpers import ILLEGAL, ITEM, LEGAL_LOW, fake_env, ref_plan, scripted_env


class TestModel(unittest.TestCase):
    def test_roundtrip(self):
        for t in ["seq[draft]", "seq[fanout(n=5)[draft]]", "seq[draft, loop(k=5, stop=best)[validate, repair]]",
                  "seq[crop_stats, draft, branch(status){illegal: loop(k=3, stop=legal)[validate, repair], "
                  "legal_low: seq[simulate, improve]}]"]:
            self.assertEqual(to_str(parse_wf(t)), t)
            self.assertEqual(validate(parse_wf(t)), [])

    def test_doc_example_calls(self):
        wf = parse_wf("seq[crop_stats, draft, branch(status){illegal: loop(k=3, stop=legal)[validate, repair], "
                      "legal_low: seq[simulate, improve]}]")
        self.assertEqual(worst_case_calls(wf), 4)  # §2.5: illegal path = 1 + 3

    def test_controls_expressible(self):
        # §2.3 hard requirement: both strong controls are in the search space
        self.assertEqual(validate(baselines.repair_loop(5)), [])
        self.assertEqual(validate(baselines.best_of(5)), [])

    def test_constraints(self):
        self.assertTrue(validate(parse_wf("seq[repair]")))                  # reads plan/diag first
        self.assertTrue(validate(parse_wf("seq[draft, repair]")))           # diag never written
        self.assertTrue(validate(parse_wf("seq[simulate]")))                # no plan writer
        self.assertTrue(validate(parse_wf("seq[draft, loop(k=2, stop=legal)[validate]]")))  # loop w/o plan
        self.assertTrue(validate(parse_wf("seq[fanout(n=5)[draft, improve]]"), Limits(max_calls=8)))
        deep = parse_wf("seq[loop(k=2, stop=best)[fanout(n=2)[loop(k=2, stop=best)[draft]]]]")
        self.assertTrue(any("depth" in e for e in validate(deep, Limits(max_calls=100))))
        # branch-only fields don't count as defined afterwards
        wf = parse_wf("seq[draft, branch(status){illegal: seq[validate]}, repair]")
        self.assertTrue(validate(wf))
        self.assertEqual(validate(parse_wf("seq[draft, validate, branch(error_type){funds: seq[repair]}]")), [])


class TestExec(unittest.TestCase):
    def test_seq_early_stop_at_reference(self):
        env = scripted_env({"draft": [ref_plan()], "improve": [LEGAL_LOW]})
        r = run_item(parse_wf("seq[draft, simulate, improve]"), ITEM, env)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.cost.llm_calls, 1)          # improve never ran
        self.assertEqual(r.cost.tool_calls, {})        # simulate never ran

    def test_loop_stop_legal_and_progress_kept(self):
        # repair #1 returns a still-illegal plan; repair #2 must see that plan (not best_plan)
        still_bad = ILLEGAL.replace('"plots": 3', '"plots": 4')
        env = scripted_env({"draft": [ILLEGAL], "repair": [still_bad, LEGAL_LOW, LEGAL_LOW]})
        r = run_item(parse_wf("seq[draft, loop(k=5, stop=legal)[validate, repair]]"), ITEM, env)
        self.assertTrue(r.legal)
        self.assertEqual(r.cost.llm_calls, 3)          # draft + 2 repairs, then stop=legal
        second_repair_prompt = [c for c in env.llm.calls if c[0] == "repair"][1][2]
        self.assertIn('"plots": 4', second_repair_prompt)
        self.assertEqual([t[2] for t in r.trace if t[1] == "loop"], [2])

    def test_loop_stop_best_runs_k(self):
        env = scripted_env({"draft": [ILLEGAL], "repair": [LEGAL_LOW]})
        r = run_item(parse_wf("seq[draft, loop(k=3, stop=best)[validate, repair]]"), ITEM, env)
        self.assertEqual(r.cost.llm_calls, 4)
        self.assertEqual(r.cost.scorer_calls, 4)
        self.assertEqual(r.cost.tool_calls, {"validate": 3})

    def test_branch_untaken_costs_nothing(self):
        wf = parse_wf("seq[draft, branch(status){illegal: loop(k=3, stop=legal)[validate, repair], "
                      "legal_low: seq[simulate, improve]}]")
        env = scripted_env({"draft": [LEGAL_LOW], "improve": [LEGAL_LOW]})
        r = run_item(wf, ITEM, env)
        self.assertEqual(r.cost.llm_calls, 2)
        self.assertEqual(r.cost.tool_calls, {"simulate": 1})
        self.assertEqual(r.label, "legal_low")
        env = scripted_env({"draft": [ILLEGAL], "repair": [LEGAL_LOW]})
        r = run_item(wf, ITEM, env)
        self.assertEqual(r.label, "illegal")
        self.assertEqual(r.cost.llm_calls, 2)
        # no case for the label -> pass through
        r = run_item(parse_wf("seq[draft, branch(status){illegal: seq[validate, repair]}]"), ITEM,
                     scripted_env({"draft": [LEGAL_LOW]}))
        self.assertEqual(r.cost.llm_calls, 1)

    def test_fanout_keeps_best(self):
        env = scripted_env({"draft": [ILLEGAL, LEGAL_LOW, ILLEGAL]})
        r = run_item(parse_wf("seq[fanout(n=3)[draft]]"), ITEM, env)
        self.assertTrue(r.legal)
        self.assertEqual(r.cost.llm_calls, 3)

    def test_fresh_seeds_per_position_and_iteration(self):
        env = scripted_env({"draft": [ILLEGAL], "repair": [ILLEGAL]})
        run_item(parse_wf("seq[fanout(n=3)[draft], loop(k=3, stop=best)[validate, repair]]"), ITEM, env)
        seeds = [c[1] for c in env.llm.calls]
        self.assertEqual(len(seeds), 6)
        self.assertEqual(len(set(seeds)), 6)
        # reproducible
        env2 = scripted_env({"draft": [ILLEGAL], "repair": [ILLEGAL]})
        run_item(parse_wf("seq[fanout(n=3)[draft], loop(k=3, stop=best)[validate, repair]]"), ITEM, env2)
        self.assertEqual(seeds, [c[1] for c in env2.llm.calls])
        # different run seed -> different samples
        env3 = scripted_env({"draft": [ILLEGAL], "repair": [ILLEGAL]})
        run_item(parse_wf("seq[fanout(n=3)[draft], loop(k=3, stop=best)[validate, repair]]"), ITEM, env3, run_seed=7)
        self.assertTrue(set(seeds).isdisjoint(c[1] for c in env3.llm.calls))

    def test_cache_replay_identical(self):
        items = gen_items(8, 5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.jsonl")
            for wf in (baselines.repair_loop(3), baselines.best_of(4)):
                a = evaluate(wf, items, fake_env(path))
                env = fake_env(path)                        # fresh process-equivalent, disk cache
                b = evaluate(wf, items, env)
                self.assertEqual(env.llm.new_calls, 0)
                self.assertEqual([(r.score, r.cost.tokens, r.cost.llm_calls) for r in a],
                                 [(r.score, r.cost.tokens, r.cost.llm_calls) for r in b])

    def test_best_of_n_prefix_shares_samples(self):
        items = gen_items(5, 9)
        env = fake_env()
        evaluate(baselines.best_of(2), items, env)
        n0 = env.llm.new_calls
        evaluate(baselines.best_of(4), items, env)
        self.assertEqual(env.llm.new_calls - n0, 2 * len(items))   # only branches 3, 4 are new


if __name__ == "__main__":
    unittest.main()
