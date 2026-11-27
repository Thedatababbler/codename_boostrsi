"""W2: tools and checks (revision §12)."""
import json
import re
import unittest

from farm.adapter import ToyFarmAdapter, standard_splits
from workflow import tools
from workflow.evaluate import evaluate
from workflow.exec import Ctx
from workflow.model import State, parse_wf

from helpers import ILLEGAL, ITEM, LEGAL_LOW, fake_env

BENCH = ToyFarmAdapter()


def ctx(item=ITEM, mode="clean"):
    return Ctx(None, BENCH, item, diag_mode=mode)


def st(plan, item=ITEM):
    sr = BENCH.score(item, plan)
    return State(problem=item.prompt, plan=plan, plan_score=sr.score, plan_legal=sr.valid, plan_cat=sr.category)


class TestTools(unittest.TestCase):
    def test_no_tool_output_contains_reference_or_relative_score(self):
        items = standard_splits()["test"]
        rs = evaluate(parse_wf("seq[fanout(n=3)[draft]]"), items, fake_env())
        n_legal = 0
        for it, r in zip(items, rs):
            for plan in {r.plan, r.state.plan}:
                s = st(plan, it)
                outs = [tools.validate(s, ctx(it)), tools.simulate(s, ctx(it)), tools.crop_stats(s, ctx(it))]
                self.assertFalse(any("reference" in o for o in outs), outs)
                if not s.plan_legal:
                    continue
                n_legal += 1
                sim = BENCH.simulate(it, plan)
                for o in outs:
                    if sim["reference_profit"] != sim["profit"]:
                        self.assertIsNone(re.search(rf"(?<![\d.]){re.escape(str(sim['reference_profit']))}(?![\d.])", o))
                    self.assertNotIn(str(sim["relative_score"]), o)
        self.assertGreater(n_legal, 10)

    def test_simulate_profit_matches_scorer(self):
        items = standard_splits()["test"]
        rs = evaluate(parse_wf("seq[fanout(n=3)[draft]]"), items, fake_env())
        n_legal = 0
        for it, r in zip(items, rs):
            sr = BENCH.score(it, r.plan)
            text = tools.simulate(st(r.plan, it), ctx(it))
            if sr.valid:
                n_legal += 1
                scorer_profit = re.search(r"profit=(-?[\d.]+)", sr.diag).group(1)
                self.assertEqual(re.search(r"profit=(-?[\d.]+)", text).group(1), scorer_profit)
            else:
                self.assertTrue(text.startswith("simulation stopped"))
        self.assertGreater(n_legal, 10)

    def test_validate_modes(self):
        legal = st(LEGAL_LOW)
        orig = tools.validate(legal, ctx(mode="original"))
        self.assertEqual(orig, BENCH.score(ITEM, LEGAL_LOW).diag)        # byte-identical
        self.assertIn("reference greedy profit=", orig)
        clean = tools.validate(legal, ctx())
        self.assertEqual(clean, orig.replace(re.search(r" \(reference greedy profit=[^)]*\)", orig).group(0), ""))
        bad = tools.validate(st(ILLEGAL), ctx())
        self.assertTrue(bad.startswith("illegal plan:"))
        self.assertIn("State at the start of day 1", bad)                 # failing-day state kept
        self.assertEqual(bad, tools.validate(st(ILLEGAL), ctx(mode="original")))

    def test_crop_stats_hand_example(self):
        # corn: margin 12-5=7, 7/3=2.33/day, last plant day 10-3=7; bean: 2, 1.00/day, day 8
        out = tools.crop_stats(st(None), ctx())
        self.assertEqual(out, "corn: grow=3d, profit/plot/cycle=7, per-day=2.33, last plant day=7; "
                              "bean: grow=2d, profit/plot/cycle=2, per-day=1.00, last plant day=8")

    def test_status_from_category(self):
        self.assertEqual(tools.status(st(ILLEGAL)), "illegal")
        self.assertEqual(tools.status(st(LEGAL_LOW)), "legal_low")      # valid_idle
        self.assertEqual(st(LEGAL_LOW).plan_cat, "valid_idle")
        for cat, lab in (("valid_ok", "legal_high"), ("valid_waste", "legal_low"), ("valid_idle", "legal_low")):
            self.assertEqual(tools.status(State(problem="", plan_legal=True, plan_cat=cat)), lab)

    def test_error_type_mapping(self):
        cats = ("insufficient_funds", "farmland_exceeded", "missing_seeds", "not_mature", "missing_fruit", "bad_day",
                "unknown_crop", "schema_error", "parse_error", "empty")
        self.assertEqual(set(cats), set(tools.ERROR_TYPE_OF))
        self.assertEqual(tools.error_type(st(ILLEGAL)), "farmland")
        early = json.dumps([{"day": 1, "action": "plant", "crop": "corn", "plots": 1},
                            {"day": 2, "action": "harvest", "crop": "corn", "plots": 1}])
        self.assertEqual(tools.error_type(st(early)), "not_mature")
        self.assertEqual(tools.error_type(st("no plan here")), "other")
        self.assertEqual(tools.error_type(st(LEGAL_LOW)), "none")

    def test_outputs_short(self):
        P = {"days": 60, "cash": 10, "plots": 2, "crops": {"bean": {"cost": 2, "grow": 2, "price": 4}}}
        from farm.adapter import make_item
        it = make_item(P, "big")
        plan = json.dumps([{"day": d, "action": a, "crop": "bean", "plots": 1}
                           for d0 in range(1, 58, 2) for d, a in ((d0, "plant"), (d0 + 2, "harvest"))])
        for fn in (tools.simulate, tools.crop_stats):
            self.assertLessEqual(len(fn(st(plan, it), ctx(it))), tools.MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
