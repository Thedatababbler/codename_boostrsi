"""W2: tools and checks."""
import ast
import importlib
import json
import os
import sys
import unittest

from farm.bench import FarmBench, standard_splits
from farm.rules import simulate as rules_sim
from workflow import tools
from workflow.evaluate import evaluate
from workflow.model import State, parse_wf

from helpers import ILLEGAL, ITEM, LEGAL_LOW, PROBLEM, fake_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestTools(unittest.TestCase):
    def test_no_reference_import_static(self):
        with open(os.path.join(ROOT, "workflow", "tools.py")) as f:
            src = f.read()
        mods = set()
        for n in ast.walk(ast.parse(src)):
            if isinstance(n, ast.Import):
                mods |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                mods.add(n.module or "")
        self.assertFalse(any("reference" in m or "bench" in m for m in mods), mods)

    def test_no_reference_import_runtime(self):
        for m in [m for m in sys.modules if m.startswith(("farm", "workflow.tools"))]:
            del sys.modules[m]
        importlib.import_module("workflow.tools")
        self.assertNotIn("farm.reference", sys.modules)

    def test_simulate_profit_matches_scorer_on_40_items(self):
        items = standard_splits()["test"]
        env = fake_env()
        bench = FarmBench()
        rs = evaluate(parse_wf("seq[fanout(n=3)[draft]]"), items, env)
        n_legal = 0
        for it, r in zip(items, rs):
            st = r.state
            res = bench.score(it, st.best_plan)
            text = tools.simulate(State(problem=it.problem, plan=st.best_plan))
            if res.legal:
                n_legal += 1
                self.assertIn(f"profit={int(res.profit)};", text)
                self.assertEqual(rules_sim(it.problem, st.best_plan).profit, res.profit)
            else:
                self.assertIn("ILLEGAL", text)
        self.assertGreater(n_legal, 10)

    def test_crop_stats_hand_example(self):
        # corn: margin 12-5=7, 7/3=2.33/day, last plant day 10-3=7; bean: 2, 1.00/day, day 8
        out = tools.crop_stats(State(problem=ITEM.problem))
        self.assertEqual(out, "corn: grow=3d, profit/plot/cycle=7, per-day=2.33, last plant day=7; "
                              "bean: grow=2d, profit/plot/cycle=2, per-day=1.00, last plant day=8")

    def test_validate_diag_and_error_type(self):
        d = tools.validate(State(problem=ITEM.problem, plan=ILLEGAL))
        self.assertTrue(d.startswith("ILLEGAL[farmland]"))
        self.assertIn("state: cash=10, idle=2", d)
        self.assertEqual(tools.error_type(State(problem="", diag=d)), "farmland")
        self.assertEqual(tools.validate(State(problem=ITEM.problem, plan=LEGAL_LOW)), "LEGAL")
        early = json.dumps([{"day": 1, "action": "plant", "crop": "corn", "plots": 1},
                            {"day": 2, "action": "harvest", "crop": "corn", "plots": 1}])
        d = tools.validate(State(problem=ITEM.problem, plan=early))
        self.assertEqual(tools.error_type(State(problem="", diag=d)), "not_mature")
        poor = json.dumps([{"day": 1, "action": "plant", "crop": "corn", "plots": 2},
                           {"day": 1, "action": "plant", "crop": "bean", "plots": 0}])
        self.assertEqual(tools.error_type(State(problem="", diag=tools.validate(
            State(problem=ITEM.problem, plan=poor)))), "other")

    def test_status(self):
        st = State(problem=ITEM.problem, plan=LEGAL_LOW, plan_legal=True, plan_score=0.3)
        self.assertEqual(tools.status(st), "legal_low")
        self.assertEqual(tools.status(State(problem=ITEM.problem, plan_legal=True, plan_score=0.9)), "legal_high")
        self.assertEqual(tools.status(State(problem=ITEM.problem, plan=ILLEGAL)), "illegal")

    def test_outputs_short(self):
        big = dict(PROBLEM, days=60)
        plan = json.dumps([{"day": d, "action": "plant", "crop": "bean", "plots": 1} for d in (1,)] +
                          [{"day": 3, "action": "harvest", "crop": "bean", "plots": 1}])
        for fn in (tools.simulate, tools.crop_stats, tools.validate):
            self.assertLessEqual(len(fn(State(problem=json.dumps(big), plan=plan))), tools.MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
