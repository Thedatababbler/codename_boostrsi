"""W5 (serial), W6 (branch), W7 (ablation switches)."""
import statistics
import unittest

from boost.branch import _View, attach_case, fit_branch
from boost.common import BoostCfg, group_sum, residual_weights
from boost.global_search import fit_global
from boost.serial import attach_tail, fit_serial, tail_results
from farm.bench import gen_items
from workflow.evaluate import evaluate
from workflow.grow import GrowCfg
from workflow.model import iter_nodes, parse_wf, to_str

from helpers import fake_env

TRAIN = gen_items(10, 11)
HELD = gen_items(12, 12)
CFG = BoostCfg(rounds=3, grow=GrowCfg(max_steps=2), accept_margin=-1.0)  # accept everything


class TestSerial(unittest.TestCase):
    def test_train_g_monotone_and_frozen_not_rebilled(self):
        env = fake_env()
        wf, info = fit_serial(TRAIN, HELD, env, CFG)
        accepted = [r for r in info["rounds"] if r.get("accepted")]
        self.assertTrue(accepted)
        # replay the sequence of accepted stages: per-item train g never decreases
        cur = parse_wf("seq[draft]")
        prev = {r.item_id: r.g for r in evaluate(cur, TRAIN, env)}
        for r in accepted:
            frag = parse_wf(r["frag"])
            cur = attach_tail(cur, frag)
            now = {x.item_id: x.g for x in evaluate(cur, TRAIN, env)}
            for k in prev:
                self.assertGreaterEqual(now[k] + 1e-12, prev[k])
            prev = now
        self.assertEqual(to_str(cur), to_str(wf))

    def test_frozen_prefix_equivalence_and_cost(self):
        env = fake_env()
        base = parse_wf("seq[draft]")
        frag = parse_wf("seq[validate, loop(k=2, stop=legal)[repair]]")
        pre, tail = tail_results(base, frag, TRAIN, env, 0)
        full = evaluate(attach_tail(base, frag), TRAIN, env)
        for p, t, f in zip(pre, tail, full):
            self.assertEqual(t.score, f.score)                         # same samples
            if p.score < 1.0:
                self.assertEqual(t.cost.tokens + p.cost.tokens, f.cost.tokens)  # tail pays only itself
        n = env.llm.new_calls
        tail_results(base, frag, TRAIN, env, 0)
        self.assertEqual(env.llm.new_calls, n)                         # frozen + cached


class TestBranch(unittest.TestCase):
    def test_weights_grouped_by_label(self):
        g = {"a": 0.0, "b": 1.0, "c": 0.5, "d": -0.5}
        w = residual_weights(g, BoostCfg(eps=0.1, cap=5.0))
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(w["a"], w["d"])                         # g clipped at 0
        self.assertGreater(w["a"], w["c"])
        self.assertGreater(w["c"], w["b"])
        m = group_sum(w, {"a": "x", "b": "y", "c": "x", "d": "y"})
        self.assertAlmostEqual(m["x"], w["a"] + w["c"])
        self.assertAlmostEqual(m["y"], w["b"] + w["d"])
        # cap at 5x mean
        w = residual_weights({str(i): (0.0 if i == 0 else 10.0) for i in range(100)}, BoostCfg(eps=0.001))
        raw = residual_weights({str(i): (0.0 if i == 0 else 10.0) for i in range(100)},
                               BoostCfg(eps=0.001, cap=1e9))
        self.assertLess(w["0"], raw["0"])

    def test_one_case_per_item_at_inference(self):
        env = fake_env()
        wf, info = fit_branch(TRAIN, HELD, env, CFG)
        self.assertEqual(wf.children[-1].kind, "branch")
        rs = evaluate(wf, HELD, env)
        for r in rs:
            branch_events = [t for t in r.trace if t[1] == "branch" and len(t[0]) == 1]
            self.assertEqual(len(branch_events), 1)
            taken = r.label if branch_events[0][4] else None
            cases = wf.children[-1].cases
            if taken is None:
                self.assertEqual(r.cost.llm_calls, 1)                  # draft only
            # nothing from other cases ran: every trace event under the branch is under the label
            b = len(wf.children) - 1
            for t in r.trace:
                if len(t[0]) >= 2 and t[0][0] == b:
                    self.assertEqual(t[0][1], r.label)

    def test_small_label_not_grown(self):
        env = fake_env()
        cfg = BoostCfg(rounds=3, grow=GrowCfg(max_steps=1), accept_margin=-1.0, min_mass=0.10)
        wf, info = fit_branch(TRAIN, HELD, env, cfg)
        for r in info["rounds"]:
            self.assertGreaterEqual(r["mass"][r["target"]], 0.10)
            tot = sum(r["mass"].values())
            self.assertAlmostEqual(tot, 1.0)
        cfg2 = BoostCfg(rounds=1, grow=GrowCfg(max_steps=1), accept_margin=-1.0, min_mass=1.01)
        wf2, info2 = fit_branch(TRAIN, HELD, env, cfg2)
        self.assertEqual(info2["rounds"], [])
        self.assertEqual(wf2.children[-1].cases, {})

    def test_branch_view_matches_full_run(self):
        env = fake_env()
        wf = parse_wf("seq[draft, branch(status){illegal: loop(k=3, stop=legal)[validate, repair]}]")
        V = _View(wf, TRAIN, env, 0)
        full = {r.item_id: r for r in evaluate(wf, TRAIN, env)}
        for i in TRAIN:
            self.assertEqual(V.cur[i.id].score, full[i.id].score)


class TestAblationSwitches(unittest.TestCase):
    def test_switches_off_reproduce(self):
        a, _ = fit_serial(TRAIN, HELD, fake_env(), CFG)
        b, _ = fit_serial(TRAIN, HELD, fake_env(), BoostCfg(**{**CFG.__dict__, "no_residual": False,
                                                               "no_tools": False}))
        self.assertEqual(to_str(a), to_str(b))
        a, _ = fit_branch(TRAIN, HELD, fake_env(), CFG)
        b, _ = fit_branch(TRAIN, HELD, fake_env(), BoostCfg(**{**CFG.__dict__, "no_residual": False}))
        self.assertEqual(to_str(a), to_str(b))

    def test_no_tools_never_uses_tools(self):
        cfg = BoostCfg(**{**CFG.__dict__, "no_tools": True})
        for fit in (fit_serial, fit_branch):
            wf, _ = fit(TRAIN, HELD, fake_env(), cfg)
            names = {n.name for _, n in iter_nodes(wf) if hasattr(n, "name")}
            self.assertFalse(names & {"simulate", "crop_stats"}, to_str(wf))

    def test_no_residual_uniform(self):
        w = residual_weights({"a": 0.0, "b": 1.0}, BoostCfg(no_residual=True))
        self.assertEqual(w, {"a": 0.5, "b": 0.5})

    def test_global_respects_budget(self):
        wf, info = fit_global(TRAIN, HELD, fake_env(), CFG, eval_budget=4)
        self.assertLessEqual(info["evals"], 4)


if __name__ == "__main__":
    unittest.main()
