"""W3 (baselines / precheck methods) and W4 (edits, proposer, f0)."""
import random
import unittest

from farm.bench import gen_items
from workflow import baselines
from workflow.edits import FRAGMENTS, Edit, Fragment, apply_edit, legal_edits
from workflow.evaluate import evaluate
from workflow.model import (Limits, get, iter_nodes, make_step, parse_wf, seq, to_str, validate,
                            worst_case_calls)
from workflow.policy import F0Policy
from workflow.proposer import WorkflowProposer
from workflow.skills import TEMPLATES

from helpers import fake_env


class TestBaselines(unittest.TestCase):
    def test_precheck_methods_run_on_fake(self):
        items = gen_items(4, 3)
        env = fake_env()
        for name, wf in baselines.precheck_methods().items():
            self.assertEqual(validate(wf), [], name)
            rs = evaluate(wf, items, env)
            for r in rs:
                self.assertLessEqual(r.cost.llm_calls, worst_case_calls(wf), name)

    def test_call_matching(self):
        m = baselines.precheck_methods()
        for f in baselines.HANDWRITTEN.values():
            c = worst_case_calls(f)
            self.assertEqual(worst_case_calls(m[f"B1_best_of_{c}"]), c)
            self.assertEqual(worst_case_calls(m[f"B2_tools_best_of_{c}"]), c)
            self.assertEqual(worst_case_calls(m[f"B3_repair_loop_k{c - 1}"]), c)
        self.assertEqual([worst_case_calls(f) for f in baselines.HANDWRITTEN.values()], [2, 3, 4])

    def test_token_alignment_rounds_down(self):
        self.assertEqual(baselines.token_aligned_n(9998, 3333), 2)
        self.assertEqual(baselines.token_aligned_n(9999, 3333), 3)
        self.assertEqual(baselines.token_aligned_n(10, 3333), 1)
        self.assertEqual(baselines.token_aligned_k(10000, 3000, 3000), 2)

    def test_same_template_with_or_without_tools(self):
        self.assertIn("{crops}", TEMPLATES["draft"])
        self.assertIn("{sim}", TEMPLATES["improve"])


class TestEdits(unittest.TestCase):
    def test_each_edit_kind(self):
        L = Limits()
        wf = parse_wf("seq[draft]")
        e = Edit("insert_step", (1,), (make_step("validate"),))
        self.assertEqual(to_str(apply_edit(wf, e)), "seq[draft, validate]")
        wf2 = apply_edit(apply_edit(wf, e), Edit("insert_step", (2,), (make_step("repair"),)))
        self.assertEqual(validate(wf2), [])
        w = apply_edit(wf2, Edit("wrap_loop", (2,), (3, "legal")))
        self.assertEqual(to_str(w), "seq[draft, validate, loop(k=3, stop=legal)[repair]]")
        w = apply_edit(w, Edit("set_param", (2,), ("k", 5)))
        self.assertEqual(get(w, (2,)).params["k"], 5)
        self.assertEqual(to_str(apply_edit(w, Edit("unwrap", (2,)))), "seq[draft, validate, repair]")
        self.assertEqual(to_str(apply_edit(wf, Edit("wrap_fanout", (0,), (3,)))), "seq[fanout(n=3)[draft]]")
        self.assertEqual(to_str(apply_edit(wf2, Edit("replace_skill", (2,), ("improve",)))),
                         "seq[draft, validate, improve]")
        self.assertEqual(to_str(apply_edit(wf2, Edit("remove", (1,)))), "seq[draft, repair]")
        b = apply_edit(wf, Edit("add_branch", (1,), ("status", "legal_low", parse_wf("seq[simulate, improve]"))))
        self.assertEqual(to_str(b), "seq[draft, branch(status){legal_low: seq[simulate, improve]}]")
        b = apply_edit(b, Edit("add_branch", (1,), ("status", "illegal", FRAGMENTS["repair_loop"])))
        self.assertEqual(set(get(b, (1,)).cases), {"legal_low", "illegal"})
        self.assertEqual(to_str(apply_edit(wf, Fragment("stats_first", (0,)))), "seq[crop_stats, draft]")
        self.assertEqual(to_str(apply_edit(wf, Fragment("multi_draft", (0,)))), "seq[fanout(n=3)[draft]]")
        self.assertEqual(validate(apply_edit(wf, Fragment("split_by_status", (1,)))), [])
        # illegal results are never offered
        legal = legal_edits(wf, L)
        self.assertFalse(any(isinstance(e, Edit) and e.op == "insert_step" and e.path == (1,)
                             and e.args[0].name == "repair" for e in legal))  # repair needs diag
        self.assertFalse(any(e.op == "remove" for e in legal if isinstance(e, Edit)))  # last plan writer

    def test_random_edit_sequences_stay_legal(self):
        rng = random.Random(0)
        for limits in (Limits(), Limits(max_calls=4), Limits(max_calls=12, max_depth=3)):
            for trial in range(25):
                wf = seq(make_step("draft"))
                for _ in range(8):
                    es = legal_edits(wf, limits)
                    if not es:
                        break
                    wf = apply_edit(wf, rng.choice(es))
                    self.assertEqual(validate(wf, limits), [], to_str(wf))
                    self.assertLessEqual(worst_case_calls(wf), limits.max_calls)

    def test_context_validation_for_fragments(self):
        base = parse_wf("seq[draft]")
        ctx = lambda f: seq(*base.children, *f.children)  # noqa: E731
        es = legal_edits(seq(), Limits(), ctx)
        self.assertTrue(any(isinstance(e, Fragment) and e.name == "repair_loop" for e in es))
        for e in es:
            self.assertEqual(validate(ctx(apply_edit(seq(), e))), [])

    def test_proposer_slots_and_f0(self):
        base = parse_wf("seq[draft]")
        ctx = lambda f: seq(*base.children, *f.children)  # noqa: E731
        pol = F0Policy()
        p = WorkflowProposer()
        c = p.propose(seq(), pol, {"residual_source": {"illegal": 0.9, "legal_low": 0.1}}, ctx)
        self.assertEqual([e.category for e in c], ["fragment", "control", "other"])
        self.assertIn(c[0].name, ("repair_loop", "split_by_status"))
        c2 = WorkflowProposer().propose(seq(), F0Policy(), {"residual_source": {"illegal": 0.1, "legal_low": 0.9}}, ctx)
        self.assertIn(c2[0].name, ("sim_improve", "split_by_status"))
        # no repeats of already-evaluated candidates
        c3 = p.propose(seq(), pol, {"residual_source": {"illegal": 0.9}}, ctx)
        self.assertTrue(set(map(str, c)).isdisjoint(map(str, c3)))

    def test_no_tools_edit_set(self):
        allowed = {"draft", "repair", "improve", "analyze", "validate", "reset_to_best"}
        es = legal_edits(parse_wf("seq[draft]"), Limits(), allowed_steps=allowed,
                         allowed_fragments={"repair_loop", "multi_draft"})
        for e in es:
            names = {n.name for _, n in iter_nodes(apply_edit(parse_wf("seq[draft]"), e)) if hasattr(n, "name")}
            self.assertFalse(names & {"simulate", "crop_stats"}, str(e))


if __name__ == "__main__":
    unittest.main()
