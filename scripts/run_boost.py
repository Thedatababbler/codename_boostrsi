#!/usr/bin/env python3
"""Run a boosting search (serial / branch) or the whole-workflow ablation, then report on test.

Refuses to run unless a passing pre-check summary is given (S1 gate), unless --skip-gate is set
(only meant for fake-model dry runs).

    python3.11 scripts/run_boost.py --mode serial --llm fake --skip-gate --out runs/serial_fake
    python3.11 scripts/run_boost.py --mode branch --precheck runs/precheck/summary.json ...
    python3.11 scripts/run_boost.py --mode branch --no-residual ...        # ablation 1
    python3.11 scripts/run_boost.py --mode global --eval-budget 24 ...     # ablation 2
    python3.11 scripts/run_boost.py --mode serial --no-tools ...           # ablation 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from boost.branch import fit_branch  # noqa: E402
from boost.common import BoostCfg  # noqa: E402
from boost.global_search import fit_global  # noqa: E402
from boost.serial import fit_serial  # noqa: E402
from farm.bench import standard_splits  # noqa: E402
from workflow.baselines import final_baselines, repair_loop, single_draft  # noqa: E402
from workflow.evaluate import evaluate  # noqa: E402
from workflow.experiment import add_common_args, compare, dump, make_env, method_metrics  # noqa: E402
from workflow.grow import GrowCfg  # noqa: E402
from workflow.model import Limits, to_str  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--mode", choices=["serial", "branch", "global"], required=True)
    ap.add_argument("--precheck", default=None, help="path to precheck summary.json (must PASS)")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.002)
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--eval-budget", type=int, default=None, help="global mode: candidate evals")
    ap.add_argument("--report-seed", type=int, default=1000)
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    if not args.skip_gate:
        if not args.precheck or not json.load(open(args.precheck)).get("pass"):
            raise SystemExit("S1 gate: a passing --precheck summary is required before search.")
    sp = standard_splits(args.difficulty, args.n_train)      # never touches sp["final"]
    env = make_env(args)
    cfg = BoostCfg(rounds=args.rounds, lam=args.lam, grow=GrowCfg(max_steps=args.max_steps),
                   limits=Limits(max_calls=args.max_calls), run_seed=args.seed,
                   report_seed=args.report_seed, no_residual=args.no_residual,
                   no_tools=args.no_tools)
    calls0 = env.llm.new_calls
    if args.mode == "serial":
        wf, info = fit_serial(sp["train"], sp["heldout"], env, cfg)
    elif args.mode == "branch":
        wf, info = fit_branch(sp["train"], sp["heldout"], env, cfg)
    else:
        wf, info = fit_global(sp["train"], sp["heldout"], env, cfg,
                              eval_budget=args.eval_budget or args.rounds * args.max_steps * 3)
    info["search_new_llm_calls"] = env.llm.new_calls - calls0
    print("final workflow:", to_str(wf))
    print(f"search: {info['evals']} candidate evals, {info['search_new_llm_calls']} new model calls")

    # heldout score for reporting, with fresh seeds (§6.4)
    ho = evaluate(wf, sp["heldout"], env, args.report_seed)
    info["heldout_report"] = {"mean_g": statistics.fmean(r.g for r in ho), "seed": args.report_seed}
    print(f"heldout (report seed) mean g = {info['heldout_report']['mean_g']:.3f}")

    os.makedirs(args.out, exist_ok=True)
    if args.skip_test:
        json.dump({"workflow": to_str(wf), **info}, open(os.path.join(args.out, "search.json"), "w"),
                  indent=1, default=str)
        return
    test = sp["test"]
    ours = evaluate(wf, test, env, args.report_seed)
    single = evaluate(single_draft(), test, env, args.report_seed)
    rep1 = evaluate(repair_loop(1), test, env, args.report_seed)
    methods = {"ours": wf, **final_baselines(ours, single, rep1)}
    results = {"ours": ours}
    for n, w in methods.items():
        if n != "ours":
            results[n] = evaluate(w, test, env, args.report_seed)
    dump(os.path.join(args.out, "test.json"), methods, results, {"search": info})
    rows = {n: [r.summary() for r in rs] for n, rs in results.items()}
    print(f"\n{'method':22s} {'mean_g':>7s} {'legal':>6s} {'tokens':>8s}   diff vs ours [90% CI]")
    for n, r in rows.items():
        m = method_metrics(r)
        c = compare(rows["ours"], r) if n != "ours" else None
        tail = f"   {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]" if c else ""
        print(f"{n:22s} {m['mean_g']:7.3f} {m['legal_rate']:6.2f} {m['tokens']:8.0f}{tail}")


if __name__ == "__main__":
    main()
