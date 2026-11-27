#!/usr/bin/env python3
"""Run a boosting search (serial / branch) or the whole-workflow ablation, then report on the test split.

Refuses to run unless a passing pre-check summary is given (S1 gate), unless --skip-gate is set (only meant for
fake-model dry runs). Writes config.json, search.json, checkpoint_<mode>.json (resume: rerun the same command),
test.json and per_item.jsonl into the run directory. Never touches the final-test splits.

    python scripts/run_boost.py --config configs/workflow_toy.yaml --mode serial --skip-gate
    python scripts/run_boost.py --config configs/workflow_d5.yaml --mode branch --precheck runs/.../summary.json
    ... --no-residual (ablation 1) | --mode global --eval-budget N (ablation 2) | --no-tools (ablation 3)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from boosttopo.config import cfg_hash  # noqa: E402
from boosttopo.runlog import RunLog  # noqa: E402
from boost.branch import fit_branch  # noqa: E402
from boost.common import BoostCfg  # noqa: E402
from boost.global_search import fit_global  # noqa: E402
from boost.serial import fit_serial  # noqa: E402
from workflow.baselines import final_baselines, repair_loop, single_draft  # noqa: E402
from workflow.evaluate import MissingY0, evaluate  # noqa: E402
from workflow.experiment import (add_common_args, compare, dump, load_cfg, make_env, make_splits,  # noqa: E402
                                 method_metrics)
from workflow.grow import GrowCfg  # noqa: E402
from workflow.model import Limits, to_str  # noqa: E402

TEST = "test_d5"


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--mode", choices=["serial", "branch", "global"], required=True)
    ap.add_argument("--precheck", default=None, help="path to precheck summary.json (must PASS)")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--eval-budget", type=int, default=None, help="global mode: candidate evals")
    ap.add_argument("--report-seed", type=int, default=1000)
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    if not args.skip_gate:
        if not args.precheck or not json.load(open(args.precheck)).get("pass"):
            raise SystemExit("S1 gate: a passing --precheck summary is required before search.")
    cfg = load_cfg(args.config, args.set)
    out = args.out or os.path.join(ROOT, "runs", f"{os.path.splitext(os.path.basename(args.config))[0]}_{args.mode}"
                                                 f"_s{args.seed}")
    log = RunLog(out)
    log.write("config.json", {"config": cfg.to_dict(), "hash": cfg_hash(cfg.to_dict()), "args": vars(args)})
    env = make_env(cfg)
    sp = make_splits(cfg, env.bench)                       # never builds the final-test splits
    b = cfg.boost
    bc = BoostCfg(rounds=b.rounds, patience=b.patience, lam=cfg.lam, accept_margin=float(cfg.accept_margin),
                  grow=GrowCfg(max_steps=b.max_steps, top_k=b.top_k, min_gain=float(cfg.search.min_gain)),
                  limits=Limits(max_calls=int(cfg.workflow.max_calls)), run_seed=args.seed,
                  report_seed=args.report_seed, no_residual=args.no_residual, no_tools=args.no_tools,
                  fresh_after_reject=bool(b.fresh_after_reject))
    env.make_start(sp["train"] + sp["heldout"])
    calls0 = env.llm.budget.new_calls
    if args.mode == "serial":
        wf, info = fit_serial(sp["train"], sp["heldout"], env, bc, runlog=log)
    elif args.mode == "branch":
        wf, info = fit_branch(sp["train"], sp["heldout"], env, bc, runlog=log)
    else:
        wf, info = fit_global(sp["train"], sp["heldout"], env, bc,
                              eval_budget=args.eval_budget or b.rounds * b.max_steps * 3)
    info["search_new_llm_calls"] = env.llm.budget.new_calls - calls0
    print("final workflow:", to_str(wf))
    print(f"search: {info['evals']} candidate evals, {info['search_new_llm_calls']} new model calls, lam={bc.lam}")

    # heldout score for reporting, with fresh seeds (§6.4)
    ho = evaluate(wf, sp["heldout"], env, args.report_seed)
    info["heldout_report"] = {"mean_g": statistics.fmean(r.g for r in ho), "seed": args.report_seed}
    print(f"heldout (report seed) mean g = {info['heldout_report']['mean_g']:.3f}")
    log.write("search.json", {"workflow": to_str(wf), **info})
    if args.skip_test:
        return
    test = sp[TEST]
    try:
        env.make_start(test, require_cached=TEST in (cfg.workflow.get("require_cached_y0") or []))
    except MissingY0 as exc:
        raise SystemExit(f"abort: {exc}")
    ours = evaluate(wf, test, env, args.report_seed)
    single = evaluate(single_draft(), test, env, args.report_seed)
    rep1 = evaluate(repair_loop(1), test, env, args.report_seed)
    methods = {"ours": wf, **final_baselines(ours, single, rep1)}
    results = {"ours": ours}
    for n, w in methods.items():
        if n != "ours":
            results[n] = evaluate(w, test, env, args.report_seed)
    dump(os.path.join(out, "test.json"), methods, results, {"search": info, "budget": env.llm.budget.snapshot()})
    with open(os.path.join(out, "per_item.jsonl"), "w") as f:
        for n, rs in results.items():
            for r in rs:
                f.write(json.dumps({"split": TEST, "method": n, "item_id": r.item_id, "g": r.g, "score": r.score,
                                    "valid": r.legal, "category": r.category, "s0": env.s0[r.item_id],
                                    **r.cost.as_dict()}) + "\n")
    rows = {n: [r.summary() for r in rs] for n, rs in results.items()}
    print(f"\n{'method':22s} {'mean_g':>7s} {'legal':>6s} {'tokens':>8s}   diff vs ours [two-sided 90% CI]")
    for n, r in rows.items():
        m = method_metrics(r)
        c = compare(rows["ours"], r) if n != "ours" else None
        tail = f"   {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]" if c else ""
        print(f"{n:22s} {m['mean_g']:7.3f} {m['legal_rate']:6.2f} {m['tokens']:8.0f}{tail}")


if __name__ == "__main__":
    main()
