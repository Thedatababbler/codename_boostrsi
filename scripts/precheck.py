#!/usr/bin/env python3
"""Pre-check (doc §7, revision §12): three hand-written workflows vs call-matched best-of-n / tools best-of-n /
repair loop, on the 40 d5 calibration items (split test_d5), one run, seed 0. Every method starts from the cached
y0 (a missing test y0 aborts; it is never generated).

Gate (§7.3): continue only if at least one F has a paired-g 90% lower bound > 0 against the stronger of B1 / B2
at the same number of calls.

    python scripts/precheck.py --config configs/workflow_d5.yaml --estimate   # cache-only, no model call
    python scripts/precheck.py --config configs/workflow_toy.yaml             # fake dry run
    python scripts/precheck.py --config configs/workflow_d5.yaml              # the real run (one job at a time)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from boosttopo.runlog import RunLog  # noqa: E402
from workflow.baselines import HANDWRITTEN, precheck_methods, seed_scheme_of  # noqa: E402
from workflow.evaluate import MissingY0, run_item, y0_cached  # noqa: E402
from workflow.experiment import (add_common_args, compare, dump, load_cfg, make_env, make_splits,  # noqa: E402
                                 method_metrics, run_methods, with_scheme)
from workflow.llm import CacheMiss  # noqa: E402
from workflow.model import to_str, worst_case_calls  # noqa: E402

SPLIT = "test_d5"
# nested families share samples (best-of-2's samples are best-of-4's first two, k=1's repair is k=3's first):
# only the largest member's calls are distinct
FAMILIES = ("B1_best_of_", "B2_tools_best_of_", "B3_repair_loop_k")


def gate(rows: dict) -> dict:
    out = {}
    for name, wf in HANDWRITTEN.items():
        c = worst_case_calls(wf)
        b1, b2 = f"B1_best_of_{c}", f"B2_tools_best_of_{c}"
        mg = {b: method_metrics(rows[b])["mean_g"] for b in (b1, b2)}
        strong = max(mg, key=mg.get)
        cmp_ = compare(rows[name], rows[strong])
        out[name] = {"calls": c, "vs": strong, **cmp_, "pass": cmp_["lo"] > 0,
                     "vs_single": compare(rows[name], rows["single_draft"]),
                     "vs_B1": compare(rows[name], rows[b1]), "vs_B2": compare(rows[name], rows[b2])}
    return out


def diagnose(g: dict) -> list[str]:
    """Map outcomes to the §7.3 decision table (advisory)."""
    msgs = []
    f1 = g.get("F1_sim_improve")
    if f1 and f1["vs_single"]["diff"] > 0 and f1["vs_B1"]["diff"] < 0:
        msgs.append("F1 beats single draft but not same-cost B1: one improve < one redraft -> "
                    "make simulate output more specific, re-test.")
    if all(abs(x["vs_B1"]["diff"]) < 0.02 for x in g.values()):
        msgs.append("All workflows tie with B1: no room for workflow structure on this task -> switch task.")
    b2_better = any(x["vs_B1"]["diff"] < x["vs_B2"]["diff"] for x in g.values())
    if b2_better and not any(x["vs_B2"]["lo"] > 0 for x in g.values()):
        msgs.append("If B2 >> B1 while no workflow beats B2: tool helps, structure doesn't -> make "
                    "crop_stats default and search for structural room.")
    return msgs


def estimate(cfg, items, methods, calls_per_hour: list[float]) -> dict:
    """Cache-only replay: per method and item, the calls served from the cache before the first miss; every
    remaining worst-case call counts as new (an upper bound: a cut-short loop or a pass-through branch costs
    less). No model call is made."""
    env = make_env(cfg, cache_only=True)
    env.parallel_fanout = False  # count hits per item deterministically
    missing = [it.item_id for it in items if not y0_cached(env.llm, it)]
    rep = {"items": len(items), "y0_missing": missing, "methods": {}}
    if missing:
        rep["note"] = "y0 missing from the cache: the pre-check would abort (test y0 is never generated)"
        return rep
    env.make_start(items, require_cached=True)
    for name, wf in methods.items():
        e = with_scheme(env, seed_scheme_of(name))
        worst = worst_case_calls(wf)
        served = new = 0
        for it in items:
            before = e.llm.budget.logical_calls
            try:
                r = run_item(wf, it, e)
                served += r.cost.llm_calls
            except CacheMiss:
                hits = e.llm.budget.logical_calls - before  # cache hits of this item before the miss
                served += max(0, hits)
                new += worst - max(0, hits)
        rep["methods"][name] = {"seed_scheme": seed_scheme_of(name), "worst_case_calls": worst * len(items),
                                "cached": served, "new_upper": new}
    tot_new = tot_cached = 0
    for name, m in rep["methods"].items():
        fam = next((f for f in FAMILIES if name.startswith(f)), None)
        if fam and any(n.startswith(fam) and worst_case_calls(methods[n]) > worst_case_calls(methods[name])
                       for n in methods):
            continue  # contained in a larger family member
        tot_new += m["new_upper"]
        tot_cached += m["cached"]
    rep["distinct_new_upper"] = tot_new
    rep["distinct_cached"] = tot_cached
    rep["hours"] = {f"{c:g}/h": tot_new / c for c in calls_per_hour}
    return rep


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--estimate", action="store_true", help="cache-only call estimate; no model call")
    ap.add_argument("--calls-per-hour", type=float, action="append", default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N items (smoke runs)")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.set)
    out = args.out or os.path.join(ROOT, "runs", "precheck_" + os.path.splitext(os.path.basename(args.config))[0])
    methods = precheck_methods()
    for n, w in methods.items():
        print(f"  {n:24s} calls<={worst_case_calls(w)}  seeds={seed_scheme_of(n):6s}  {to_str(w)}")

    env = make_env(cfg, cache_only=args.estimate)
    items = make_splits(cfg, env.bench)[SPLIT][: args.limit]
    if args.estimate:
        rep = estimate(cfg, items, methods, args.calls_per_hour or [64.0, 200.0])
        print(json.dumps({k: v for k, v in rep.items() if k != "methods"}, indent=1))
        for n, m in rep["methods"].items():
            print(f"  {n:24s} cached={m['cached']:4d}  new<={m['new_upper']:4d}")
        return

    need = SPLIT in (cfg.workflow.get("require_cached_y0") or [])
    try:
        env.make_start(items, require_cached=need)
    except MissingY0 as exc:
        raise SystemExit(f"abort: {exc}")
    t0 = time.time()
    rows_obj = run_methods(methods, items, env, args.seed, scheme_of=seed_scheme_of)
    dt = time.time() - t0
    rows = {n: [r.summary() for r in rs] for n, rs in rows_obj.items()}
    g = gate(rows)
    b = env.llm.budget.snapshot()
    summary = {"metrics": {n: method_metrics(r) for n, r in rows.items()}, "gate": g,
               "pass": any(x["pass"] for x in g.values()), "diagnosis": diagnose(g),
               "budget": b, "wall_seconds": dt, "llm": cfg.llm.kind, "seed": args.seed, "split": SPLIT}
    log = RunLog(out)
    dump(os.path.join(out, "precheck.json"), methods, rows_obj, {"summary": summary})
    with open(os.path.join(out, "per_item.jsonl"), "w") as f:
        for n, rs in rows_obj.items():
            for r in rs:
                f.write(json.dumps({"split": SPLIT, "method": n, "item_id": r.item_id, "g": r.g, "score": r.score,
                                    "valid": r.legal, "category": r.category, **r.cost.as_dict()}) + "\n")
    log.write("summary.json", summary)
    print(f"\nnew model calls: {b['new_calls']}, logical calls: {b['logical_calls']}, {dt:.0f}s")
    print(f"{'method':26s} {'mean_g':>7s} {'score':>6s} {'legal':>6s} {'calls':>6s} {'tokens':>8s}")
    for n, m in summary["metrics"].items():
        print(f"{n:26s} {m['mean_g']:7.3f} {m['mean_score']:6.3f} {m['legal_rate']:6.2f} "
              f"{m['llm_calls']:6.2f} {m['tokens']:8.0f}")
    print("\nGate (paired g vs stronger of B1/B2 at equal calls, two-sided 90% CI):")
    for n, x in g.items():
        print(f"  {n:20s} vs {x['vs']:22s} diff={x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] "
              f"{'PASS' if x['pass'] else 'fail'}")
    print("\nGATE:", "PASS - search experiments may start" if summary["pass"] else
          "FAIL - stop and report (do not start search)")
    for m in summary["diagnosis"]:
        print("  -", m)


if __name__ == "__main__":
    main()
