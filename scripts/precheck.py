#!/usr/bin/env python3
"""Pre-check (doc §7): three hand-written workflows vs call-matched best-of-n / tools best-of-n /
repair loop, on the 40 difficulty-5 calibration items (= the test split), one run, seed 0.

Gate (§7.3): continue only if at least one F has a paired-g 90% lower bound > 0 against the
stronger of B1 / B2 at the same number of calls.

    python3.11 scripts/precheck.py --estimate                       # worst-case call count only
    python3.11 scripts/precheck.py --llm fake --out runs/precheck_fake
    python3.11 scripts/precheck.py --llm openai --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-14B --cache runs/cache/qwen3.jsonl --out runs/precheck
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from farm.bench import standard_splits  # noqa: E402
from workflow.baselines import HANDWRITTEN, precheck_methods  # noqa: E402
from workflow.experiment import add_common_args, compare, dump, make_env, method_metrics, run_methods  # noqa: E402
from workflow.model import to_str, worst_case_calls  # noqa: E402


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


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--estimate", action="store_true", help="print worst-case call counts and exit")
    ap.add_argument("--calls-per-hour", type=float, default=64.0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N items (smoke runs)")
    args = ap.parse_args()

    items = standard_splits(args.difficulty)["test"][: args.limit]
    methods = precheck_methods()
    worst = sum(worst_case_calls(w) for w in methods.values()) * len(items)
    print(f"{len(methods)} methods x {len(items)} items; worst-case calls (no cache sharing) = {worst}"
          f" ~ {worst / args.calls_per_hour:.1f} h at {args.calls_per_hour}/h")
    for n, w in methods.items():
        print(f"  {n:24s} calls<={worst_case_calls(w)}  {to_str(w)}")
    if args.estimate:
        return

    env = make_env(args)
    t0 = time.time()
    rows_obj = run_methods(methods, items, env, args.seed)
    dt = time.time() - t0
    rows = {n: [r.summary() for r in rs] for n, rs in rows_obj.items()}
    g = gate(rows)
    summary = {"metrics": {n: method_metrics(r) for n, r in rows.items()}, "gate": g,
               "pass": any(x["pass"] for x in g.values()), "diagnosis": diagnose(g),
               "new_llm_calls": env.llm.new_calls, "cache_hits": env.llm.hits,
               "wall_seconds": dt, "llm": args.llm, "seed": args.seed}
    os.makedirs(args.out, exist_ok=True)
    dump(os.path.join(args.out, "precheck.json"), methods, rows_obj, {"summary": summary})
    print(f"\nnew model calls: {env.llm.new_calls}, cache hits: {env.llm.hits}, {dt:.0f}s")
    print(f"{'method':26s} {'mean_g':>7s} {'score':>6s} {'legal':>6s} {'calls':>6s} {'tokens':>8s}")
    for n, m in summary["metrics"].items():
        print(f"{n:26s} {m['mean_g']:7.3f} {m['mean_score']:6.3f} {m['legal_rate']:6.2f} "
              f"{m['llm_calls']:6.2f} {m['tokens']:8.0f}")
    print("\nGate (paired g vs stronger of B1/B2 at equal calls, 90% CI):")
    for n, x in g.items():
        print(f"  {n:20s} vs {x['vs']:22s} diff={x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] "
              f"{'PASS' if x['pass'] else 'fail'}")
    print("\nGATE:", "PASS - search experiments may start" if summary["pass"] else
          "FAIL - stop and report (do not start search)")
    for m in summary["diagnosis"]:
        print("  -", m)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
