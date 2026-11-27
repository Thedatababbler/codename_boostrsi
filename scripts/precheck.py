#!/usr/bin/env python3
"""Pre-check (doc §7; revision §12; plan of 2026-11-27): three hand-written workflows F1-F3 against single draft and
call-matched baselines on the 40 d5 test items (seeds 6000-6039), one repeat, seed 0. Every method starts from y0,
which is generated here (repeat 0 of the benchmark prompt) and charged to every method.

Baselines at each F's worst-case call count c: B1 best-of-c, B2 crop_stats + best-of-c, B3 draft + (c-1) repairs
(clean diagnosis), B4 draft + (c-1) improves (the fixed baseline seq[draft, loop(k, stop=best)[improve]]).
All methods use seed_scheme path, so identical requests are shared across methods (common random numbers).

Execution: y0 for every item first, then one thread pool over all (method, item) pairs. Every finished pair is
appended to <out>/checkpoint.jsonl; a restarted run skips those pairs, and the calls of unfinished pairs that did
finish are served from the shared LLM cache. Model calls are logged to <out>/calls.jsonl.

Gate (§7.3, one-sided per revision §12): an F passes if the 10th percentile of the paired-g bootstrap against the
stronger of B1 / B2 at the same number of calls is > 0. Informational only: no search is started from here.

    python scripts/precheck.py --config configs/workflow_toy.yaml                  # fake dry run
    python scripts/precheck.py --config configs/workflow_d5.yaml --set llm.api_base=http://h:p/v1,http://h:q/v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from boosttopo.runlog import RunLog  # noqa: E402
from workflow.baselines import HANDWRITTEN, precheck_methods, seed_scheme_of  # noqa: E402
from workflow.evaluate import run_item  # noqa: E402
from workflow.experiment import (add_common_args, compare, load_cfg, make_env, make_splits,  # noqa: E402
                                 method_metrics, with_scheme)
from workflow.llm import route  # noqa: E402
from workflow.model import to_str, worst_case_calls  # noqa: E402

SPLIT = "test_d5"


def baselines_at(c: int) -> list[str]:
    return [f"B1_best_of_{c}", f"B2_tools_best_of_{c}", f"B3_repair_loop_k{c - 1}", f"B4_improve_loop_k{c - 1}"]


def gate(rows: dict) -> dict:
    out = {}
    for name, wf in HANDWRITTEN.items():
        c = worst_case_calls(wf)
        b1, b2 = f"B1_best_of_{c}", f"B2_tools_best_of_{c}"
        mg = {b: method_metrics(rows[b])["mean_g"] for b in (b1, b2)}
        strong = max(mg, key=mg.get)
        cmp_ = compare(rows[name], rows[strong])
        out[name] = {"calls": c, "vs": strong, **cmp_, "pass": cmp_["p10"] > 0}
    return out


def all_pairs(rows: dict) -> dict:
    """Every F against every baseline (single draft and B1-B4 at every call count)."""
    base = [n for n in rows if n not in HANDWRITTEN]
    return {f: {b: compare(rows[f], rows[b]) for b in base} for f in HANDWRITTEN if f in rows}


def diagnose(rows: dict, g: dict) -> list[str]:
    """Map outcomes to the §7.3 decision table (advisory)."""
    msgs = []
    c = {f: worst_case_calls(HANDWRITTEN[f]) for f in g}
    vs = lambda f, b: compare(rows[f], rows[b])  # noqa: E731
    f1 = "F1_sim_improve"
    if f1 in g and vs(f1, "single_draft")["diff"] > 0 and vs(f1, f"B1_best_of_{c[f1]}")["diff"] < 0:
        msgs.append("F1 beats single draft but not same-cost B1: one improve < one redraft -> "
                    "make simulate output more specific, re-test.")
    if all(abs(vs(f, f"B1_best_of_{c[f]}")["diff"]) < 0.02 for f in g):
        msgs.append("All workflows tie with B1: no room for workflow structure on this task -> switch task.")
    b2_better = any(vs(f, f"B1_best_of_{c[f]}")["diff"] < vs(f, f"B2_tools_best_of_{c[f]}")["diff"] for f in g)
    if b2_better and not any(vs(f, f"B2_tools_best_of_{c[f]}")["p10"] > 0 for f in g):
        msgs.append("If B2 >> B1 while no workflow beats B2: tool helps, structure doesn't -> make "
                    "crop_stats default and search for structural room.")
    return msgs


def load_checkpoint(path: str) -> dict:
    done: dict = {}
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut by a kill
                done[(r["method"], r["item"])] = r["summary"]
    return done


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N items (smoke runs)")
    ap.add_argument("--pair-workers", type=int, default=None,
                    help="concurrent (method, item) pairs (default: servers x llm.max_concurrency)")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.set)
    out = args.out or os.path.join(ROOT, "runs", "precheck_" + os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out, exist_ok=True)
    methods = precheck_methods()
    for n, w in methods.items():
        print(f"  {n:24s} calls<={worst_case_calls(w)}  seeds={seed_scheme_of(n):6s}  {to_str(w)}")

    env = make_env(cfg, call_log=os.path.join(out, "calls.jsonl"))
    items = make_splits(cfg, env.bench)[SPLIT][: args.limit]
    route(env, items)
    n_srv = len(getattr(env.llm.inner, "clients", None) or getattr(getattr(env.llm.inner, "inner", None),
                                                                     "clients", None) or [1])
    t0 = time.time()
    env.make_start(items)  # y0 (generated on a miss), frozen for every method
    t_y0 = time.time() - t0
    print(f"y0 ready for {len(items)} items in {t_y0:.0f}s; new calls so far {env.llm.budget.new_calls}", flush=True)

    ck_path = os.path.join(out, "checkpoint.jsonl")
    done = load_checkpoint(ck_path)
    todo = [(n, it) for n in methods for it in items if (n, it.item_id) not in done]
    print(f"pairs: {len(methods) * len(items)} total, {len(done)} from the checkpoint, {len(todo)} to run",
          flush=True)
    envs = {n: with_scheme(env, seed_scheme_of(n)) for n in methods}
    lock = threading.Lock()
    workers = args.pair_workers or n_srv * int(cfg.llm.get("max_concurrency", 8))

    def one(pair):
        n, it = pair
        return n, it, run_item(methods[n], it, envs[n], args.seed)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, open(ck_path, "a") as ck:
        futs = [ex.submit(one, p) for p in todo]
        for k, f in enumerate(as_completed(futs), 1):
            n, it, r = f.result()
            s = r.summary()
            with lock:
                done[(n, it.item_id)] = s
                ck.write(json.dumps({"method": n, "item": it.item_id, "summary": s}) + "\n")
                ck.flush()
            if k % 20 == 0 or k == len(futs):
                print(f"  {k}/{len(futs)} pairs, new calls {env.llm.budget.new_calls}, "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    dt = time.time() - t0

    ids = [it.item_id for it in items]
    rows = {n: [done[(n, i)] for i in ids] for n in methods}
    g = gate(rows)
    b = env.llm.budget.snapshot()
    summary = {"metrics": {n: method_metrics(r) for n, r in rows.items()}, "gate": g,
               "pass": any(x["pass"] for x in g.values()), "diagnosis": diagnose(rows, g),
               "pairs": all_pairs(rows), "budget": b, "wall_seconds": dt, "y0_seconds": t_y0,
               "servers": n_srv, "llm": {k: cfg.llm.get(k) for k in ("model", "temperature", "max_tokens",
                                                                     "extra_body", "cache_tag", "api_base")},
               "seed": args.seed, "split": SPLIT, "items": len(items)}
    log = RunLog(out)
    with open(os.path.join(out, "precheck.json"), "w") as f:
        json.dump({"methods": {n: to_str(w) for n, w in methods.items()}, "results": rows, "summary": summary},
                  f, indent=1, default=str)
    with open(os.path.join(out, "per_item.jsonl"), "w") as f:
        for n in methods:
            for r in rows[n]:
                f.write(json.dumps({"split": SPLIT, "method": n, "item_id": r["item"], "g": r["g"],
                                    "score": r["score"], "valid": r["legal"], "category": r["category"],
                                    **r["cost"]}) + "\n")
    log.write("summary.json", summary)
    print(f"\nnew model calls: {b['new_calls']}, logical calls: {b['logical_calls']}, {dt:.0f}s")
    print(f"{'method':26s} {'mean_g':>7s} {'score':>6s} {'legal':>6s} {'calls':>6s} {'tokens':>8s}")
    for n, m in summary["metrics"].items():
        print(f"{n:26s} {m['mean_g']:7.3f} {m['mean_score']:6.3f} {m['legal_rate']:6.2f} "
              f"{m['llm_calls']:6.2f} {m['tokens']:8.0f}")
    print("\nPaired g differences (F - baseline): mean [two-sided 90%] p10  win/tie/loss")
    for f_, d in summary["pairs"].items():
        c = worst_case_calls(HANDWRITTEN[f_])
        for b_ in ["single_draft"] + baselines_at(c):
            x = d[b_]
            print(f"  {f_:18s} - {b_:22s} {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] {x['p10']:+.3f}  "
                  f"{x['win']}/{x['tie']}/{x['loss']}")
    print("\nGate (vs stronger of B1/B2 at equal calls, one-sided p10 > 0; informational):")
    for n, x in g.items():
        print(f"  {n:20s} vs {x['vs']:22s} diff={x['diff']:+.3f} p10={x['p10']:+.3f} "
              f"{'PASS' if x['pass'] else 'fail'}")
    for m in summary["diagnosis"]:
        print("  -", m)


if __name__ == "__main__":
    main()
