#!/usr/bin/env python3
"""Smoke test before the pre-check (plan of 2026-11-27, step 2), on the 12 d5 train items (seeds 1000-1011):

1. y0 and one writer call (seq[draft] from y0) per item: legal rate and mean score of y0 and of the writer's own
   plan, tokens per call, truncated share (finish_reason = length), measured calls per hour per GPU;
2. improve on 5 train items starting from a legal plan (the item's y0 if legal, else its writer plan), with
   simulate's output in the prompt: seq[simulate, improve]. Prompts and raw outputs are saved.

Stop rule (reported in smoke.json["stop"]): y0 or writer legal rate < 0.3, more than 20% of calls truncated, or all
5 improve outputs illegal. The calls go through the shared cache (cache_dir), so the pre-check reuses nothing from
here: train items are not test items.

    python scripts/smoke.py --config configs/workflow_d5.yaml --set llm.api_base=http://host:port/v1 --gpus 1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from workflow.baselines import single_draft  # noqa: E402
from workflow.evaluate import run_item  # noqa: E402
from workflow.exec import start_state  # noqa: E402
from workflow.experiment import add_common_args, load_cfg, make_env, make_splits  # noqa: E402
from workflow.llm import route  # noqa: E402
from workflow.model import parse_wf  # noqa: E402
from workflow.tools import simulate  # noqa: E402

N_IMPROVE = 5


def calls_since(path: str, t: float) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [r for r in map(json.loads, open(path)) if r["t_end"] >= t]


def call_stats(calls: list[dict], seconds: float, gpus: int) -> dict:
    if not calls:
        return {"calls": 0}
    tr = [c for c in calls if c.get("finish_reason") == "length"]
    comp = [c["completion_tokens"] for c in calls if c.get("completion_tokens") is not None]
    return {"calls": len(calls), "tokens_per_call": statistics.fmean(c["tokens"] for c in calls),
            "completion_tokens_per_call": statistics.fmean(comp) if comp else None,
            "max_completion_tokens": max(comp) if comp else None,
            "truncated": len(tr), "truncated_share": len(tr) / len(calls),
            "seconds_per_call_mean": (statistics.fmean(c["seconds"] for c in calls if "seconds" in c)
                                      if any("seconds" in c for c in calls) else None),
            "wall_seconds": seconds, "calls_per_hour_per_gpu": len(calls) / (seconds / 3600) / gpus}


def main():
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--gpus", type=int, default=1, help="GPUs serving the model (for calls / hour / GPU)")
    args = ap.parse_args()
    cfg = load_cfg(args.config, args.set)
    out = args.out or os.path.join(ROOT, "runs", "smoke_" + os.path.splitext(os.path.basename(args.config))[0])
    os.makedirs(out, exist_ok=True)
    log = os.path.join(out, "calls.jsonl")
    env = make_env(cfg, call_log=log)
    items = make_splits(cfg, env.bench)["train"]
    route(env, items)
    assert [it.seed for it in items] == list(range(1000, 1012)) or cfg.llm.kind == "fake", [it.seed for it in items]
    rep: dict = {"items": [it.item_id for it in items], "llm": {k: cfg.llm.get(k) for k in (
        "model", "temperature", "max_tokens", "extra_body", "cache_tag", "api_base")}, "gpus": args.gpus}

    # 1a. y0
    t0 = time.time()
    env.make_start(items)
    t1 = time.time()
    y0 = {it.item_id: env.bench.score(it, env.y0[it.item_id]) for it in items}
    rep["y0"] = {"legal_rate": statistics.fmean(s.valid for s in y0.values()),
                 "mean_score": statistics.fmean(s.score for s in y0.values()),
                 "categories": {i: s.category for i, s in y0.items()},
                 **call_stats(calls_since(log, t0), t1 - t0, args.gpus)}
    print("y0:", json.dumps({k: v for k, v in rep["y0"].items() if k != "categories"}), flush=True)

    # 1b. one writer call on y0 (the writer's own plan is st.plan; the method's output would be the better one)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        res = list(ex.map(lambda it: run_item(single_draft(), it, env, args.seed), items))
    t2 = time.time()
    w = {r.item_id: r for r in res}
    rep["writer"] = {"legal_rate": statistics.fmean(r.state.plan_legal for r in res),
                     "mean_score": statistics.fmean(r.state.plan_score for r in res),
                     "method_mean_g": statistics.fmean(r.g for r in res),
                     "categories": {r.item_id: r.state.plan_cat for r in res},
                     **call_stats(calls_since(log, t1), t2 - t1, args.gpus)}
    print("writer:", json.dumps({k: v for k, v in rep["writer"].items() if k != "categories"}), flush=True)

    # 2. improve from a legal plan, simulate's output in the prompt
    starts = []
    for it in items:
        if y0[it.item_id].valid:
            starts.append((it, "y0", env.y0[it.item_id]))
        elif w[it.item_id].state.plan_legal:
            starts.append((it, "writer", w[it.item_id].state.plan))
    starts = starts[:N_IMPROVE]
    env.llm.inner.keep_text = True
    wf = parse_wf("seq[simulate, improve]")

    def imp(a):
        it, src, plan = a
        st = start_state(it, plan, env.bench.score(it, plan))
        r = run_item(wf, it, env, args.seed, start=st)
        return it, src, plan, st, r

    with ThreadPoolExecutor(max_workers=N_IMPROVE) as ex:
        outs = list(ex.map(imp, starts))
    t3 = time.time()
    env.llm.inner.keep_text = False
    calls = {(c["item"], c["role"]): c for c in calls_since(log, t2)}
    rows = []
    for it, src, plan, st, r in outs:
        ctx = env.ctx(it)
        c = next((v for (i, _), v in calls.items() if i == it.item_id), {})
        sr = env.bench.score(it, r.state.plan)
        rows.append({"item": it.item_id, "start_from": src, "start_score": st.plan_score,
                     "simulate_output": simulate(st, ctx), "improve_prompt": c.get("prompt"),
                     "improve_raw_output": c.get("output"), "finish_reason": c.get("finish_reason"),
                     "improved_plan": r.state.plan, "improved_legal": sr.valid, "improved_score": sr.score,
                     "improved_diag": sr.diag if not sr.valid else None, "category": sr.category})
    rep["improve"] = {"n": len(rows), "legal": sum(r["improved_legal"] for r in rows),
                      "better_than_start": sum(r["improved_score"] > r["start_score"] for r in rows),
                      **call_stats(calls_since(log, t2), t3 - t2, args.gpus)}
    with open(os.path.join(out, "improve_io.json"), "w") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print("improve:", json.dumps(rep["improve"]), flush=True)

    allc = calls_since(log, t0)
    rep["all_calls"] = call_stats(allc, t3 - t0, args.gpus)
    reasons = []
    if rep["y0"]["legal_rate"] < 0.3:
        reasons.append(f"y0 legal rate {rep['y0']['legal_rate']:.2f} < 0.3")
    if rep["writer"]["legal_rate"] < 0.3:
        reasons.append(f"writer legal rate {rep['writer']['legal_rate']:.2f} < 0.3")
    if rep["all_calls"].get("truncated_share", 0) > 0.2:
        reasons.append(f"truncated share {rep['all_calls']['truncated_share']:.2f} > 0.2")
    if rows and not any(r["improved_legal"] for r in rows):
        reasons.append("all improve outputs illegal")
    notes = [f"only {len(rows)} train items have a legal starting plan (wanted {N_IMPROVE})"] \
        if len(rows) < N_IMPROVE else []
    rep["stop"] = {"stop": bool(reasons), "reasons": reasons, "notes": notes}
    with open(os.path.join(out, "smoke.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("STOP" if reasons else "OK", reasons, flush=True)


if __name__ == "__main__":
    main()
