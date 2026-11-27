#!/usr/bin/env python3
"""Cache-only acceptance checks for the workflow integration (revision §12). Makes no model call: every client is
wrapped in workflow.llm.CacheOnly, so a missing record raises instead of generating. Writes nothing into the run
directories it reads.

For one stored multi-agent run directory (config.json + per_item.jsonl [+ final.json, ensemble.json]):
  (b)  the original code (boosttopo.infer) recomputes every stored row of the split, with the stored eval cache
       ("b") and with the LLM cache only ("b_llm_only": a fresh, empty eval cache);
  (a)  each method written as a workflow (seed_scheme legacy, diag_mode original) reproduces the stored row
       (score, g, tokens) and the original code's final plan, item by item;
  (c)  on every plan seen (y0 and each method's answer), simulate's profit equals the scorer's, and no tool output
       contains the reference profit or the relative score.

    PYTHONPATH=src:. python scripts/replay_check.py --run-dir runs/r4_d5 [--split test_d5] [--out runs/replay_r4_d5]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from boosttopo.bench.base import make_bench  # noqa: E402
from boosttopo.config import Cfg, load_config  # noqa: E402
from boosttopo.evalx import Evaluator  # noqa: E402
from boosttopo.graph import Subgraph  # noqa: E402
from boosttopo.infer import (SEED_STRIDE, best_of_n, best_of_n_subgraph, hand_topology, infer,  # noqa: E402
                             repair_loop, writer_role)
from boosttopo.run import make_splits  # noqa: E402
from workflow import legacy  # noqa: E402
from workflow.evaluate import Env, run_item  # noqa: E402
from workflow.exec import Ctx  # noqa: E402
from workflow.llm import CacheMiss, make_llm  # noqa: E402
from workflow.model import State, to_str  # noqa: E402
from workflow.tools import crop_stats, simulate, validate  # noqa: E402

KEYS = ("score", "g", "tokens", "valid")


class Recorder:
    """Bench wrapper: remembers the last solution scored per item (the answer of every boosttopo method is the
    last solution it scores for that item)."""

    def __init__(self, bench):
        self.b = bench
        self.last: dict[str, str] = {}
        self._lock = threading.Lock()

    def score(self, item, solution):
        with self._lock:
            self.last[item.item_id] = solution
        return self.b.score(item, solution)

    def __getattr__(self, k):
        return getattr(self.b, k)


def stored(run_dir: Path, split: str):
    rows = [json.loads(l) for l in (run_dir / "per_item.jsonl").read_text().splitlines() if l.strip()]
    by: dict[str, dict] = {}
    for r in rows:
        if r.get("split", split) == split and r.get("rep", 0) == 0:
            by.setdefault(r["method"], {})[r["item_id"]] = r
    return by


def method_spec(m: str, final: dict, split: str, ensemble, cfg):
    """(original-code callable, workflow or a NotExpressible reason) for a stored method name."""
    sec = (final.get("splits") or {}).get(split, {}).get(m, {})
    n_rep = 1
    if m == "writer_once":
        return (lambda items, ev: infer("adaboost", [writer_role()], items, ev, n_rep)), legacy.writer_once()
    if m in ("best_of_5", "best_of_n_writer"):
        n = 5 if m == "best_of_5" else int(sec["n_samples"])
        return (lambda items, ev: best_of_n_subgraph(writer_role(), items, ev, n)), legacy.best_of_n_writer(n)
    if m == "repair_loop5":
        return (lambda items, ev: repair_loop(items, ev, math.inf, 1, max_repairs=5)), legacy.repair_loop_k(5)
    if m == "repair_loop":
        b = float(sec["token_budget"])
        return (lambda items, ev: repair_loop(items, ev, b, n_rep)), legacy.repair_loop(b)
    if m == "ours":
        mode = ensemble["mode"]
        members = [Subgraph.from_dict(d) for d in ensemble["ensemble"]]
        orig = lambda items, ev: infer(mode, members, items, ev, n_rep)  # noqa: E731
        if mode != "gbdt":
            return orig, f"mode {mode}: only GBDT chains are written as one workflow"
        try:
            return orig, legacy.ensemble_workflow(members, bool(cfg.get("fresh_duplicate_stages", False)))
        except legacy.NotExpressible as exc:
            return orig, str(exc)
    if m == "hand_topology":
        orig = lambda items, ev: infer("adaboost", [hand_topology()], items, ev, n_rep)  # noqa: E731
        try:
            return orig, legacy.ensemble_workflow([hand_topology()])
        except legacy.NotExpressible as exc:
            return orig, str(exc)
    if m == "single_writer":
        return (lambda items, ev: infer("adaboost", [], items, ev, 1)), legacy.seq()
    if m == "best_of_n_matched":
        n = int(sec["n_samples"])
        return (lambda items, ev: best_of_n(items, ev, n, seed=int(cfg.seed))), \
            "best_of_n_matched resamples the benchmark prompt (y0 request), not a workflow skill"
    return None, f"unknown method {m}"


def cmp_rows(got: dict, want: dict) -> list[str]:
    bad = []
    for iid, w in want.items():
        g = got.get(iid)
        if g is None:
            bad.append(f"{iid}: missing")
            continue
        for k in KEYS:
            if float(g[k]) != float(w[k]):
                bad.append(f"{iid}: {k} {g[k]} != stored {w[k]}")
    return bad


def make_ev(cfg, bench, llm, eval_dir):
    ev = Evaluator(bench, llm, eval_dir, n_repeats=1, max_workers=int(cfg.llm.get("max_concurrency", 8)))
    ev.dedup_stages = bool(cfg.get("fresh_duplicate_stages", False))
    ev.rep_seed = int(cfg.seed) * SEED_STRIDE
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--split", default="test_d5")
    ap.add_argument("--cache-dir", default=None, help="default: the run config's cache_dir")
    ap.add_argument("--methods", default=None, help="comma list (default: every stored method of the split)")
    ap.add_argument("--out", default=None, help="report directory (default runs/replay_<run>)")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    cfg = Cfg.wrap(json.loads((run_dir / "config.json").read_text())["config"])
    cfg = load_config(None, cfg.to_dict())
    if a.cache_dir:
        cfg.cache_dir = a.cache_dir
    final = json.loads((run_dir / "final.json").read_text()) if (run_dir / "final.json").exists() else {}
    ensemble = json.loads((run_dir / "ensemble.json").read_text()) if (run_dir / "ensemble.json").exists() else None
    want = stored(run_dir, a.split)
    methods = a.methods.split(",") if a.methods else sorted(want)

    bench = make_bench(cfg)
    rec = Recorder(bench)
    llm = make_llm(cfg, cache_only=True)
    items = make_splits(cfg, bench)[a.split]
    report = {"run_dir": str(run_dir), "split": a.split, "n_items": len(items), "methods": {}}

    ev_b = make_ev(cfg, rec, llm, cfg.cache_dir)  # stored eval cache
    tmp = tempfile.mkdtemp(prefix="replay_eval_")
    ev_l = make_ev(cfg, rec, llm, tmp)            # empty eval cache: LLM cache only
    env = Env(llm, bench, seed_scheme="legacy", diag_mode="original", workers=ev_b.pool._max_workers,
              parallel_fanout=False)
    try:
        ev_b.make_start(items)
        ev_l.make_start(items)
        env.make_start(items, require_cached=True)
    except CacheMiss as exc:
        raise SystemExit(f"y0 not in the cache: {exc}")
    plans: dict[str, set] = {it.item_id: {env.y0[it.item_id]} for it in items}

    for m in methods:
        res: dict = {}
        orig, wf = method_spec(m, final, a.split, ensemble, cfg)
        if orig is None:
            report["methods"][m] = {"skipped": wf}
            continue
        for tag, ev in (("b", ev_b), ("b_llm_only", ev_l)):
            rec.last.clear()
            try:
                out = orig(items, ev)
                got = {r["item_id"]: r for r in out["rows"] if r.get("rep", 0) == 0}
                res[tag] = {"mismatches": cmp_rows(got, want[m])}
            except CacheMiss as exc:
                res[tag] = {"cache_miss": str(exc)}
            if tag == "b_llm_only":
                orig_plan = dict(rec.last)
        if isinstance(wf, str):
            res["a"] = {"not_expressible": wf}
        else:
            got, bad = {}, []
            for it in items:
                try:
                    r = run_item(wf, it, env, run_seed=int(cfg.seed))
                except CacheMiss as exc:
                    bad.append(f"{it.item_id}: cache miss ({exc})")
                    continue
                got[it.item_id] = {"score": r.score, "g": r.g, "tokens": r.cost.tokens, "valid": float(r.legal)}
                plans[it.item_id].add(r.plan)
                if it.item_id in orig_plan and r.plan != orig_plan[it.item_id]:
                    bad.append(f"{it.item_id}: plan differs from the original code's answer")
            res["a"] = {"workflow": to_str(wf), "mismatches": cmp_rows(got, want[m]) + bad}
        report["methods"][m] = res
        print(m, {k: (len(v.get("mismatches", [])) if "mismatches" in v else v) for k, v in res.items()}, flush=True)

    # (c) simulate vs scorer on every plan seen; tool outputs free of the reference
    by_id = {it.item_id: it for it in items}
    c_bad, n_legal = [], 0
    for iid, ps in (plans.items() if hasattr(bench, "simulate") else ()):  # (c) needs the Farm simulator
        it = by_id[iid]
        for p in ps:
            sr = bench.score(it, p)
            ctx = Ctx(llm, bench, it, diag_mode="clean")
            st = State(problem=it.prompt, plan=p)
            outs = [validate(st, ctx), simulate(st, ctx), crop_stats(st, ctx)]
            if any("reference" in o for o in outs):
                c_bad.append(f"{iid}: a tool output mentions the reference")
            if not sr.valid:
                continue
            n_legal += 1
            sim = bench.simulate(it, p)
            m_ = re.search(r"profit=(-?[\d.]+)", outs[1])
            s_ = re.search(r"profit=(-?[\d.]+)", sr.diag)
            if not m_ or not s_ or float(m_.group(1)) != float(s_.group(1)) or float(m_.group(1)) != float(sim["profit"]):
                c_bad.append(f"{iid}: simulate profit {m_ and m_.group(1)} vs scorer {s_ and s_.group(1)}")
            ref = sim.get("reference_profit")
            if ref is not None and float(ref) != float(sim["profit"]):
                for o in outs:
                    if re.search(rf"(?<![\d.]){re.escape(str(ref))}(?![\d.])", o):
                        c_bad.append(f"{iid}: reference profit {ref} appears in a tool output")
            rel = sim.get("relative_score")
            if rel is not None and any(f"{rel}" in o for o in outs):
                c_bad.append(f"{iid}: relative score appears in a tool output")
    report["c"] = {"plans": sum(len(v) for v in plans.values()), "legal_plans": n_legal, "mismatches": c_bad}
    report["new_model_calls"] = llm.budget.new_calls  # must be 0
    out = Path(a.out or os.path.join(ROOT, "runs", f"replay_{run_dir.name}"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay_check.json").write_text(json.dumps(report, indent=1))
    print(f"(c) {n_legal} legal plans checked, {len(c_bad)} problems; new model calls: {llm.budget.new_calls}")
    print("report:", out / "replay_check.json")


if __name__ == "__main__":
    main()
