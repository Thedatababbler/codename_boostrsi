"""Shared experiment plumbing: environment construction, running method sets, metrics (§8.2)."""
from __future__ import annotations

import dataclasses
import json
import os
import statistics
from collections import Counter

from .evaluate import Env, evaluate
from .model import to_str
from .stats import mean_ci, paired_diffs


WORKFLOW_DEFAULTS = {
    "workflow": {"seed_scheme": "path", "diag_mode": "clean", "start": "y0", "max_calls": 12,
                 "require_cached_y0": ["test_d5"]},
    "boost": {"rounds": 4, "patience": 2, "max_steps": 4, "top_k": 3, "fresh_after_reject": True},
}


def load_cfg(path: str | None, sets: list[str] = ()):
    """configs/*.yaml + --set a.b=c overrides, over boosttopo's defaults and the workflow defaults."""
    from boosttopo.config import DEFAULTS, deep_merge, load_config, parse_override
    import yaml
    over: dict = {}
    for s_ in sets:
        over = deep_merge(over, parse_override(s_))
    raw = yaml.safe_load(open(path, encoding="utf-8")) if path else {}
    return load_config(None, deep_merge(deep_merge(WORKFLOW_DEFAULTS, raw or {}), over))


def make_bench(cfg):
    if cfg.bench.game == "farm-toy":
        from farm.adapter import ToyFarmAdapter
        return ToyFarmAdapter()
    from boosttopo.bench.base import make_bench as mb
    return mb(cfg)


def make_env(cfg, cache_only: bool = False, bench=None) -> Env:
    from .llm import make_llm
    w = cfg.workflow
    return Env(make_llm(cfg, cache_only), bench or make_bench(cfg), seed_scheme=w.seed_scheme,
               diag_mode=w.diag_mode, start=w.start, workers=int(cfg.llm.get("max_concurrency", 8)))


def make_splits(cfg, bench) -> dict:
    """boosttopo.run.make_splits: train, heldout, dev tests; never the final-test splits."""
    from boosttopo.run import make_splits as ms
    return ms(cfg, bench, final_test=False)


def add_common_args(ap):
    ap.add_argument("--config", required=True, help="configs/*.yaml (boosttopo format + workflow / boost blocks)")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set llm.max_concurrency=24")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="run directory (default runs/<config stem>)")


def with_scheme(env: Env, scheme: str) -> Env:
    """The same environment (shared LLM, y0, locks) with another seed scheme."""
    return dataclasses.replace(env, seed_scheme=scheme)


def run_methods(methods: dict, items, env: Env, run_seed: int = 0, scheme_of=None) -> dict:
    env.make_start([it for it in items if it.item_id not in env.y0])
    out = {}
    for name, wf in methods.items():
        e = with_scheme(env, scheme_of(name)) if scheme_of else env
        out[name] = evaluate(wf, items, e, run_seed)
    return out


def dump(path: str, methods: dict, results: dict, extra: dict | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {"methods": {n: to_str(w) for n, w in methods.items()},
            "results": {n: [r.summary() for r in rs] for n, rs in results.items()},
            **(extra or {})}
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=str)


# ------------------------------------------------------------------ metrics (§8.2)
# Work on the JSON summaries so scripts/report.py can recompute from saved files.


def method_metrics(rows: list[dict]) -> dict:
    legal = [r for r in rows if r["legal"]]
    c = [r["cost"] for r in rows]
    branches, loops = Counter(), Counter()
    for r in rows:
        for t in r["trace"]:
            if len(t) > 1 and t[1] == "branch":
                branches[f"{t[3]}{'' if t[4] else ' (pass)'}"] += 1
            if len(t) > 1 and t[1] == "loop":
                loops[t[2]] += 1
    f = statistics.fmean
    return {
        "n": len(rows), "mean_g": f(r["g"] for r in rows), "mean_score": f(r["score"] for r in rows),
        "legal_rate": len(legal) / max(1, len(rows)),
        "legal_mean_score": f(r["score"] for r in legal) if legal else None,
        "tokens": f(x["tokens"] for x in c), "llm_calls": f(x["llm_calls"] for x in c),
        "new_llm_calls": sum(x["new_llm_calls"] for x in c),
        "scorer_calls": f(x["scorer_calls"] for x in c), "tool_calls": f(x["tool_calls"] for x in c),
        "branch_usage": dict(branches), "loop_iterations": dict(sorted(loops.items())),
    }


def compare(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Paired g difference a - b with 90% CI, and where the gain comes from."""
    ga = {r["item"]: r["g"] for r in rows_a}
    gb = {r["item"]: r["g"] for r in rows_b}
    m, lo, hi = mean_ci(paired_diffs(ga, gb))
    A = {r["item"]: r for r in rows_a}
    B = {r["item"]: r for r in rows_b}
    fixed = up = 0.0
    for k in sorted(set(A) & set(B)):
        d = A[k]["g"] - B[k]["g"]
        if not B[k]["legal"] and A[k]["legal"]:
            fixed += d
        elif B[k]["legal"] and A[k]["legal"]:
            up += d
    n = max(1, len(set(A) & set(B)))
    return {"diff": m, "lo": lo, "hi": hi, "from_illegal_to_legal": fixed / n,
            "from_legal_score_up": up / n}
