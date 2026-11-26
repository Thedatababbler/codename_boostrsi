"""Shared experiment plumbing: environment construction, running method sets, metrics (§8.2)."""
from __future__ import annotations

import json
import os
import statistics
from collections import Counter

from .evaluate import Env, evaluate
from .llm import CachedLLM, OpenAICompatLLM
from .model import to_str
from .stats import mean_ci, paired_diffs


def make_env(args) -> Env:
    from farm.bench import FarmBench
    if args.llm == "fake":
        from .fake import FakeFarmLLM
        inner = FakeFarmLLM()
    else:
        if not args.base_url or not args.model:
            raise SystemExit("--base-url and --model are required for --llm openai")
        inner = OpenAICompatLLM(args.base_url, args.model, api_key=os.environ.get(args.api_key_env)
                                if args.api_key_env else None)
    llm = CachedLLM(inner, args.cache, max_concurrency=args.concurrency)
    return Env(llm, FarmBench(), {"status_threshold": args.status_threshold,
                                  "status_mode": args.status_mode},
               workers=args.workers)


def add_common_args(ap):
    ap.add_argument("--llm", choices=["fake", "openai"], default="fake")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="served model name, e.g. Qwen/Qwen3-14B")
    ap.add_argument("--api-key-env", default=None, help="name of an env var holding the key (optional)")
    ap.add_argument("--cache", default="runs/cache/llm.jsonl")
    ap.add_argument("--concurrency", type=int, default=4, help="max concurrent model requests")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--difficulty", type=int, default=5)
    ap.add_argument("--status-threshold", type=float, default=0.8)
    ap.add_argument("--status-mode", choices=["score", "bound"], default="score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/out")


def run_methods(methods: dict, items, env: Env, run_seed: int = 0) -> dict:
    return {name: evaluate(wf, items, env, run_seed) for name, wf in methods.items()}


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
        "legal_mean_raw": f(r["raw"] for r in legal) if legal else None,
        "tokens": f(x["tokens"] for x in c), "llm_calls": f(x["llm_calls"] for x in c),
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
