"""Setup calibration (round 4, Part 1): per difficulty, three single-call measurements on the same items.

    python scripts/calibrate_v2.py --config configs/calibrate_thinking_long.yaml --difficulties 2,3,4,5 --n 40

Per item (sample index 0 throughout):
  y0            the benchmark prompt (= the single writer / start solution of a run in this setup)
  writer_empty  the writer role prompt on an empty current plan (writer from scratch)
  writer_on_y0  the writer role prompt run once on that item's y0 (= the 'writer once' baseline)
Per (difficulty, measurement) cell it reports
  * legal rate; share truncated (finish_reason == "length"); share unparseable (parse / schema / empty)
  * the unclipped score (simulator relative_score = profit / reference profit) among legal plans:
    mean, quantiles, share >= 0.95 (i.e. no room to improve)
  * the share of each failure type among illegal plans
  * tokens per item (prompt + completion; completion includes the thinking tokens) and seconds per item
    (per-call latency, and wall clock of the cell divided by n; calls run concurrently)
Seeds come from calibration.seed_start (default 6000), which no other split uses.
Writes runs/calibration_v2_<model>_<think|nothink>.{json,md}; the json is rewritten after every cell.
"""
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean, median, quantiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boosttopo.bench.base import make_bench  # noqa: E402
from boosttopo.config import deep_merge, load_config, parse_override  # noqa: E402
from boosttopo.llm import atomic_write_text, build_llm  # noqa: E402
from boosttopo.roles import role_prompt, y0_request  # noqa: E402

UNPARSEABLE = {"parse_error", "schema_error", "empty"}
MEASURES = ("y0", "writer_empty", "writer_on_y0")


def unclipped(bench, item, text):
    try:
        return float(bench.simulate(item, text)["relative_score"] or 0.0)
    except Exception:
        return None


def cell_stats(res):
    n = len(res)
    legal = [r for r in res if r["valid"]]
    illegal = [r for r in res if not r["valid"]]
    sc = sorted(r["raw_score"] for r in legal if r["raw_score"] is not None)
    q = quantiles(sc, n=4) if len(sc) >= 2 else (sc * 3 if sc else [])
    fails = Counter(r["category"] for r in illegal)
    lat = [r["seconds"] for r in res if r["seconds"] is not None]
    return {
        "n": n, "legal_rate": len(legal) / n,
        "truncated": mean(r["finish_reason"] == "length" for r in res),
        "truncated_known": sum(r["finish_reason"] is not None for r in res),
        "unparseable": mean(r["category"] in UNPARSEABLE for r in res),
        "legal_score": {"n": len(sc), "mean": mean(sc) if sc else None, "min": sc[0] if sc else None,
                        "q1": q[0] if sc else None, "median": median(sc) if sc else None,
                        "q3": q[2] if sc else None, "max": sc[-1] if sc else None,
                        "share_ge_0.95": mean(s >= 0.95 for s in sc) if sc else None, "values": sc},
        "fail_share": {k: v / len(illegal) for k, v in fails.most_common()} if illegal else {},
        "tokens_per_item": mean(r["tokens"] for r in res),
        "completion_tokens_per_item": mean(r["completion_tokens"] for r in res if r["completion_tokens"] is not None)
        if any(r["completion_tokens"] is not None for r in res) else None,
        "latency_s_median": median(lat) if lat else None,
    }


def qualifies(st):
    """The three criteria of the round-4 prompt; returns (ok, reasons)."""
    reasons = []
    if not 0.30 <= st["legal_rate"] <= 0.70:
        reasons.append("legal rate outside 30-70%")
    ls = st["legal_score"]
    if ls["n"] == 0 or ls["share_ge_0.95"] is None or ls["share_ge_0.95"] > 0.5:
        reasons.append("legal plans near the reference (>50% score >= 0.95)")
    if sum(v >= 0.15 for v in st["fail_share"].values()) < 2:
        reasons.append("fewer than two failure types >= 15%")
    return not reasons, reasons


def fmt(x, p=2):
    return "–" if x is None else f"{x:.{p}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--difficulties", default="2,3,4,5")
    a = ap.parse_args(argv)
    over = {}
    for s in a.set:
        over = deep_merge(over, parse_override(s))
    cfg = load_config(a.config, over)
    bench = make_bench(cfg)
    llm = build_llm(cfg, ROOT / cfg.cache_dir)
    start = (cfg.get("calibration") or {}).get("seed_start", 6000)
    think = bool(((cfg.llm.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking"))
    model = cfg.llm.model
    tag = f"{model.replace('/', '_')}_{'think' if think else 'nothink'}"
    gen = {k: cfg.llm.get(k) for k in ("temperature", "max_tokens", "extra_body")}
    out = {"model": model, "gen": gen, "seeds": [start, start + a.n], "cells": {}, "rows": []}
    pool = ThreadPoolExecutor(max_workers=cfg.llm.get("max_concurrency", 16))

    def measure(items, reqs, d, m):
        t0 = time.time()
        outs = list(pool.map(llm.call_meta, reqs))
        wall = time.time() - t0
        res = []
        for it, (text, tok, meta) in zip(items, outs):
            sr = bench.score(it, text)
            res.append({"difficulty": d, "measure": m, "item_id": it.item_id, "valid": bool(sr.valid),
                        "score": sr.score, "raw_score": unclipped(bench, it, text) if sr.valid else None,
                        "category": sr.category, "tokens": tok, "completion_tokens": meta.get("completion_tokens"),
                        "finish_reason": meta.get("finish_reason"), "seconds": meta.get("seconds")})
        st = cell_stats(res)
        st["wall_s_per_item"] = wall / len(items)
        st["qualifies"], st["why_not"] = qualifies(st)
        return res, st, [o[0] for o in outs]

    for d in [int(x) for x in a.difficulties.split(",")]:
        items = bench.make_items(cfg.bench.game, d, list(range(start, start + a.n)))
        res, st, y0s = measure(items, [y0_request(it, 0) for it in items], d, "y0")
        out["rows"] += res
        out["cells"][f"d{d}/y0"] = st
        cells = [("writer_empty", [role_prompt("writer", it, "", [], None, rep=0) for it in items]),
                 ("writer_on_y0", [role_prompt("writer", it, y, [], None, rep=0) for it, y in zip(items, y0s)])]
        for m, reqs in cells:
            res, st, _ = measure(items, reqs, d, m)
            out["rows"] += res
            out["cells"][f"d{d}/{m}"] = st
        for m in MEASURES:
            print(f"d{d} {m}: " + json.dumps({k: v for k, v in out["cells"][f'd{d}/{m}'].items()
                                               if k not in ("legal_score",)}), flush=True)
        atomic_write_text(ROOT / "runs" / f"calibration_v2_{tag}.json", json.dumps(out))
    pool.shutdown()
    md = [f"# Setup calibration: `{model}`, thinking {'on' if think else 'off'}", "",
          f"generation: `{json.dumps(gen)}`; seeds [{start}, {start + a.n}) per difficulty; 1 sample per item and "
          "measurement. writer_on_y0 = the writer role run once on that item's y0.", "",
          "| cell | legal | truncated | unparseable | legal score: mean / min / q1 / median / q3 / max | "
          "share ≥0.95 | failure types among illegal | tokens/item | latency s (median) | wall s/item | "
          "qualifies |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, st in out["cells"].items():
        ls = st["legal_score"]
        dist = " / ".join(fmt(ls[x]) for x in ("mean", "min", "q1", "median", "q3", "max"))
        fails = ", ".join(f"{c} {v:.0%}" for c, v in st["fail_share"].items())
        q = "yes" if st["qualifies"] else "no: " + "; ".join(st["why_not"])
        md.append(f"| {k} | {st['legal_rate']:.2f} | {st['truncated']:.2f} | {st['unparseable']:.2f} | {dist} (n={ls['n']}) | "
                  f"{fmt(ls['share_ge_0.95'])} | {fails or '–'} | {st['tokens_per_item']:.0f} | "
                  f"{fmt(st['latency_s_median'], 0)} | {st['wall_s_per_item']:.1f} | {q} |")
    atomic_write_text(ROOT / "runs" / f"calibration_v2_{tag}.md", "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
