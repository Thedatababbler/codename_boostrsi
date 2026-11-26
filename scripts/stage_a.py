"""Round 4 Stage A: baselines only on the test split (the 40 calibrated d5 items).

    PYTHONPATH=src python scripts/stage_a.py --config configs/r4_d5.yaml [--out runs/r4_stageA]

Methods (one repeat per item, run seed cfg.seed):
  writer_once   writer role once on y0
  best_of_5     writer role 5 times on y0, the scorer keeps the best (of y0 and the 5)
  repair_loop5  writer on y0, then up to 5 repairs of its lineage, keep the best (stops early at g = 1)
  hand_topology analyst -> writer -> repairer (fixed cost; actual tokens reported)
y0 must come from the reused cache records (llm.y0_cache_tags); the script aborts if any y0 would be generated.
Each method's rows are written as soon as it finishes (per_item.jsonl, results.json), so a stop leaves the
completed methods. Gate: repair_loop5 mean g >= 0.75 -> GATE_STOP.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boosttopo.config import deep_merge, load_config, parse_override  # noqa: E402
from boosttopo.infer import SEED_STRIDE, best_of_n_subgraph, hand_topology, infer, repair_loop, writer_role  # noqa: E402
from boosttopo.llm import atomic_write_text  # noqa: E402
from boosttopo.run import build, make_splits  # noqa: E402
from report import paired_ci  # noqa: E402

METHODS = ("writer_once", "repair_loop5", "best_of_5", "hand_topology")
GATE = 0.75


def run_method(m, items, ev):
    if m == "writer_once":
        return infer("adaboost", [writer_role()], items, ev, 1)
    if m == "best_of_5":
        return best_of_n_subgraph(writer_role(), items, ev, 5)
    if m == "repair_loop5":
        return repair_loop(items, ev, math.inf, 1, max_repairs=5)
    if m == "hand_topology":
        return infer("adaboost", [hand_topology()], items, ev, 1)
    raise ValueError(m)


def table(res, base="repair_loop5"):
    """Markdown table: g, legal rate, mean score among legal plans, tokens and scorer calls per item, and the
    paired 90% interval of (method - base) in per-item g."""
    lines = ["| method | mean g | Δg vs repair loop [90% CI] | legal rate | mean score among legal | "
             "tokens/item | scorer calls/item | n |", "|---|---|---|---|---|---|---|---|"]
    b = {r["item_id"]: r["g"] for r in res[base]["rows"]} if base in res else None
    for m, v in res.items():
        if b is not None and m != base:
            g = {r["item_id"]: r["g"] for r in v["rows"]}
            d = [g[i] - b[i] for i in sorted(b) if i in g]
            lo, hi = paired_ci(d)
            dd = f"{mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]"
        else:
            dd = "–"
        ls = v["legal_mean_score"]
        lines.append(f"| {m} | {v['mean_g']:.3f} | {dd} | {v['valid_rate']:.2f} | "
                     f"{'–' if ls is None else f'{ls:.3f}'} | {v['tokens_per_item'] / 1000:.1f}k | "
                     f"{v['scorer_calls_per_item']:.1f} | {v['n']} |")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--split", default="test_d5")
    ap.add_argument("--out", default="runs/r4_stageA")
    a = ap.parse_args(argv)
    over = {}
    for s in a.set:
        over = deep_merge(over, parse_override(s))
    cfg = load_config(a.config, over)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    _, llm, ev, _ = build(cfg, None)
    ev.rep_seed = int(cfg.seed) * SEED_STRIDE
    items = make_splits(cfg, ev.bench)[a.split]
    ev.make_start(items)
    if llm.budget.new_calls:
        raise SystemExit(f"{llm.budget.new_calls} y0 plans were generated instead of reused; check y0_cache_tags")
    print(f"y0 reused for {len(items)} items; llm name {llm.name}; y0 names {llm.y0_names}", flush=True)
    atomic_write_text(out / "config.json", json.dumps({"config": cfg.to_dict(), "split": a.split,
                                                       "items": [it.item_id for it in items]}))
    prev = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    res, rows_all = {}, []
    for m in METHODS:
        t0, b0 = time.time(), llm.budget.snapshot()
        v = run_method(m, items, ev)
        b1 = llm.budget.snapshot()
        v["wall_s"] = time.time() - t0
        v["new_calls"] = b1["new_calls"] - b0["new_calls"]
        v["new_tokens"] = b1["new_tokens"] - b0["new_tokens"]
        res[m] = v
        rows_all += [{"method": m, "s0": ev.s0[r["item_id"]], **r} for r in v["rows"]]
        atomic_write_text(out / "per_item.jsonl", "".join(json.dumps(r) + "\n" for r in rows_all))
        summ = {k: {kk: vv for kk, vv in x.items() if kk != "rows"} for k, x in res.items()}
        atomic_write_text(out / "results.json", json.dumps({**prev, **summ}, indent=1))
        print(f"{m}: g={v['mean_g']:.3f} legal={v['valid_rate']:.2f} tokens/item={v['tokens_per_item']:.0f} "
              f"new_calls={v['new_calls']} wall={v['wall_s'] / 3600:.2f}h "
              f"calls/h={v['new_calls'] / max(1e-9, v['wall_s'] / 3600):.1f}", flush=True)
    md = [f"# Round 4 Stage A: baselines on {a.split} ({len(items)} items, 1 repeat, seed {cfg.seed})", ""]
    md += table(res)
    g_rl = res["repair_loop5"]["mean_g"]
    md += ["", f"repair loop mean g = {g_rl:.3f}; gate {'STOP (>= 0.75)' if g_rl >= GATE else 'continue (< 0.75)'}"]
    atomic_write_text(out / "REPORT.md", "\n".join(md) + "\n")
    print("\n".join(md), flush=True)
    print("GATE_STOP" if g_rl >= GATE else "GATE_CONTINUE", flush=True)


if __name__ == "__main__":
    main()
