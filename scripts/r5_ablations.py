"""Round 5 step 8: fixed-structure ablations of the seed-0 ensemble on the test split (no search).

    PYTHONPATH=src python scripts/r5_ablations.py --step 8a|8b|report [--run r4_gbdt_lm_s0]

8a  the seed-0 ensemble with stage 0 (analyst -> writer) replaced by a plain writer call [IN->writer]; stages 1-2
    unchanged; GBDT chaining and early stop exactly as ours (infer). The plain writer call at repeat 0 is the same
    request as writer_once / best-of-N sample 0 (one shared cached sample).
8b  stage 0 of the seed-0 ensemble (its cached gated output), then independent writer samples, each reading stage
    0's gated output (no chaining), keep the best by score. Items already at g = 1 are skipped and sampling of an
    item stops once it reaches g = 1 (ours' rule). Samples use repeat indices 0, 1, ... of the plain writer stage
    (disjoint from ours' loop indices j * 7919). Samples are drawn round by round; another round is drawn only
    while the projected mean tokens/item stays within ours' final tokens/item (rounding N down).
Results: runs/r5_ablations/<step>.json and per_item_<step>.jsonl; report -> runs/r5_ablations/REPORT.md.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boosttopo.config import load_config  # noqa: E402
from boosttopo.graph import Subgraph  # noqa: E402
from boosttopo.infer import SEED_STRIDE, infer, writer_role, y0_tokens  # noqa: E402
from boosttopo.llm import atomic_write_text  # noqa: E402
from boosttopo.run import build, make_splits  # noqa: E402
from report import paired_ci  # noqa: E402

OUT = ROOT / "runs" / "r5_ablations"


def setup(run, split):
    d = ROOT / "runs" / run
    cfg = load_config(None, json.loads((d / "config.json").read_text())["config"])
    ens = [Subgraph.from_dict(x) for x in json.loads((d / "ensemble.json").read_text())["ensemble"]]
    fin = json.loads((d / "final.json").read_text())["splits"][split]
    _, llm, ev, _ = build(cfg, None)
    ev.rep_seed = int(cfg.seed) * SEED_STRIDE
    items = make_splits(cfg, ev.bench)[split]
    ev.make_start(items)
    return cfg, ens, fin, llm, ev, items


def save(step, res, rows, extra):
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_text(OUT / f"{step}.json", json.dumps({**{k: v for k, v in res.items() if k != "rows"}, **extra},
                                                       indent=1))
    atomic_write_text(OUT / f"per_item_{step}.jsonl", "".join(json.dumps(r) + "\n" for r in rows))


def step_8a(cfg, ens, fin, llm, ev, items):
    ens_a = [writer_role()] + ens[1:]
    res = infer("gbdt", ens_a, items, ev, 1)
    return res, res["rows"], {"ensemble": [h.describe() for h in ens_a]}


def step_8b(cfg, ens, fin, llm, ev, items, target):
    st = {}
    for it in items:  # stage 0 of ours (cached), gated
        iid = it.item_id
        y, tok = ev.y0[iid], y0_tokens(ev, it)
        if ev.g(iid, ev.s0[iid]) < 1.0:
            so = ev._one(ens[0], it, y, 0, 0)
            y, tok = so.solution, tok + so.cost
        s = ev.score_of(it, y)
        st[iid] = {"it": it, "y1": y, "tokens": tok, "best": s, "best_y": y, "n": 0, "calls": 2}
    pool = ThreadPoolExecutor(max_workers=cfg.llm.get("max_concurrency", 24))
    r, last_cost = 0, None
    log = []
    while True:
        open_ = [v for v in st.values() if ev.g(v["it"].item_id, v["best"]) < 1.0]
        if not open_:
            break
        cur = mean(v["tokens"] for v in st.values())
        if last_cost is not None and cur + last_cost * len(open_) / len(st) > target:
            break  # the next round would exceed ours' tokens/item (projected): N rounds down
        outs = list(pool.map(lambda v: ev._one(writer_role(), v["it"], v["y1"], r), open_))
        costs = []
        for v, so in zip(open_, outs):
            v["tokens"] += so.cost
            v["calls"] += so.scorer_calls
            v["n"] += 1
            costs.append(so.cost)
            if so.score > v["best"]:
                v["best"], v["best_y"] = so.score, so.solution
        last_cost = mean(costs)
        r += 1
        log.append({"round": r, "n_open": len(open_), "mean_tokens": mean(v["tokens"] for v in st.values()),
                    "mean_g": mean(ev.g(i, v["best"]) for i, v in st.items())})
        print(f"8b round {r}: open {len(open_)} mean tokens {log[-1]['mean_tokens']:.0f} g {log[-1]['mean_g']:.3f}",
              flush=True)
    pool.shutdown()
    rows = []
    for iid, v in st.items():
        sr = ev.bench.score(v["it"], v["best_y"])
        rows.append({"item_id": iid, "rep": 0, "g": ev.g(iid, sr.score), "score": sr.score, "tokens": v["tokens"],
                     "scorer_calls": v["calls"], "valid": float(sr.valid), "category": sr.category,
                     "n_samples": v["n"]})
    legal = [x["score"] for x in rows if x["valid"]]
    res = {"n": len(rows), "mean_g": mean(x["g"] for x in rows), "valid_rate": mean(x["valid"] for x in rows),
           "legal_mean_score": mean(legal) if legal else None,
           "tokens_per_item": mean(x["tokens"] for x in rows),
           "scorer_calls_per_item": mean(x["scorer_calls"] for x in rows), "N_rounds": r}
    return res, rows, {"rounds": log, "target_tokens": target}


def report(run, split):
    d = ROOT / "runs" / run
    rows = {}
    for l in (d / "per_item.jsonl").read_text().splitlines():
        x = json.loads(l)
        if x["split"] == split:
            rows.setdefault(x["method"], {})[x["item_id"]] = x
    for step in ("8a", "8b"):
        p = OUT / f"per_item_{step}.jsonl"
        if p.exists():
            rows[step] = {json.loads(l)["item_id"]: json.loads(l) for l in p.read_text().splitlines()}
    ours = rows["ours"]
    ids = sorted(ours)
    md = [f"# Round 5 step 8: ablations of {run} on {split} ({len(ids)} items, 1 repeat)", "",
          "| method | mean g | Δg (ours − row) [90% CI] | ours wins/ties/losses | legal rate | mean score among legal | "
          "tokens/item |", "|---|---|---|---|---|---|---|"]
    names = {"ours": "ours (seed-0 ensemble)", "8a": "8a: stage 0 → plain writer", "8b": "8b: stage 0 + independent "
             "writer samples", "best_of_n_writer": "best-of-11 writer", "repair_loop": "repair loop (matched)"}
    for m in ("ours", "8a", "8b", "best_of_n_writer", "repair_loop"):
        if m not in rows:
            continue
        v = rows[m]
        dd = [ours[i]["g"] - v[i]["g"] for i in ids]
        lo, hi = paired_ci(dd)
        wtl = f"{sum(x > 1e-9 for x in dd)}/{sum(abs(x) <= 1e-9 for x in dd)}/{sum(x < -1e-9 for x in dd)}"
        legal = [v[i]["score"] for i in ids if v[i]["valid"]]
        md.append(f"| {names[m]} | {mean(v[i]['g'] for i in ids):.3f} | "
                  + ("–" if m == "ours" else f"{mean(dd):+.3f} [{lo:+.3f}, {hi:+.3f}]")
                  + f" | {'–' if m == 'ours' else wtl} | {mean(v[i]['valid'] for i in ids):.2f} | "
                  f"{mean(legal) if legal else float('nan'):.3f} | {mean(v[i]['tokens'] for i in ids) / 1000:.1f}k |")
    if "8b" in rows:
        ns = [rows["8b"][i].get("n_samples", 0) for i in ids]
        md += ["", f"8b samples per item: mean {mean(ns):.2f}, max {max(ns)} (items at g = 1 stop early)."]
    atomic_write_text(OUT / "REPORT.md", "\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["8a", "8b", "report"])
    ap.add_argument("--run", default="r4_gbdt_lm_s0")
    ap.add_argument("--split", default="test_d5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = ROOT / a.out
    if a.step == "report":
        return report(a.run, a.split)
    cfg, ens, fin, llm, ev, items = setup(a.run, a.split)
    t0, b0 = time.time(), llm.budget.snapshot()
    if a.step == "8a":
        res, rows, extra = step_8a(cfg, ens, fin, llm, ev, items)
    else:
        res, rows, extra = step_8b(cfg, ens, fin, llm, ev, items, fin["ours"]["tokens_per_item"])
    b1 = llm.budget.snapshot()
    extra.update({"wall_s": time.time() - t0, "new_calls": b1["new_calls"] - b0["new_calls"],
                  "new_tokens": b1["new_tokens"] - b0["new_tokens"]})
    save(a.step, res, rows, extra)
    print(f"{a.step}: g={res['mean_g']:.3f} tokens/item={res['tokens_per_item']:.0f} new_calls={extra['new_calls']} "
          f"wall={extra['wall_s'] / 3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
