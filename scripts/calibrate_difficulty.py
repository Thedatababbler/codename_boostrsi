"""Difficulty calibration: the single writer (benchmark prompt, as y0) on N items per difficulty.

    python scripts/calibrate_difficulty.py --config configs/calibrate.yaml [--prompt benchmark|writer]
                                           [--set llm.model=...] [--n 20]

--prompt benchmark: the benchmark's own prompt (= y0 / single writer). --prompt writer: the writer role prompt
(ledger before JSON) on an empty current plan, i.e. the writer role writing from scratch.

The model / endpoint / generation settings come from the config's `llm:` block, so rerunning on another
model is a config change. Seeds come from `calibration.seeds` (default [6000, 6000+n)), which no other
split uses. Writes runs/calibration_<model>.{json,md} and prints the table.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boosttopo.bench.base import make_bench  # noqa: E402
from boosttopo.config import deep_merge, load_config, parse_override  # noqa: E402
from boosttopo.llm import atomic_write_text, build_llm  # noqa: E402
from boosttopo.roles import role_prompt, y0_request  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "calibrate.yaml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--difficulties", default="1,2,3,4,5")
    ap.add_argument("--prompt", choices=("benchmark", "writer"), default="benchmark")
    a = ap.parse_args(argv)
    over = {}
    for s in a.set:
        over = deep_merge(over, parse_override(s))
    cfg = load_config(a.config, over)
    bench = make_bench(cfg)
    llm = build_llm(cfg, ROOT / cfg.cache_dir)
    start = (cfg.get("calibration") or {}).get("seed_start", 6000)
    rows, table = [], []
    for d in [int(x) for x in a.difficulties.split(",")]:
        items = bench.make_items(cfg.bench.game, d, list(range(start, start + a.n)))
        with ThreadPoolExecutor(max_workers=cfg.llm.get("max_concurrency", 32)) as pool:
            req = (lambda it: y0_request(it, 0)) if a.prompt == "benchmark" else \
                (lambda it: role_prompt("writer", it, "", [], None, rep=0))
            outs = list(pool.map(lambda it: llm.call(req(it)), items))
        res = []
        for it, (text, tok) in zip(items, outs):
            sr = bench.score(it, text)
            res.append({"difficulty": d, "item_id": it.item_id, "valid": bool(sr.valid), "score": sr.score,
                        "category": sr.category, "tokens": tok})
        rows += res
        legal = [r["score"] for r in res if r["valid"]]
        cats = {}
        for r in res:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        table.append({"difficulty": d, "n": len(res), "legal_rate": mean(r["valid"] for r in res),
                      "mean_score": mean(r["score"] for r in res),
                      "mean_score_legal": mean(legal) if legal else None,
                      "tokens_per_item": mean(r["tokens"] for r in res), "categories": cats})
        print(json.dumps(table[-1]), flush=True)
    model = cfg.llm.model
    gen = {k: cfg.llm.get(k) for k in ("temperature", "max_tokens", "extra_body")}
    think = bool(((cfg.llm.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking"))
    md = [f"# Difficulty calibration: `{model}`, {a.prompt} prompt, thinking {'on' if think else 'off'}", "",
          f"generation: `{json.dumps(gen)}`; seeds [{start}, {start + a.n}) per difficulty; 1 sample per item.", "",
          "| difficulty | n | legal rate | mean score | mean score among legal | tokens/item | failure categories |",
          "|---|---|---|---|---|---|---|"]
    for t in table:
        msl = f"{t['mean_score_legal']:.3f}" if t["mean_score_legal"] is not None else "–"
        md.append(f"| {t['difficulty']} | {t['n']} | {t['legal_rate']:.2f} | {t['mean_score']:.3f} | {msl} | "
                  f"{t['tokens_per_item']:.0f} | {', '.join(f'{k}:{v}' for k, v in sorted(t['categories'].items()))} |")
    tag = f"{model.replace('/', '_')}_{a.prompt}_{'think' if think else 'nothink'}"
    atomic_write_text(ROOT / "runs" / f"calibration_{tag}.json", json.dumps({"model": model, "gen": gen, "prompt": a.prompt,
                                                                             "table": table, "rows": rows}))
    atomic_write_text(ROOT / "runs" / f"calibration_{tag}.md", "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
