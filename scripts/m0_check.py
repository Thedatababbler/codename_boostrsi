"""M0: single writer on the 40 Farm training items; a rerun must make no new LLM calls."""
import sys, time, json
from statistics import mean
sys.path.insert(0, "src")
from boosttopo.config import load_config
from boosttopo.run import build, make_splits
from boosttopo.graph import single_writer
from collections import Counter

cfg = load_config("configs/farm_gbdt.yaml")
for attempt in (1, 2):
    bench, llm, ev, log = build(cfg, None)
    sp = make_splits(cfg, bench)
    t = time.time()
    ev.make_start(sp["train"] + sp["heldout"])
    tr = sp["train"]
    res = ev.run(single_writer(), tr, {it.item_id: ev.y0[it.item_id] for it in tr})
    s0 = [ev.s0[it.item_id] for it in tr]
    print(json.dumps({"attempt": attempt, "secs": round(time.time() - t, 1), "budget": llm.budget.snapshot(),
        "y0_mean_score": mean(s0), "y0_valid": mean(bench.score(it, ev.y0[it.item_id]).valid for it in tr),
        "y0_cats": Counter(bench.score(it, ev.y0[it.item_id]).category for it in tr),
        "writer_on_y0_mean_g": mean(r.g for r in res.values()),
        "writer_on_y0_mean_score": mean(r.score for r in res.values())}))
