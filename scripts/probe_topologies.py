"""Probe a few fixed topologies on the first N training items (gated, on y0)."""
import sys, time
from statistics import mean
sys.path.insert(0, "src")
from boosttopo.config import load_config
from boosttopo.run import build, make_splits
from boosttopo.graph import single_writer, apply_edit, expand_macro, Limits, Edit, IN, Subgraph, Node
from boosttopo.infer import hand_topology

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cfg = load_config(sys.argv[2] if len(sys.argv) > 2 else "configs/farm_gbdt.yaml")
bench, llm, ev, log = build(cfg, None)
sp = make_splits(cfg, bench)
tr = sp["train"][:N]
ev.make_start(tr)
L = Limits()
tops = {
    "writer": single_writer(),
    "repairer_only": Subgraph((Node("n0", "repairer"),), ((IN, "n0"),), "n0"),
    "writer->repairer": apply_edit(single_writer(), expand_macro("repair_tail", single_writer(), L)[0], L),
    "analyst->writer": apply_edit(single_writer(), expand_macro("analyze_first", single_writer(), L)[0], L),
    "hand(analyst->writer->repairer)": hand_topology(),
}
for name, h in tops.items():
    t = time.time()
    res = ev.run(h, tr, {it.item_id: ev.y0[it.item_id] for it in tr})
    print(f"{name:34s} g={mean(r.g for r in res.values()):.3f} raw={mean(r.raw_score for r in res.values()):.3f} "
          f"gated={mean(r.score for r in res.values()):.3f} tok={ev.mean_cost(res):.0f} {time.time()-t:.0f}s", flush=True)
