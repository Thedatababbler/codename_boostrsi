import json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from attempts import NoLLM, ROOT
from boosttopo.config import load_config
from boosttopo.graph import Subgraph
from boosttopo.infer import SEED_STRIDE
from boosttopo.run import build, make_splits
for run in sys.argv[1:]:
    d = ROOT / "runs" / run
    stored = json.loads((d / "config.json").read_text())["config"]
    stored.setdefault("fresh_duplicate_stages", False)  # runs before round 4 had no duplicate-stage fix
    cfg = load_config(None, stored)
    _, llm, ev, _ = build(cfg, None); llm.inner = NoLLM(); ev.rep_seed = int(cfg.seed) * SEED_STRIDE
    sp = make_splits(cfg, ev.bench)
    ens = [Subgraph.from_dict(x) for x in json.loads((d / "ensemble.json").read_text())["ensemble"]]
    for s in ("heldout", "dev_d2"):
        ev.make_start(sp[s]); ran = dup = 0
        for it in sp[s]:
            for r in range(cfg.eval.n_repeats):
                y = ev.y0[it.item_id]; seen = set()
                for h in ens:
                    if ev.g(it.item_id, ev.score_of(it, y)) >= 1.0: break
                    k = (h.key(), y); ran += 1
                    if k in seen: dup += 1
                    seen.add(k); y = ev._one(h, it, y, r).solution
        print(run, s, f"stage runs {ran}, exact repeats of an earlier stage (same subgraph, same input, same rep) {dup}")
