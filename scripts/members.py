"""Structure of every accepted member/stage, and (AdaBoost) which members solve which items.

    PYTHONPATH=src python scripts/members.py --runs-dir ../boostrsi/runs RUN [RUN ...] > runs/MEMBERS.md

"Solved by member m" on an item = m's gated output beats y0 (g > 0) in at least one of the n_repeats samples,
run alone on y0 -- exactly the samples AdaBoost inference draws (eval-cache hits, same run seed). Splits:
heldout and dev_d2.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boosttopo.config import Cfg  # noqa: E402
from boosttopo.graph import Subgraph  # noqa: E402
from boosttopo.infer import SEED_STRIDE  # noqa: E402
from boosttopo.run import build, make_splits  # noqa: E402


def structure(h: Subgraph) -> str:
    rc = {k: v for k, v in h.role_counts().items() if v}
    loops = ", ".join(f"{r} x{k} {s}" for r, k, s in h.loops()) or "none"
    return f"{len(h.nodes)} nodes {rc}, depth {h.depth()}, loops: {loops}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"))
    ap.add_argument("--splits", default="heldout,dev_d2")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    out = ["# Accepted members / stages", ""]
    for run in a.runs:
        d = Path(a.runs_dir) / run
        ens = json.loads((d / "ensemble.json").read_text())
        cfg = Cfg.wrap(json.loads((d / "config.json").read_text())["config"])
        cfg.cache_dir = str(ROOT / cfg.cache_dir) if not Path(cfg.cache_dir).is_absolute() else cfg.cache_dir
        members = [Subgraph.from_dict(x) for x in ens["ensemble"]]
        out += [f"## {run} ({ens['mode']}, seed {cfg.seed})", ""]
        rounds = [json.loads(l) for l in (d / "rounds.jsonl").read_text().splitlines()]
        acc = [r for r in rounds if r["accepted"]]
        if ens.get("extra", {}).get("replaced"):
            out.append("- round-built ensemble was replaced by offline selection; members below are the final ones")
        for i, h in enumerate(members):
            rr = next((r for r in acc if r["chosen"] == h.describe()), None)
            src = f"round {rr['round']}" if rr else "offline selection"
            out.append(f"{i + 1}. `{h.describe()}` -- {structure(h)}; from {src}")
        out.append("")
        if not members:
            continue
        bench, llm, ev, _ = build(cfg, None)
        ev.rep_seed = int(cfg.seed) * SEED_STRIDE
        splits = make_splits(cfg, bench)
        for sname in a.splits.split(","):
            items = splits[sname]
            ev.make_start(items)
            y0 = {it.item_id: ev.y0[it.item_id] for it in items}
            # every member run alone on y0 (for GBDT this is not how members run at inference, where each
            # refines the previous stage; it measures what each structure can solve on its own)
            res = [ev.run(h, items, y0, cfg.eval.n_repeats, occ=ev.occurrence(h, members[:j]))
                   for j, h in enumerate(members)]
            solved = {it.item_id: [any(ev.g(it.item_id, s) > 0 for s in r[it.item_id].scores) for r in res]
                      for it in items}
            n_any = sum(any(v) for v in solved.values())
            n_one = sum(sum(v) == 1 for v in solved.values())
            # between two repeats of the same member: items solved in exactly one of its repeats
            rep_excl = []
            for j in range(len(members)):
                per = [[ev.g(it.item_id, s) > 0 for s in res[j][it.item_id].scores] for it in items]
                rep_excl.append((sum(any(p) for p in per), sum(len(p) > 1 and sum(p) == 1 for p in per)))
            out += [f"**{sname}**: {len(items)} items; solved by any member: {n_any}; by exactly one member: {n_one}; "
                    f"per member: " + ", ".join(f"m{j + 1}={sum(v[j] for v in solved.values())}" for j in range(len(members))),
                    "", "- between repeats of the same member (solved in any repeat / in exactly one of the repeats): "
                    + ", ".join(f"m{j + 1} {a}/{b}" for j, (a, b) in enumerate(rep_excl)),
                    "", "| item | y0 legal | " + " | ".join(f"m{j + 1} g" for j in range(len(members))) + " | solved by |",
                    "|---|---|" + "---|" * len(members) + "---|"]
            for it in items:
                v = solved[it.item_id]
                if not any(v):
                    continue
                y0_legal = bench.score(it, ev.y0[it.item_id]).valid
                out.append(f"| {it.item_id} | {'yes' if y0_legal else 'no'} | " +
                           " | ".join(f"{res[j][it.item_id].g:.2f}" for j in range(len(members))) + " | " +
                           ",".join(f"m{j + 1}" for j in range(len(members)) if v[j]) + " |")
            out += ["", f"(items solved by no member omitted: {len(items) - n_any})", ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
