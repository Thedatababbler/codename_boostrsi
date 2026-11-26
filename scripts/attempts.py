"""Cache-only check: repair attempts per item, for the repair loop and for the GBDT ensembles.

    PYTHONPATH=src python scripts/attempts.py [--splits heldout,dev_d2] r2_farm_gbdt_s0 ...

Replays inference from the LLM / stage caches; any cache miss raises (no LLM is ever called).
Per (item, repeat):
  * repair_loop: number of repairer calls made (n_repairs) and the attempt at which the lineage first held
    a legal plan (0 = the writer's plan was already legal; None = never)
  * ours (gbdt): repairer node executions over the stages that ran, and the stage / cumulative repairer count
    at which the running plan first became legal
Split by whether the final plan is legal.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

from boosttopo.config import load_config
from boosttopo.graph import Subgraph
from boosttopo.infer import SEED_STRIDE, y0_tokens
from boosttopo.roles import role_prompt
from boosttopo.run import build, make_splits

ROOT = Path(__file__).resolve().parents[1]


class CacheMiss(Exception):
    pass


class NoLLM:
    name = "none"

    def call(self, req):
        raise CacheMiss(f"cache miss: role={req.role} rep={req.rep}")


def replay_repair_loop(it, ev, r, budget, max_repairs=40):
    """Same control flow as infer._repair_loop_one, plus the attempt index of the first legal plan."""
    rep = r + ev.rep_seed
    y0 = ev.y0[it.item_id]
    tokens = y0_tokens(ev, it)
    s0 = ev.score_of(it, y0)
    text, t = ev.llm.call(role_prompt("writer", it, y0, [], None, rep=rep))
    tokens += t
    cur, cur_sr = text, ev.bench.score(it, text)
    best_s, best_valid = (cur_sr.score, cur_sr.valid) if cur_sr.score > s0 else (s0, ev.bench.score(it, y0).valid)
    first_legal = 0 if cur_sr.valid else None
    n_rep = 0
    while tokens < budget and n_rep < max_repairs and ev.g(it.item_id, best_s) < 1.0:
        text, t = ev.llm.call(role_prompt("repairer", it, cur, [], cur_sr.diag, rep=rep + 7919 * (n_rep + 1)))
        tokens, n_rep = tokens + t, n_rep + 1
        sr = ev.bench.score(it, text)
        if sr.valid and first_legal is None:
            first_legal = n_rep
        if sr.score >= cur_sr.score:
            cur, cur_sr = text, sr
        if sr.score > best_s:
            best_s, best_valid = sr.score, sr.valid
    return {"attempts": n_rep, "first_legal": first_legal, "legal": bool(best_valid), "tokens": tokens}


def replay_gbdt(ensemble, it, ev, r):
    y = ev.y0[it.item_id]
    reps, first_legal_stage, first_legal_reps = 0, None, None
    if ev.bench.score(it, y).valid:
        first_legal_stage, first_legal_reps = 0, 0
    for k, h in enumerate(ensemble, 1):
        if ev.g(it.item_id, ev.score_of(it, y)) >= 1.0:
            break
        so = ev._one(h, it, y, r)
        reps += h.role_counts().get("repairer", 0)
        y = so.solution
        if first_legal_stage is None and ev.bench.score(it, y).valid:
            first_legal_stage, first_legal_reps = k, reps
    return {"attempts": reps, "first_legal": first_legal_reps, "first_legal_stage": first_legal_stage,
            "legal": bool(ev.bench.score(it, y).valid)}


def dist(xs):
    if not xs:
        return "n=0"
    c = Counter(xs)
    return (f"n={len(xs)}, mean {mean(xs):.2f}, median {median(xs):g}, range {min(xs)}–{max(xs)}; "
            + " ".join(f"{k}:{c[k]}" for k in sorted(c)))


def summarize(rows, label):
    out = [f"**{label}**", ""]
    for name, sel in (("final legal", [r for r in rows if r["legal"]]),
                      ("final illegal", [r for r in rows if not r["legal"]])):
        out.append(f"- {name}: attempts {dist([r['attempts'] for r in sel])}")
        if name == "final legal":
            fl = [r["first_legal"] for r in sel if r["first_legal"] is not None]
            out.append(f"  - attempt at which a legal plan first appeared: {dist(fl)}")
    out.append("")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="heldout,dev_d2")
    ap.add_argument("--out", default="ATTEMPTS.md")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    lines = ["# Repair attempts per item (cache-only replay)", "",
             "Unit: one (item, repeat). 'attempts' = repairer calls. Repair loop: attempt 0 = the writer's first plan. "
             "Ours: repairer node executions summed over the stages that ran (a stage is skipped once g = 1).", ""]
    all_rows = {}
    for run in a.runs:
        d = ROOT / "runs" / run
        stored = json.loads((d / "config.json").read_text())["config"]
        stored.setdefault("fresh_duplicate_stages", False)  # runs before round 4 had no duplicate-stage fix
        cfg = load_config(None, stored)
        _, llm, ev, _ = build(cfg, None)
        llm.inner = NoLLM()
        ev.rep_seed = int(cfg.seed) * SEED_STRIDE
        splits = make_splits(cfg, ev.bench)
        ens = [Subgraph.from_dict(x) for x in json.loads((d / "ensemble.json").read_text())["ensemble"]]
        fin = json.loads((d / "final.json").read_text())
        n_rep = cfg.eval.n_repeats
        lines.append(f"## {run}")
        for s in a.splits.split(","):
            items = splits[s]
            ev.make_start(items)
            budget = fin["splits"][s]["repair_loop"]["token_budget"]
            rl = [replay_repair_loop(it, ev, r, budget) for it in items for r in range(n_rep)]
            ou = [replay_gbdt(ens, it, ev, r) for it in items for r in range(n_rep)]
            # consistency with the stored results
            st = fin["splits"][s]
            assert abs(mean(r["legal"] for r in rl) - st["repair_loop"]["valid_rate"]) < 1e-9, (run, s, "repair_loop")
            assert abs(mean(r["legal"] for r in ou) - st["ours"]["valid_rate"]) < 1e-9, (run, s, "ours")
            lines += summarize(rl, f"{s}: repair loop (budget {budget / 1000:.1f}k tokens)")
            lines += summarize(ou, f"{s}: ours (gbdt)")
            all_rows.setdefault(s, {"rl": [], "ou": []})
            all_rows[s]["rl"] += rl
            all_rows[s]["ou"] += ou
    lines.append("## Pooled over runs")
    for s, v in all_rows.items():
        lines += summarize(v["rl"], f"{s}: repair loop")
        lines += summarize(v["ou"], f"{s}: ours (gbdt)")
    (ROOT / "runs" / a.out).write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
