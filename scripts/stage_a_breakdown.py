"""Cache-only breakdown of Stage A (any LLM cache miss raises; no LLM is called).

    PYTHONPATH=src python scripts/stage_a_breakdown.py --config configs/r4_d5.yaml [--out runs/r4_stageA/BREAKDOWN.md]

1. Repair loop (<= 5 repairs, keep best): every repair, split by whether the plan being repaired (the lineage's
   current plan) was illegal or already legal; per group how often the repair's output scored higher than the plan
   it repaired, and by how much (mean score change, over all repairs of the group and over the raising ones).
2. Total gain (mean g) of best-of-5 and of the repair loop, split into items whose y0 was illegal and whose answer
   is legal ("illegal -> legal") and items whose y0 was legal and whose answer scores higher ("legal -> higher").
Replays exactly the Stage A control flow (same requests and repeat indices), and checks the replayed mean g against
runs/r4_stageA/results.json.
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boosttopo.config import load_config  # noqa: E402
from boosttopo.infer import SEED_STRIDE, writer_role  # noqa: E402
from boosttopo.roles import role_prompt  # noqa: E402
from boosttopo.run import build, make_splits  # noqa: E402


class CacheMiss(Exception):
    pass


class NoLLM:
    def __init__(self, inner):
        self.name, self.temperature = inner.name, inner.temperature
        self.max_tokens, self.extra_body = getattr(inner, "max_tokens", None), getattr(inner, "extra_body", None)

    def call(self, req):
        raise CacheMiss(f"cache miss: role={req.role} rep={req.rep}")


def replay_loop(it, ev, max_repairs=5):
    """infer._repair_loop_one with budget inf and r = 0, recording every repair."""
    rep = ev.rep_seed
    y0, s0 = ev.y0[it.item_id], ev.s0[it.item_id]
    text, _ = ev.llm.call(role_prompt("writer", it, y0, [], None, rep=rep))
    cur, cur_sr = text, ev.bench.score(it, text)
    best, best_s = (cur, cur_sr.score) if cur_sr.score > s0 else (y0, s0)
    repairs, n = [], 0
    while n < max_repairs and ev.g(it.item_id, best_s) < 1.0:
        text, _ = ev.llm.call(role_prompt("repairer", it, cur, [], cur_sr.diag, rep=rep + 7919 * (n + 1)))
        n += 1
        sr = ev.bench.score(it, text)
        repairs.append({"item": it.item_id, "j": n, "in_legal": cur_sr.valid, "in_score": cur_sr.score,
                        "out_legal": sr.valid, "out_score": sr.score, "in_cat": cur_sr.category,
                        "best_before": best_s})
        if sr.score >= cur_sr.score:
            cur, cur_sr = text, sr
        if sr.score > best_s:
            best, best_s = text, sr.score
    return best, repairs


def replay_bon(it, ev, n=5):
    best, best_s = ev.y0[it.item_id], ev.s0[it.item_id]
    for r in range(n):
        so = ev._one(writer_role(), it, ev.y0[it.item_id], r)
        if so.score > best_s:
            best, best_s = so.solution, so.score
    return best


def split_gain(items, ev, answers):
    out = {"illegal -> legal": [0, 0.0], "legal -> higher": [0, 0.0], "illegal -> illegal": [0, 0.0],
           "legal, no gain": [0, 0.0]}
    for it in items:
        y0_legal = ev.bench.score(it, ev.y0[it.item_id]).valid
        sr = ev.bench.score(it, answers[it.item_id])
        g = ev.g(it.item_id, sr.score)
        k = ("illegal -> legal" if sr.valid else "illegal -> illegal") if not y0_legal else \
            ("legal -> higher" if g > 0 else "legal, no gain")
        out[k][0] += 1
        out[k][1] += g / len(items)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r4_d5.yaml")
    ap.add_argument("--split", default="test_d5")
    ap.add_argument("--stage-a", default="runs/r4_stageA")
    ap.add_argument("--out", default="runs/r4_stageA/BREAKDOWN.md")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    _, llm, ev, _ = build(cfg, None)
    llm.inner = NoLLM(llm.inner)
    ev.rep_seed = int(cfg.seed) * SEED_STRIDE
    items = make_splits(cfg, ev.bench)[a.split]
    ev.make_start(items)
    loop_ans, reps = {}, []
    for it in items:
        loop_ans[it.item_id], rs = replay_loop(it, ev)
        reps += rs
    bon_ans = {it.item_id: replay_bon(it, ev) for it in items}
    stored = json.loads((ROOT / a.stage_a / "results.json").read_text())
    for m, ans in (("repair_loop5", loop_ans), ("best_of_5", bon_ans)):
        g = mean(ev.g(it.item_id, ev.bench.score(it, ans[it.item_id]).score) for it in items)
        assert abs(g - stored[m]["mean_g"]) < 1e-9, (m, g, stored[m]["mean_g"])

    md = [f"# Stage A breakdown (cache-only replay; {len(items)} items, seed {cfg.seed})", "",
          "## Repair loop: repairs by the legality of the plan being repaired", "",
          "'raised' = the repair's output scores higher than the plan it repaired (illegal plans score 0). "
          "'raised the best' = higher than the best plan so far (y0 included), i.e. it changed the answer.", "",
          "| plan being repaired | repairs | raised | mean Δscore (all) | mean Δscore (raised) | output legal | "
          "raised the best | mean gain over best (all) |", "|---|---|---|---|---|---|---|---|"]
    for name, sel in (("illegal", [r for r in reps if not r["in_legal"]]),
                      ("legal", [r for r in reps if r["in_legal"]]), ("all", reps)):
        if not sel:
            md.append(f"| {name} | 0 | – | – | – | – | – | – |")
            continue
        up = [r for r in sel if r["out_score"] > r["in_score"]]
        ub = [r for r in sel if r["out_score"] > r["best_before"]]
        md.append(f"| {name} | {len(sel)} | {len(up)} ({len(up) / len(sel):.0%}) | "
                  f"{mean(r['out_score'] - r['in_score'] for r in sel):+.3f} | "
                  f"{mean(r['out_score'] - r['in_score'] for r in up) if up else 0:+.3f} | "
                  f"{mean(r['out_legal'] for r in sel):.0%} | {len(ub)} ({len(ub) / len(sel):.0%}) | "
                  f"{mean(max(0.0, r['out_score'] - r['best_before']) for r in sel):+.3f} |")
    md += ["", "By repair index (all items):", "", "| repair # | n | plan being repaired legal | raised | "
           "raised the best |", "|---|---|---|---|---|"]
    for j in range(1, 6):
        sel = [r for r in reps if r["j"] == j]
        if sel:
            md.append(f"| {j} | {len(sel)} | {mean(r['in_legal'] for r in sel):.0%} | "
                      f"{mean(r['out_score'] > r['in_score'] for r in sel):.0%} | "
                      f"{mean(r['out_score'] > r['best_before'] for r in sel):.0%} |")
    md += ["", "## Where the total gain comes from (by y0 legality and the final answer)", "",
           "| method | illegal → legal: items, share of mean g | legal → higher: items, share | "
           "illegal → illegal | legal, no gain | mean g |", "|---|---|---|---|---|---|"]
    for m, ans in (("best_of_5", bon_ans), ("repair_loop5", loop_ans)):
        s = split_gain(items, ev, ans)
        tot = sum(v[1] for v in s.values())
        md.append(f"| {m} | {s['illegal -> legal'][0]}, {s['illegal -> legal'][1]:.3f} | "
                  f"{s['legal -> higher'][0]}, {s['legal -> higher'][1]:.3f} | {s['illegal -> illegal'][0]} | "
                  f"{s['legal, no gain'][0]} | {tot:.3f} |")
    (ROOT / a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
