"""Cache-only checks on Stage B seed 0 (round 5 Part 1). No LLM is called: any LLM cache miss raises.

    PYTHONPATH=src python scripts/r5_checks.py [--run r4_gbdt_lm_s0] [--out runs/r4_gbdt_lm_s0/CHECKS.md]

Stage outputs are recomputed from the LLM cache (the stage cache is redirected to an empty directory), so every LLM
request each method makes is seen, and the replay is checked against the stored per-item rows.
  1. cumulative mean g and tokens per item after each accepted stage; best-of-N writer for N = 1..11 from the cached
     samples (reps 0..N-1); the repair loop after each repair (matched budget) -> table + SVG curves over tokens
  2. per-item wins / ties / losses of ours against best-of-11 and the matched repair loop
  5. duplicate stage runs; LLM request keys shared between methods (fresh samples) and repeated inside ours
  6. early stops at g = 1 (score >= reference): how often, calls not made, tokens saved (estimated)
"""
import argparse
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from boosttopo.config import load_config  # noqa: E402
from boosttopo.graph import Subgraph  # noqa: E402
from boosttopo.infer import SEED_STRIDE, writer_role  # noqa: E402
from boosttopo.roles import role_prompt, y0_request  # noqa: E402
from boosttopo.run import build, make_splits  # noqa: E402


class CacheMiss(Exception):
    pass


class NoLLM:
    def __init__(self, inner):
        self.name, self.temperature = inner.name, inner.temperature
        self.max_tokens, self.extra_body = getattr(inner, "max_tokens", None), getattr(inner, "extra_body", None)

    def call(self, req):
        raise CacheMiss(f"cache miss: role={req.role} rep={req.rep}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r4_gbdt_lm_s0")
    ap.add_argument("--split", default="test_d5")
    ap.add_argument("--n-max", type=int, default=11)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = ROOT / "runs" / a.run
    cfg = load_config(None, json.loads((d / "config.json").read_text())["config"])
    ens = [Subgraph.from_dict(x) for x in json.loads((d / "ensemble.json").read_text())["ensemble"]]
    fin = json.loads((d / "final.json").read_text())["splits"][a.split]
    stored = {}
    for l in (d / "per_item.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r["split"] == a.split:
            stored[(r["method"], r["item_id"])] = r
    _, llm, ev, _ = build(cfg, None)
    llm.inner = NoLLM(llm.inner)
    ev.rep_seed = int(cfg.seed) * SEED_STRIDE
    ev.dir = Path(tempfile.mkdtemp(prefix="r5checks_"))  # empty stage cache: stages re-run from the LLM cache
    items = make_splits(cfg, ev.bench)[a.split]
    ev.make_start(items)

    keys = {"ours": Counter(), "bon": Counter(), "loop": Counter()}
    label = {"m": None}
    inner_call_meta = llm.call_meta

    def spy(req):
        keys[label["m"]][llm.key(req)] += 1
        return inner_call_meta(req)
    llm.call_meta = spy
    y0tok = {it.item_id: llm.call_y0(y0_request(it, 0))[1] for it in items}

    # ---------------------------------------------------------------- ours, stage by stage
    label["m"] = "ours"
    ours = {}
    skips = Counter()
    loop_saved = {i: {"items": 0, "calls": 0, "tokens_est": 0.0} for i in range(len(ens))}
    stage_cost = {i: [] for i in range(len(ens))}
    per_call = {i: [] for i in range(len(ens))}
    for it in items:
        iid = it.item_id
        y, tok = ev.y0[iid], y0tok[iid]
        traj = [(tok, ev.g(iid, ev.s0[iid]))]
        for i, h in enumerate(ens):
            if ev.g(iid, ev.score_of(it, y)) >= 1.0:
                skips[i] += 1
                traj.append(traj[-1])
                continue
            so = ev._one(h, it, y, 0, ev.occurrence(h, ens[:i]))
            stage_cost[i].append(so.cost)
            per_call[i] += [t["tokens"] for t in so.trace]
            for n in h.nodes:
                if n.k > 1:
                    ran = sum(1 for t in so.trace if t["nid"] == n.nid)
                    if ran < n.k and n.stop == "keep_best":  # keep_best stops early only at g = 1
                        loop_saved[i]["items"] += 1
                        loop_saved[i]["calls"] += n.k - ran
            y, tok = so.solution, tok + so.cost
            traj.append((tok, ev.g(iid, ev.score_of(it, y))))
        ours[iid] = traj
    mism = sum(abs(ours[it.item_id][-1][1] - stored[("ours", it.item_id)]["g"]) > 1e-9
               or abs(ours[it.item_id][-1][0] - stored[("ours", it.item_id)]["tokens"]) > 0.5 for it in items)

    # ---------------------------------------------------------------- best-of-N writer
    label["m"] = "bon"
    bon = {}
    for it in items:
        iid = it.item_id
        best, tok, rows = ev.s0[iid], y0tok[iid], []
        for r in range(a.n_max):
            so = ev._one(writer_role(), it, ev.y0[iid], r)
            tok += so.cost
            best = max(best, so.score)
            rows.append((tok, ev.g(iid, best)))
        bon[iid] = rows
    n_bon = fin["best_of_n_writer"]["n_samples"]
    mism_bon = sum(abs(bon[it.item_id][n_bon - 1][1] - stored[("best_of_n_writer", it.item_id)]["g"]) > 1e-9
                   for it in items)

    # ---------------------------------------------------------------- repair loop (matched budget)
    label["m"] = "loop"
    budget = fin["repair_loop"]["token_budget"]
    loop, loop_stop = {}, Counter()
    loop_saved_tok = []
    for it in items:
        iid = it.item_id
        rep = ev.rep_seed
        tok, s0 = y0tok[iid], ev.s0[iid]
        text, t = llm.call(role_prompt("writer", it, ev.y0[iid], [], None, rep=rep))
        tok += t
        cur, cur_sr = text, ev.bench.score(it, text)
        best_s = max(cur_sr.score, s0)
        traj = [(y0tok[iid], ev.g(iid, s0)), (tok, ev.g(iid, best_s))]
        n = 0
        while tok < budget and n < 40 and ev.g(iid, best_s) < 1.0:
            text, t = llm.call(role_prompt("repairer", it, cur, [], cur_sr.diag, rep=rep + 7919 * (n + 1)))
            tok, n = tok + t, n + 1
            sr = ev.bench.score(it, text)
            if sr.score >= cur_sr.score:
                cur, cur_sr = text, sr
            best_s = max(best_s, sr.score)
            traj.append((tok, ev.g(iid, best_s)))
        why = "g=1" if ev.g(iid, best_s) >= 1.0 else ("budget" if tok >= budget else "max_repairs")
        loop_stop[why] += 1
        if why == "g=1":
            loop_saved_tok.append(budget - tok)
        loop[iid] = traj
    mism_loop = sum(abs(loop[it.item_id][-1][1] - stored[("repair_loop", it.item_id)]["g"]) > 1e-9 for it in items)

    # ---------------------------------------------------------------- outputs
    ids = [it.item_id for it in items]
    md = [f"# Cache-only checks: {a.run} on {a.split} ({len(ids)} items, 1 repeat)", "",
          f"Replay vs stored rows: ours {mism} mismatches, best-of-{n_bon} {mism_bon}, repair loop {mism_loop} "
          "(stages recomputed from the LLM cache).", "",
          "## 1. Mean g against mean tokens per item", "",
          "| point | mean tokens/item | mean g |", "|---|---|---|"]
    curves = {"ours (after stage k)": [], "best-of-N writer (N = 1..11)": [], "repair loop (after r repairs)": []}
    for k in range(len(ens) + 1):
        p = (mean(ours[i][k][0] for i in ids), mean(ours[i][k][1] for i in ids))
        curves["ours (after stage k)"].append(p)
        md.append(f"| ours, y0 + {k} stage(s) | {p[0] / 1000:.1f}k | {p[1]:.3f} |" if k else
                  f"| y0 only | {p[0] / 1000:.1f}k | {p[1]:.3f} |")
    for n in range(1, a.n_max + 1):
        p = (mean(bon[i][n - 1][0] for i in ids), mean(bon[i][n - 1][1] for i in ids))
        curves["best-of-N writer (N = 1..11)"].append(p)
        md.append(f"| best-of-{n} writer | {p[0] / 1000:.1f}k | {p[1]:.3f} |")
    r_max = max(len(v) for v in loop.values())
    for r in range(r_max):  # items that stopped earlier carry their final state
        p = (mean(loop[i][min(r, len(loop[i]) - 1)][0] for i in ids), mean(loop[i][min(r, len(loop[i]) - 1)][1] for i in ids))
        curves["repair loop (after r repairs)"].append(p)
        md.append(f"| repair loop, writer + {r - 1} repairs | {p[0] / 1000:.1f}k | {p[1]:.3f} |" if r else
                  f"| repair loop start (y0) | {p[0] / 1000:.1f}k | {p[1]:.3f} |")

    md += ["", "## 2. Per item: ours against each baseline", "", "| baseline | ours wins | ties | ours loses |",
           "|---|---|---|---|"]
    for name, other in ((f"best-of-{n_bon} writer", {i: bon[i][n_bon - 1][1] for i in ids}),
                        ("repair loop (matched)", {i: loop[i][-1][1] for i in ids})):
        dd = [ours[i][-1][1] - other[i] for i in ids]
        md.append(f"| {name} | {sum(x > 1e-9 for x in dd)} | {sum(abs(x) <= 1e-9 for x in dd)} | "
                  f"{sum(x < -1e-9 for x in dd)} |")
        ties = [i for i, x in zip(ids, dd) if abs(x) <= 1e-9]
        md.append(f"|  (ties with both at g = 1: {sum(ours[i][-1][1] >= 1 for i in ties)}) | | | |")

    dup = sum(stored[("ours", i)].get("dup_stage_runs", 0) for i in ids)
    runs_ = sum(stored[("ours", i)].get("stage_runs", 0) for i in ids)
    ko, kb, kl = set(keys["ours"]), set(keys["bon"]), set(keys["loop"])
    md += ["", "## 5. Duplicates and fresh samples", "",
           f"- duplicate stage runs (same structure, input and repeat): {dup}/{runs_}",
           f"- LLM requests: ours {sum(keys['ours'].values())} ({len(ko)} distinct), best-of-{a.n_max} "
           f"{sum(keys['bon'].values())} ({len(kb)} distinct), repair loop {sum(keys['loop'].values())} "
           f"({len(kl)} distinct)",
           f"- requests made more than once inside ours: {sum(1 for v in keys['ours'].values() if v > 1)}",
           f"- shared keys: ours ∩ best-of-N {len(ko & kb)}, ours ∩ repair loop {len(ko & kl)}, "
           f"best-of-N ∩ repair loop {len(kb & kl)} (expected: the repair loop's writer call = best-of-N sample 0)"]

    md += ["", "## 6. Early stops at g = 1 (score >= max(1, s0 + 0.05), i.e. at least the reference profit)", ""]
    for i, h in enumerate(ens):
        c = mean(stage_cost[i]) if stage_cost[i] else 0
        pc = mean(per_call[i]) if per_call[i] else 0
        md.append(f"- ours stage {i} `{h.describe()}`: skipped on {skips[i]} items (est. {skips[i] * c / 1000:.0f}k "
                  f"tokens saved at its mean cost {c / 1000:.1f}k)" + (
                      f"; loop stopped early on {loop_saved[i]['items']} items, {loop_saved[i]['calls']} calls not "
                      f"made (est. {loop_saved[i]['calls'] * pc / 1000:.0f}k tokens at {pc / 1000:.1f}k/call)"
                      if any(n.k > 1 for n in h.nodes) else ""))
    tot_ours = sum(skips[i] * (mean(stage_cost[i]) if stage_cost[i] else 0) +
                   loop_saved[i]["calls"] * (mean(per_call[i]) if per_call[i] else 0) for i in range(len(ens)))
    md += [f"- ours total: est. {tot_ours / len(ids) / 1000:.1f}k tokens/item saved",
           f"- repair loop (budget {budget / 1000:.1f}k): stop reasons {dict(loop_stop)}; on g = 1 stops the unused "
           f"budget is {sum(loop_saved_tok) / len(ids) / 1000:.1f}k tokens/item (mean over all items)",
           "- best-of-N writer: no early stop (always N samples); the reference enters only through the score, "
           "which is profit / reference profit, a fixed positive rescaling per item, so the ranking is unchanged"]
    out = Path(a.out) if a.out else d / "CHECKS.md"
    out.write_text("\n".join(md) + "\n")
    (out.parent / "curves.json").write_text(json.dumps(curves))
    svg(curves, out.parent / "curves_g_vs_tokens.svg")
    print("\n".join(md))


def svg(curves, path):
    W, H, L, B, T, R = 640, 400, 60, 50, 20, 20
    pts = [p for c in curves.values() for p in c]
    xm = max(p[0] for p in pts) * 1.05 / 1000
    ym = min(1.0, max(p[1] for p in pts) * 1.1)
    X = lambda x: L + (x / 1000) / xm * (W - L - R)  # noqa: E731
    Y = lambda y: H - B - y / ym * (H - B - T)  # noqa: E731
    cols = ["#2563eb", "#dc2626", "#16a34a"]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#333"/>',
         f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="#333"/>',
         f'<text x="{W / 2}" y="{H - 12}" text-anchor="middle">mean tokens per item (k)</text>',
         f'<text x="15" y="{H / 2}" transform="rotate(-90 15 {H / 2})" text-anchor="middle">mean g</text>']
    for k in range(0, int(xm) + 1, 50):
        s.append(f'<text x="{X(k * 1000)}" y="{H - B + 15}" text-anchor="middle">{k}</text>')
    for j in range(0, 11):
        y = j / 10
        if y <= ym:
            s.append(f'<text x="{L - 6}" y="{Y(y) + 4}" text-anchor="end">{y:.1f}</text>')
    for (name, c), col, yy in zip(curves.items(), cols, (T + 10, T + 26, T + 42)):
        s.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="'
                 + " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in c) + '"/>')
        s += [f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{col}"/>' for x, y in c]
        s.append(f'<text x="{L + 15}" y="{yy}" fill="{col}">{name}</text>')
    s.append("</svg>")
    path.write_text("\n".join(s))


if __name__ == "__main__":
    main()
