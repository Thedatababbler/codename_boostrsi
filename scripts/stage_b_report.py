"""Round 4 Stage B report (cache-only: any LLM cache miss raises).

    PYTHONPATH=src python scripts/stage_b_report.py r4_gbdt_lm_s0 [r4_gbdt_lm_s1 ...] [--stage-a runs/r4_stageA]

Per run, on the test split (test_d5):
  * ours vs the repair loop at ours' token budget (matched): mean g, paired 90% bootstrap interval of the per-item
    difference; legal rate; mean score among legal plans; tokens and scorer calls per item; duplicate stage runs
  * the same against the Stage A baselines (same items, seed-0 samples)
  * accepted members: structure, loop settings, and per member the split of its test-set gain into
    illegal -> legal items and legal items scoring higher (replayed stage by stage from the stage cache)
"""
import argparse
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from boosttopo.config import load_config  # noqa: E402
from boosttopo.graph import Subgraph  # noqa: E402
from boosttopo.infer import SEED_STRIDE  # noqa: E402
from boosttopo.run import build, make_splits  # noqa: E402
from report import paired_ci  # noqa: E402


class CacheMiss(Exception):
    pass


class NoLLM:
    def __init__(self, inner):
        self.name, self.temperature = inner.name, inner.temperature
        self.max_tokens, self.extra_body = getattr(inner, "max_tokens", None), getattr(inner, "extra_body", None)

    def call(self, req):
        raise CacheMiss(f"cache miss: role={req.role} rep={req.rep}")


def rows_of(path, split=None):
    if not Path(path).exists():
        return []
    rs = [json.loads(l) for l in Path(path).read_text().splitlines()]
    return [r for r in rs if split is None or r.get("split", split) == split]


def by_method(rows):
    out = {}
    for r in rows:
        out.setdefault(r["method"], {})[r["item_id"]] = r
    return out


def summary(rs):
    v = list(rs.values())
    legal = [r["score"] for r in v if r["valid"]]
    return {"g": mean(r["g"] for r in v), "legal": mean(r["valid"] for r in v),
            "legal_score": mean(legal) if legal else None, "tokens": mean(r["tokens"] for r in v),
            "calls": mean(r["scorer_calls"] for r in v),
            "dup": sum(r.get("dup_stage_runs", 0) for r in v), "stages": sum(r.get("stage_runs", 0) for r in v)}


def delta(rs, base):
    if base is None or rs is base:
        return "–"
    d = [rs[i]["g"] - base[i]["g"] for i in sorted(base) if i in rs]
    lo, hi = paired_ci(d)
    return f"{mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]"


def line(name, rs, base, base2=None):
    s = summary(rs)
    dd = delta(rs, base) + " | " + delta(rs, base2)
    ls = "–" if s["legal_score"] is None else f"{s['legal_score']:.3f}"
    dup = f"{s['dup']}/{s['stages']}" if s["stages"] else "–"
    return (f"| {name} | {s['g']:.3f} | {dd} | {s['legal']:.2f} | {ls} | {s['tokens'] / 1000:.1f}k | "
            f"{s['calls']:.1f} | {dup} |")


def member_attribution(run_dir, cfg, ens, split):
    """Replay ours on the split stage by stage (cache only) and split each member's gain."""
    _, llm, ev, _ = build(cfg, None)
    llm.inner = NoLLM(llm.inner)
    ev.rep_seed = int(cfg.seed) * SEED_STRIDE
    items = make_splits(cfg, ev.bench)[split]
    ev.make_start(items)
    per = [{"n_run": 0, "to_legal": 0, "to_legal_g": 0.0, "higher": 0, "higher_g": 0.0, "other_g": 0.0}
           for _ in ens]
    n = len(items) * cfg.eval.n_repeats
    for it in items:
        for r in range(cfg.eval.n_repeats):
            y = ev.y0[it.item_id]
            for i, h in enumerate(ens):
                g_in = ev.g(it.item_id, ev.score_of(it, y))
                if g_in >= 1.0:
                    break
                legal_in = ev.bench.score(it, y).valid
                so = ev._one(h, it, y, r, ev.occurrence(h, ens[:i]))
                y = so.solution
                sr = ev.bench.score(it, y)
                dg = ev.g(it.item_id, sr.score) - g_in
                p = per[i]
                p["n_run"] += 1
                if not legal_in and sr.valid:
                    p["to_legal"] += 1
                    p["to_legal_g"] += dg
                elif legal_in and sr.valid and dg > 0:
                    p["higher"] += 1
                    p["higher_g"] += dg
                else:
                    p["other_g"] += dg
    for p in per:
        for k in ("to_legal_g", "higher_g", "other_g"):
            p[k] /= n  # contribution to the mean g over the split
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--split", default="test_d5")
    ap.add_argument("--stage-a", default="runs/r4_stageA")
    ap.add_argument("--out", default="runs/R4_STAGE_B.md")
    a = ap.parse_args()
    stage_a = by_method(rows_of(ROOT / a.stage_a / "per_item.jsonl"))
    md = ["# Round 4 Stage B: GBDT + loop edit + macro switch (d5, Qwen3-14B thinking, fp8 KV cache)", "",
          f"Test split `{a.split}` (the 40 calibrated d5 items), 1 repeat. Intervals: paired bootstrap 90% over "
          "items. `repair_loop` = the repair loop at ours' actual tokens per item (matched); Stage A baselines are "
          "seed-0 samples on the same items. Scores among legal plans are the scorer's score (clipped at 0 "
          "below only).", ""]
    for run in a.runs:
        d = ROOT / "runs" / run
        cfg = load_config(None, json.loads((d / "config.json").read_text())["config"])
        ensj = json.loads((d / "ensemble.json").read_text())
        ens = [Subgraph.from_dict(x) for x in ensj["ensemble"]]
        rows = by_method(rows_of(d / "per_item.jsonl", a.split))
        md += [f"## {run} (seed {cfg.seed})", "",
               "| method | mean g | Δg (row − repair loop) [90% CI] | Δg (row − best-of-N writer) [90% CI] | "
               "legal rate | mean score among legal | tokens/item | scorer calls/item | duplicate stage runs |",
               "|---|---|---|---|---|---|---|---|---|"]
        base, base2 = rows.get("repair_loop"), rows.get("best_of_n_writer")
        for m in ("ours", "repair_loop", "best_of_n_writer"):
            if m in rows:
                md.append(line(m, rows[m], base, base2))
        for m, rs in stage_a.items():
            md.append(line(f"Stage A {m}", rs, base, base2))
        if "ours" in rows:
            md += ["", "Ours minus each Stage A baseline: " + "; ".join(
                f"{m} {mean(rows['ours'][i]['g'] - rs[i]['g'] for i in rs):+.3f} "
                f"[{paired_ci([rows['ours'][i]['g'] - rs[i]['g'] for i in sorted(rs)])[0]:+.3f}, "
                f"{paired_ci([rows['ours'][i]['g'] - rs[i]['g'] for i in sorted(rs)])[1]:+.3f}]"
                for m, rs in stage_a.items())]
        fin = json.loads((d / "final.json").read_text())
        rl = fin["splits"][a.split].get("repair_loop", {})
        bw = fin["splits"][a.split].get("best_of_n_writer")
        if bw:
            md += ["", f"Best-of-N writer: N = {bw['n_samples']} (rounded down from ours' tokens), actual "
                   f"{bw['tokens_per_item'] / 1000:.1f}k tokens/item vs ours "
                   f"{fin['splits'][a.split]['ours']['tokens_per_item'] / 1000:.1f}k."]
        md += ["", f"Repair loop budget {rl.get('token_budget', 0) / 1000:.1f}k tokens/item, mean repairs "
               f"{rl.get('mean_repairs', float('nan')):.1f}. Fit: {ensj.get('fit_seconds', 0) / 3600:.1f} h, "
               f"budget {ensj.get('fit_budget')}.", "", "### Rounds", "",
               "| round | chosen | heldout gain mean | lower CI | accepted |", "|---|---|---|---|---|"]
        for h in ensj["history"]:
            md.append(f"| {h['round']} | `{h['chosen']}` | {h['ho_gain_mean']:+.3f} | {h['ho_gain_lci']:+.3f} | "
                      f"{h['accepted']} |")
        md += ["", "### Accepted members (test split, replayed from cache)", "",
               "| # | structure | loops (node: k, stop rule) | stage runs | illegal→legal: items, Δg | "
               "legal→higher: items, Δg | other Δg |", "|---|---|---|---|---|---|---|"]
        try:
            att = member_attribution(d, cfg, ens, a.split)
        except CacheMiss as exc:
            att = None
            md.append(f"(attribution skipped: {exc})")
        for i, h in enumerate(ens):
            loops = ", ".join(f"{n.nid}: {n.k}, {n.stop}" for n in h.nodes if n.k > 1) or "none"
            p = att[i] if att else None
            md.append(f"| {i} | `{h.describe()}` | {loops} | " + (
                f"{p['n_run']} | {p['to_legal']}, {p['to_legal_g']:+.3f} | {p['higher']}, {p['higher_g']:+.3f} | "
                f"{p['other_g']:+.3f} |" if p else "– | – | – | – |"))
        md.append("")
    (ROOT / a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
