"""Aggregate runs into runs/REPORT.md (+ learning-curve plot).

    python scripts/report.py [--out REPORT.md] RUN [RUN ...]

Runs whose ids differ only by a trailing `_s<seed>` are pooled as one preset (e.g. farm_gbdt_s0..s2).
Per preset and split, for each method:
  * mean g, per-seed mean g
  * legal rate (fraction of (item, repeat) final plans that are legal) and mean raw score among legal plans
  * paired bootstrap 90% intervals of the mean difference in g against the single writer and against
    best-of-N on the same items (per item: method - baseline inside the same run, averaged over seeds;
    items are resampled)
  * inference tokens and scorer (verifier) calls per item
Needs runs/<id>/per_item.jsonl (written by boosttopo.run since this round); older runs get a reduced table.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

RUNS = Path(__file__).resolve().parents[1] / "runs"
BASES = ("single_writer", "writer_once", "best_of_n_matched", "repair_loop")
BASE_LABEL = {"single_writer": "Δg vs y0", "writer_once": "Δg vs writer once",
              "best_of_n_matched": "Δg vs best-of-N (y0 prompt)", "repair_loop": "Δg vs repair loop"}
METHOD_ORDER = ("ours", "single_writer", "writer_once", "best_of_n_matched", "best_of_n_writer", "repair_loop",
                "hand_topology")


def load(run):
    d = RUNS / run
    rd = lambda n: json.loads((d / n).read_text()) if (d / n).exists() else None  # noqa: E731
    rounds = [json.loads(l) for l in (d / "rounds.jsonl").read_text().splitlines()] if (d / "rounds.jsonl").exists() else []
    rows = [json.loads(l) for l in (d / "per_item.jsonl").read_text().splitlines()] if (d / "per_item.jsonl").exists() else None
    return rd("final.json"), rd("ensemble.json"), rounds, rows


def paired_ci(diffs, level=0.90, n_boot=4000, seed=0):
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(diffs)
    ms = sorted(mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(n_boot))
    a = (1 - level) / 2
    return ms[int(a * n_boot)], ms[min(n_boot - 1, int((1 - a) * n_boot))]


def preset_of(run):
    return re.sub(r"_s\d+$", "", run)


def item_means(rows):
    """rows of one (run, split, method) -> {item: mean g over repeats}."""
    by = defaultdict(list)
    for r in rows:
        by[r["item_id"]].append(r["g"])
    return {i: mean(v) for i, v in by.items()}


def split_table(per_run_rows, split):
    """per_run_rows: {run: rows}. Returns markdown lines for one split pooled over the runs (seeds)."""
    sel = {run: [r for r in rows if r["split"] == split] for run, rows in per_run_rows.items()}
    methods = sorted({r["method"] for rows in sel.values() for r in rows},
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    n_items = len({r["item_id"] for rows in sel.values() for r in rows})
    out = [f"**{split}** ({n_items} items, {len(sel)} seed(s))", "",
           "| method | mean g | per-seed g | " + " | ".join(f"{BASE_LABEL[b]} [90% CI]" for b in BASES) +
           " | legal rate | score among legal | tokens/item | scorer calls/item | duplicate stage runs |",
           "|---|---|---|" + "---|" * len(BASES) + "---|---|---|---|---|"]
    for m in methods:
        per_seed, legal_flags, legal_scores, toks, calls = [], [], [], [], []
        dup, stages = 0, 0
        diffs = {b: defaultdict(list) for b in BASES}
        for run, rows in sel.items():
            mr = [r for r in rows if r["method"] == m]
            if not mr:
                continue
            im = item_means(mr)
            per_seed.append(mean(im.values()))
            legal_flags += [r["valid"] for r in mr]
            legal_scores += [r["score"] for r in mr if r["valid"]]
            dup += sum(r.get("dup_stage_runs", 0) for r in mr)
            stages += sum(r.get("stage_runs", 0) for r in mr)
            by_item = defaultdict(list)
            for r in mr:
                by_item[r["item_id"]].append(r)
            toks.append(mean(mean(x["tokens"] for x in v) for v in by_item.values()))
            calls.append(mean(mean(x["scorer_calls"] for x in v) for v in by_item.values()))
            for b in BASES:
                bi = item_means([r for r in rows if r["method"] == b])
                for i, g in im.items():
                    if i in bi:
                        diffs[b][i].append(g - bi[i])
        cells = []
        for b in BASES:
            d = [mean(v) for v in diffs[b].values()]
            if m == b or not d:
                cells.append("–")
            else:
                lo, hi = paired_ci(d)
                cells.append(f"{mean(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")
        ls = f"{mean(legal_scores):.3f} (n={len(legal_scores)})" if legal_scores else "– (n=0)"
        out.append(f"| {m} | {mean(per_seed):.3f} | {', '.join(f'{x:.3f}' for x in per_seed)} | {' | '.join(cells)} | "
                   f"{mean(legal_flags):.2f} | {ls} | {mean(toks) / 1000:.1f}k | {mean(calls):.1f} | "
                   f"{f'{dup / stages:.1%} ({dup}/{stages})' if stages else '–'} |")
    return out + [""]


def run_section(run, fin, ens, rounds, curves):
    out = [f"### {run}"]
    if ens:
        out.append(f"- fit tokens: {ens['fit_budget']['logical_tokens'] / 1e6:.2f}M (new {ens['fit_budget']['new_tokens'] / 1e6:.2f}M); "
                   f"fit time {ens['fit_seconds'] / 3600:.2f} h")
        out.append("- ensemble: " + "; ".join(f"`{s}`" for s in ens["describe"]))
        x = ens.get("extra", {})
        if x.get("offline_desc") is not None:
            out.append(f"- offline selection heldout g {x.get('offline_heldout_g', 0):.3f} vs round-built "
                       f"{x.get('round_heldout_g', 0):.3f}; replaced={x.get('replaced')}")
    if fin and fin["splits"]:
        any_s = next(iter(fin["splits"].values()))
        if "best_of_n_matched" in any_s:
            out.append("- best-of-N samples (y0 prompt / writer role), mean repairs in repair loop: " + ", ".join(
                f"{s}={v['best_of_n_matched'].get('n_samples')}/{v.get('best_of_n_writer', {}).get('n_samples', '-')}"
                f"/{v.get('repair_loop', {}).get('mean_repairs', float('nan')):.1f}" for s, v in fin["splits"].items()))
    if rounds:
        out += ["", "| round | chosen | ho gain mean | ho gain LCB90 | accepted | train g | heldout g | fit tokens (M) |",
                "|---|---|---|---|---|---|---|---|"]
        for r in rounds:
            out.append(f"| {r['round']} | `{r['chosen']}` | {r['ho_gain_mean']:.3f} | {r['ho_gain_lci']:.3f} | "
                       f"{r['accepted']} | {r['train_g']:.3f} | {r['heldout_g']:.3f} | {r['budget']['logical_tokens'] / 1e6:.2f} |")
        curves[run] = [(0.0, 0.0)] + [(r["budget"]["logical_tokens"] / 1e6, r["heldout_g"]) for r in rounds]
    return out + [""]


def legacy_table(fin):
    splits = fin["splits"]
    methods = sorted({m for s in splits.values() for m in s}, key=lambda m: (m != "ours", m))
    out = ["(no per-item file: older run, no paired intervals)", "", "| method | " + " | ".join(splits) + " |",
           "|---|" + "---|" * len(splits)]
    for m in methods:
        out.append(f"| {m} | " + " | ".join(f"g={splits[s][m]['mean_g']:.3f} legal={splits[s][m]['valid_rate']:.2f}"
                                            if m in splits[s] else "-" for s in splits) + " |")
    return out + [""]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="REPORT.md")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args(argv)
    presets = defaultdict(list)
    for run in a.runs:
        presets[preset_of(run)].append(run)
    out = ["# Boosted Topology -- experiment report", "",
           "g = gap closure vs the run's start solution y0 (single_writer = y0, benchmark prompt), so y0 has g = 0 by "
           "construction and 'Δg vs y0' equals the method's g. writer_once = the writer role (ledger prompt) on y0, "
           "sampled once per repeat and gated like every stage; best_of_n_writer / repair_loop use that role at our "
           "token budget. Intervals: paired bootstrap 90% over items. "
           "'heldout' is used for round acceptance, so it is not a test split; dev_* splits are development tests "
           "(evaluated by every run).", ""]
    curves = {}
    for preset, runs in presets.items():
        out.append(f"## {preset}")
        loaded = {r: load(r) for r in runs}
        rows = {r: v[3] for r, v in loaded.items() if v[3] is not None}
        if rows:
            splits = []
            for rr in rows.values():
                for r in rr:
                    if r["split"] not in splits:
                        splits.append(r["split"])
            for s in splits:
                out += split_table(rows, s)
        for r, (fin, ens, rounds, rws) in loaded.items():
            out += run_section(r, fin, ens, rounds, curves)
            if rws is None and fin:
                out += legacy_table(fin)
    (RUNS / a.out).write_text("\n".join(out))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for run, pts in curves.items():
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=run)
        ax.set_xlabel("cumulative LLM tokens during fitting (M)")
        ax.set_ylabel("heldout mean g (ensemble)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(RUNS / (Path(a.out).stem + "_heldout_g_vs_tokens.png"), dpi=150)
    except ImportError:
        pass
    print("\n".join(out))


if __name__ == "__main__":
    main()
