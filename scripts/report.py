#!/usr/bin/env python3
"""Report (doc §8.2) from a saved results JSON (precheck.json or test.json).

    python3.11 scripts/report.py runs/precheck/precheck.json [--ref single_draft]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow.experiment import compare, method_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--ref", default=None, help="method to compare every other method against")
    args = ap.parse_args()
    d = json.load(open(args.path))
    rows, wfs = d["results"], d["methods"]
    ref = args.ref or ("ours" if "ours" in rows else "single_draft")
    print(f"# Report: {args.path}  (all numbers measured from the saved run)\n")
    print("| method | mean g | score | legal | legal raw | tokens | calls | scorer | tools |"
          f" diff vs {ref} [90% CI] | ill->legal | legal up |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, r in rows.items():
        m = method_metrics(r)
        c = compare(r, rows[ref]) if n != ref else None
        lr = f"{m['legal_mean_raw']:.3f}" if m["legal_mean_raw"] is not None else "-"
        cs = f"{c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]" if c else "-"
        fx = f"{c['from_illegal_to_legal']:+.3f}" if c else "-"
        up = f"{c['from_legal_score_up']:+.3f}" if c else "-"
        print(f"| {n} | {m['mean_g']:.3f} | {m['mean_score']:.3f} | {m['legal_rate']:.2f} | {lr} | "
              f"{m['tokens']:.0f} | {m['llm_calls']:.2f} | {m['scorer_calls']:.2f} | "
              f"{m['tool_calls']:.2f} | {cs} | {fx} | {up} |")
    print("\n## Workflow structure and usage\n")
    for n, r in rows.items():
        m = method_metrics(r)
        print(f"- **{n}**: `{wfs.get(n, '?')}`")
        if m["branch_usage"]:
            print(f"  - branch usage (items per label): {m['branch_usage']}")
        if m["loop_iterations"]:
            print(f"  - loop iterations run -> count: {m['loop_iterations']}")
    if "search" in d:
        s = d["search"]
        print(f"\n## Search\n\nmode={s.get('mode')}, candidate evals={s.get('evals')}, "
              f"new model calls={s.get('search_new_llm_calls')}")
        for rd in s.get("rounds", []):
            print(f"- round {rd['round']}: frag `{rd['frag']}` target={rd.get('target', '-')} "
                  f"accepted={rd.get('accepted')} heldout={rd.get('heldout')}")


if __name__ == "__main__":
    main()
