"""Cache-only extra checks on Stage B seed 0 (appended to runs/<run>/CHECKS.md). No LLM call: a miss raises.

  a. paired bootstrap 90% interval of ours-after-stage-1 minus best-of-10 writer, and minus the repair loop after
     9 repairs (the matched-budget loop truncated at 9 repairs)
  b. best-of-N writer that stops sampling an item once it reaches g = 1 (the rule ours uses): tokens/item at N = 10
     and 11, and the largest N affordable at ours' final tokens/item, with its g when those samples are cached
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
from boosttopo.infer import SEED_STRIDE, writer_role  # noqa: E402
from boosttopo.llm import sha  # noqa: E402
from boosttopo.roles import prompt_version, role_prompt, y0_request  # noqa: E402
from boosttopo.run import build, make_splits  # noqa: E402
from r5_checks import NoLLM  # noqa: E402
from report import paired_ci  # noqa: E402


def stage_cached(ev, h, it, y_in, rep):
    """True if ev._one(h, it, y_in, rep) is in the stage cache (or its LLM requests are in the LLM cache)."""
    r = rep + ev.rep_base + ev.rep_seed
    k = sha(json.dumps([h.key(), it.item_id, sha(y_in), r, ev.llm.name, prompt_version()]))
    if (ev.dir / k[:2] / f"{k}.json").exists():
        return True
    lk = ev.llm.key(role_prompt("writer", it, y_in, [], None, rep=r))
    return (ev.llm.dir / lk[:2] / f"{lk}.json").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r4_gbdt_lm_s0")
    ap.add_argument("--split", default="test_d5")
    a = ap.parse_args()
    d = ROOT / "runs" / a.run
    cfg = load_config(None, json.loads((d / "config.json").read_text())["config"])
    ens = [Subgraph.from_dict(x) for x in json.loads((d / "ensemble.json").read_text())["ensemble"]]
    fin = json.loads((d / "final.json").read_text())["splits"][a.split]
    _, llm, ev, _ = build(cfg, None)
    llm.inner = NoLLM(llm.inner)
    ev.rep_seed = int(cfg.seed) * SEED_STRIDE
    items = make_splits(cfg, ev.bench)[a.split]
    ev.make_start(items)
    ids = [it.item_id for it in items]
    y0tok = {it.item_id: llm.call_y0(y0_request(it, 0))[1] for it in items}

    # ours after stage 1
    o1 = {}
    for it in items:
        y = ev.y0[it.item_id]
        tok = y0tok[it.item_id]
        for i, h in enumerate(ens[:2]):
            if ev.g(it.item_id, ev.score_of(it, y)) >= 1.0:
                continue
            so = ev._one(h, it, y, 0, ev.occurrence(h, ens[:i]))
            y, tok = so.solution, tok + so.cost
        o1[it.item_id] = (tok, ev.g(it.item_id, ev.score_of(it, y)))
    # best-of-N samples (cached reps 0..)
    samples = {}
    for it in items:
        xs, r = [], 0
        while stage_cached(ev, writer_role(), it, ev.y0[it.item_id], r):
            so = ev._one(writer_role(), it, ev.y0[it.item_id], r)
            xs.append((so.cost, so.score))
            r += 1
        samples[it.item_id] = xs
    n_cached = min(len(v) for v in samples.values())

    def bon(iid, n, early):
        tok, best = y0tok[iid], ev.s0[iid]
        for c, s in samples[iid][:n]:
            if early and ev.g(iid, best) >= 1.0:
                break
            tok += c
            best = max(best, s)
        return tok, ev.g(iid, best)

    # repair loop after 9 repairs (matched-budget loop, truncated)
    budget = fin["repair_loop"]["token_budget"]
    rl9 = {}
    for it in items:
        iid, rep = it.item_id, ev.rep_seed
        text, t = llm.call(role_prompt("writer", it, ev.y0[iid], [], None, rep=rep))
        tok = y0tok[iid] + t
        cur, cur_sr = text, ev.bench.score(it, text)
        best = max(cur_sr.score, ev.s0[iid])
        n = 0
        while tok < budget and n < 9 and ev.g(iid, best) < 1.0:
            text, t = llm.call(role_prompt("repairer", it, cur, [], cur_sr.diag, rep=rep + 7919 * (n + 1)))
            tok, n = tok + t, n + 1
            sr = ev.bench.score(it, text)
            if sr.score >= cur_sr.score:
                cur, cur_sr = text, sr
            best = max(best, sr.score)
        rl9[iid] = (tok, ev.g(iid, best))

    def ci(x, y):
        dd = [x[i][1] - y[i][1] for i in ids]
        lo, hi = paired_ci(dd)
        w = sum(v > 1e-9 for v in dd), sum(abs(v) <= 1e-9 for v in dd), sum(v < -1e-9 for v in dd)
        return f"{mean(dd):+.3f} [{lo:+.3f}, {hi:+.3f}] (wins/ties/losses {w[0]}/{w[1]}/{w[2]})"

    b10 = {i: bon(i, 10, False) for i in ids}
    md = ["", "## a. Ours after stage 1 against the baselines at about the same tokens (cache-only)", "",
          f"- ours after stage 1: {mean(v[0] for v in o1.values()) / 1000:.1f}k tokens/item, g "
          f"{mean(v[1] for v in o1.values()):.3f}",
          f"- minus best-of-10 writer ({mean(v[0] for v in b10.values()) / 1000:.1f}k, g "
          f"{mean(v[1] for v in b10.values()):.3f}): {ci(o1, b10)}",
          f"- minus repair loop after 9 repairs ({mean(v[0] for v in rl9.values()) / 1000:.1f}k, g "
          f"{mean(v[1] for v in rl9.values()):.3f}): {ci(o1, rl9)}",
          "", "## b. Best-of-N writer with ours' early stop (stop sampling an item once it reaches g = 1)", "",
          f"Cached writer samples per item: {n_cached} (reps 0..{n_cached - 1}).", "",
          "| N | tokens/item, no early stop | tokens/item, early stop | g (same either way) |", "|---|---|---|---|"]
    for n in range(1, n_cached + 1):
        a_, b_ = [bon(i, n, False) for i in ids], [bon(i, n, True) for i in ids]
        assert all(abs(x[1] - y[1]) < 1e-12 for x, y in zip(a_, b_))
        md.append(f"| {n} | {mean(x[0] for x in a_) / 1000:.1f}k | {mean(x[0] for x in b_) / 1000:.1f}k | "
                  f"{mean(x[1] for x in a_):.3f} |")
    target = fin["ours"]["tokens_per_item"]
    afford = max(n for n in range(1, n_cached + 1) if mean(bon(i, n, True)[0] for i in ids) <= target)
    tok_n = mean(bon(i, n_cached, True)[0] for i in ids)
    # per-sample mean cost of items still below g = 1 after n_cached samples (to extrapolate beyond the cache)
    open_items = [i for i in ids if bon(i, n_cached, True)[1] < 1.0]
    per = mean(c for i in open_items for c, _ in samples[i]) if open_items else 0
    md += ["", f"- ours' final cost: {target / 1000:.1f}k tokens/item. Largest cached N within it under the early "
           f"stop: N = {afford} (g {mean(bon(i, afford, True)[1] for i in ids):.3f})."]
    if afford == n_cached:
        extra = (target - tok_n) / (len(open_items) * per / len(ids)) if open_items and per else 0
        md.append(f"- Beyond the cache (inferred, not run): {len(open_items)} items stay below g = 1 after "
                  f"{n_cached} samples; at their mean {per / 1000:.1f}k tokens/sample the remaining "
                  f"{(target - tok_n) / 1000:.1f}k tokens/item buy about {extra:.1f} more samples per open item, "
                  f"i.e. N ≈ {n_cached + int(extra)} (rounded down). g at that N is not available "
                  "(samples not cached).")
    with open(d / "CHECKS.md", "a") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
