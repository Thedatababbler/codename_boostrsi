"""End-to-end run: items -> y0 -> boosting (gbdt | adaboost) -> inference on heldout and test sets + baselines.

    python -m boosttopo.run --config configs/farm_gbdt.yaml [--set key.sub=value ...] [--no-test]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path

from .bench.base import make_bench
from .boost.adaboost import fit_adaboost
from .boost.common import Ctx
from .boost.gbdt import fit_gbdt
from .boost.library import Library
from .boost.sampler import make_sampler
from .config import cfg_hash, deep_merge, load_config, parse_override
from .evalx import Evaluator
from .infer import SEED_STRIDE, best_of_n, best_of_n_subgraph, hand_topology, infer, repair_loop, writer_role
from .llm import build_llm
from .policy import make_policy
from .runlog import RunLog
from .search.proposers import make_proposer


def seeds_of(spec) -> list[int]:
    a, b = spec.seeds
    return list(range(a, b))


def code_version(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        h = hashlib.sha256()
        for p in sorted((root / "src").rglob("*.py")):
            h.update(p.read_bytes())
        return "src-" + h.hexdigest()[:12]


def build(cfg, run_dir: str | None):
    bench = make_bench(cfg)
    fake = None
    if cfg.llm.kind == "fake":
        from .bench.toy import toy_rule

        fake = toy_rule
    llm = build_llm(cfg, cfg.cache_dir, fake_rule=fake)
    ev = Evaluator(bench, llm, cfg.cache_dir, n_repeats=cfg.eval.n_repeats, lam=0.0,
                   max_workers=cfg.llm.get("max_concurrency", 32))
    ev.dedup_stages = bool(cfg.get("fresh_duplicate_stages", False))
    log = RunLog(run_dir)
    return bench, llm, ev, log


def make_splits(cfg, bench, final_test: bool = False) -> dict:
    """train, heldout and the dev-test splits; the final-test splits only when final_test=True.
    Item ids may not overlap across any splits, including the final-test ones (checked either way)."""
    b = cfg.bench
    game = b.game
    splits = {"train": bench.make_items(game, b.train.difficulty, seeds_of(b.train)),
              "heldout": bench.make_items(game, b.heldout.difficulty, seeds_of(b.heldout))}
    for name, spec in (b.get("dev_tests") or {}).items():
        splits[name] = bench.make_items(game, spec.difficulty, seeds_of(spec))
    final = {name: bench.make_items(game, spec.difficulty, seeds_of(spec))
             for name, spec in (b.get("final_tests") or {}).items()}
    if set(final) & set(splits):
        raise ValueError("final-test split names clash with other splits")
    ids = [it.item_id for its in list(splits.values()) + list(final.values()) for it in its]
    if len(ids) != len(set(ids)):
        raise ValueError("item ids overlap between splits")
    if final_test:
        splits.update(final)
    return splits


def eval_splits(cfg, final_test: bool) -> list[str]:
    """Splits evaluated after fitting: dev tests (+ heldout) normally, only the final tests with --final-test."""
    if final_test:
        return list((cfg.bench.get("final_tests") or {}).keys())
    return ["heldout"] + list((cfg.bench.get("dev_tests") or {}).keys())


def calibrate_lam(cfg, ev, train) -> float:
    """lam so that a 3-node subgraph's cost penalty ~= 0.02 g (doc section 10.1)."""
    sample = train[: min(8, len(train))]
    res = ev.run(hand_topology(), sample, {it.item_id: ev.y0[it.item_id] for it in sample}, n_repeats=1)
    c = ev.mean_cost(res)
    return 0.02 * 1000.0 / c if c > 0 else 0.0


def fit(cfg, splits, ev, log):
    train, heldout = splits["train"], splits["heldout"]
    ctx = Ctx(cfg=cfg, ev=ev, proposer=make_proposer(cfg), policy=make_policy(cfg), sampler=make_sampler(cfg),
              log=log, library=Library(log))
    fn = fit_gbdt if cfg.mode == "gbdt" else fit_adaboost
    return fn(train, heldout, ctx)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-test", action="store_true")
    ap.add_argument("--refit", action="store_true", help="refit even if runs/<id>/ensemble.json exists")
    ap.add_argument("--baselines", action="store_true", help="also run the baselines of section 11.1")
    ap.add_argument("--methods", default=None,
                    help="comma list: only (re)compute these methods and merge them into the stored results")
    ap.add_argument("--splits", default=None, help="comma list of evaluation splits (default: all dev splits)")
    ap.add_argument("--final-test", action="store_true",
                    help="evaluate the one-time final-test splits (and nothing else); never use during development")
    args = ap.parse_args(argv)
    over = {}
    for s in args.set:
        over = deep_merge(over, parse_override(s))
    cfg = load_config(args.config, over)
    run_id = cfg.run_id or Path(args.config).stem
    root = Path(__file__).resolve().parents[2]
    run_dir = root / "runs" / run_id
    bench, llm, ev, log = build(cfg, str(run_dir))
    ev.rep_seed = int(cfg.seed) * SEED_STRIDE
    splits = make_splits(cfg, bench, final_test=args.final_test)
    log.write("manifest_final.json" if args.final_test else "manifest.json",
              {k: [it.item_id for it in v] for k, v in splits.items()})
    t0 = time.time()
    ev.make_start(splits["train"] + splits["heldout"])
    if cfg.lam == "auto":
        cfg.lam = calibrate_lam(cfg, ev, splits["train"])
    ev.lam = float(cfg.lam)
    # a test-only rerun on an already fitted ensemble keeps the fit's config.json and records its own
    reuse = bool(log.read("ensemble.json")) and not args.refit
    log.write("config_test.json" if reuse else "config.json",
              {"config": cfg.to_dict(), "hash": cfg_hash(cfg.to_dict()), "code_version": code_version(root)})
    print(f"[{run_id}] y0 ready; lam={ev.lam:.5f}; budget={llm.budget.snapshot()}", flush=True)

    prev = log.read("ensemble.json")
    if prev and not args.refit:
        from .boost.common import FitResult
        from .graph import Subgraph

        fr = FitResult(prev["mode"], [Subgraph.from_dict(d) for d in prev["ensemble"]], prev["history"],
                       prev.get("extra", {}))
        print(f"[{run_id}] reusing fitted ensemble from ensemble.json", flush=True)
        return _test(args, cfg, run_id, splits, ev, llm, log, fr, prev["fit_budget"])
    if args.final_test:
        raise SystemExit("--final-test needs a fitted runs/<id>/ensemble.json; fit first without it")
    fr = fit(cfg, splits, ev, log)
    fit_budget = llm.budget.snapshot()
    log.write("ensemble.json", {"mode": fr.mode, "ensemble": [h.to_dict() for h in fr.ensemble],
                                "describe": [h.describe() for h in fr.ensemble], "history": fr.history,
                                "extra": fr.extra, "fit_budget": fit_budget, "fit_seconds": time.time() - t0})
    print(f"[{run_id}] fit done: {[h.describe() for h in fr.ensemble]}; budget={fit_budget}", flush=True)
    if args.no_test:
        return fr
    return _test(args, cfg, run_id, splits, ev, llm, log, fr, fit_budget)


ALL_BASELINES = ("single_writer", "hand_topology", "best_of_n_matched", "writer_once", "best_of_n_writer",
                 "repair_loop")


def _test(args, cfg, run_id, splits, ev, llm, log, fr, fit_budget):
    """Inference on the evaluation splits. With --methods / --splits only those are (re)computed and merged
    into the existing final.json / per_item.jsonl; every other stored result is kept as it was."""
    rnd = math.floor if (cfg.get("baselines") or {}).get("bon_round", "ceil") == "floor" else math.ceil
    tag = "_final_test" if args.final_test else ""
    only = set(args.methods.split(",")) if getattr(args, "methods", None) else None
    split_sel = args.splits.split(",") if getattr(args, "splits", None) else eval_splits(cfg, args.final_test)
    prev = log.read(f"final{tag}.json") if only else None
    report = prev or {"run_id": run_id, "mode": fr.mode, "ensemble": [h.describe() for h in fr.ensemble],
                      "fit_budget": fit_budget, "lam": ev.lam, "seed": cfg.seed, "splits": {}}
    rows_out = []
    if only and log.dir and (log.dir / f"per_item{tag}.jsonl").exists():
        rows_out = [json.loads(l) for l in (log.dir / f"per_item{tag}.jsonl").read_text().splitlines()]
        rows_out = [r for r in rows_out if not (r["split"] in split_sel and r["method"] in only)]
    want = (lambda m: m in only) if only else (lambda m: m == "ours" or (args.baselines and m in ALL_BASELINES))
    n_rep = cfg.eval.n_repeats
    for name in split_sel:
        items = splits[name]
        ev.make_start(items)
        sec, aux = {}, {}
        cache = {}

        def get(m):  # computed on demand (dependencies are cache hits once computed)
            if m not in cache:
                if m == "ours":
                    cache[m] = infer(fr.mode, fr.ensemble, items, ev, n_rep)
                elif m == "single_writer":
                    cache[m] = infer("adaboost", [], items, ev, 1)
                elif m == "hand_topology":
                    cache[m] = infer("adaboost", [hand_topology()], items, ev, n_rep)
                elif m == "best_of_n_matched":
                    n = max(1, rnd(get("ours")["tokens_per_item"] / max(1.0, get("single_writer")["tokens_per_item"])))
                    cache[m] = best_of_n(items, ev, n, seed=int(cfg.seed))
                elif m == "writer_once":  # writer role (ledger prompt) on y0
                    cache[m] = infer("adaboost", [writer_role()], items, ev, n_rep)
                elif m == "best_of_n_writer":
                    y0_tok = get("single_writer")["tokens_per_item"]
                    w_tok = get("writer_once")["tokens_per_item"] - y0_tok
                    n = max(1, rnd((get("ours")["tokens_per_item"] - y0_tok) / max(1.0, w_tok)))
                    cache[m] = best_of_n_subgraph(writer_role(), items, ev, n)
                elif m == "repair_loop":
                    cache[m] = repair_loop(items, ev, get("ours")["tokens_per_item"], n_rep)
                else:
                    raise ValueError(m)
            return cache[m]

        for m in ("ours",) + ALL_BASELINES:
            if want(m):
                sec[m] = get(m)
        for m, v in sec.items():
            for r in v.pop("rows"):
                rows_out.append({"split": name, "method": m, "s0": ev.s0[r["item_id"]], **r})
        report["splits"].setdefault(name, {}).update(sec)
        print(f"[{run_id}] {name}: " + ", ".join(f"{k}={v['mean_g']:.3f}" for k, v in sec.items()), flush=True)
    report["total_budget"] = llm.budget.snapshot()
    log.write(f"final{tag}.json", report)
    if log.dir:  # one line per (split, method, item, repeat)
        from .llm import atomic_write_text

        atomic_write_text(log.dir / f"per_item{tag}.jsonl", "".join(json.dumps(r) + "\n" for r in rows_out))
    return fr


if __name__ == "__main__":
    main()
