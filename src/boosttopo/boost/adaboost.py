"""AdaBoost mode: every subgraph runs independently on y0; output = best member; weights on hard items
(doc section 7)."""

from __future__ import annotations

from statistics import mean

from ..config import sub_seed
from ..graph import Subgraph
from ..search.grow import grow_subgraph
from .common import Ctx, FitResult, lower_ci, pick_candidate
from .library import pick_seed, select_ensemble
from .residual import residual_report, residual_weights


def fit_ada(res: dict, items, g_ens: dict, r: dict | None) -> float:
    """sum_i w_i [max(g_ens_i, g_i(h)) - g_ens_i] over the multiset ``items``; w = r renormalised over S
    (reweighting), or uniform when ``r`` is None (resampling)."""
    ws = [(r[it.item_id] if r is not None else 1.0) for it in items]
    z = sum(ws)
    if z <= 0:
        return 0.0
    return sum(w * (max(g_ens[it.item_id], res[it.item_id].g) - g_ens[it.item_id])
               for w, it in zip(ws, items)) / z


def ensemble_g(ev, ensemble, items, y0) -> dict[str, float]:
    g = {it.item_id: 0.0 for it in items}
    for i_, h in enumerate(ensemble):
        res = ev.run(h, items, y0, occ=ev.occurrence(h, ensemble[:i_]))
        g = {i: max(g[i], res[i].g) for i in g}
    return g


def fit_adaboost(train, heldout, ctx: Ctx, resume: bool = True) -> FitResult:
    cfg, ev, log = ctx.cfg, ctx.ev, ctx.log
    resample = cfg.sampler.kind == "weighted_bootstrap"  # never both reweight and resample
    y0_tr = {it.item_id: ev.y0[it.item_id] for it in train}
    y0_ho = {it.item_id: ev.y0[it.item_id] for it in heldout}
    ensemble: list[Subgraph] = []
    g_tr = {it.item_id: 0.0 for it in train}
    g_ho = {it.item_id: 0.0 for it in heldout}
    stall, start_round, history = 0, 0, []
    n_rejected = 0
    ck = log.read("checkpoint.json") if resume else None
    if ck and ck.get("mode") == "adaboost":
        ensemble = [Subgraph.from_dict(d) for d in ck["ensemble"]]
        g_tr, g_ho, stall, start_round = ck["g_tr"], ck["g_ho"], ck["stall"], ck["round"] + 1
        history = ck.get("history", [])
        n_rejected = ck.get("n_rejected", 0)
        ev.rep_base = n_rejected * 1000
        if stall >= cfg.patience:  # the checkpointed run had already stopped
            start_round = cfg.rounds
        import json as _json
        from pathlib import Path as _P
        if log.dir is not None:
            graphs = {}
            for line in (_P(log.dir) / "library.jsonl").read_text().splitlines() if (_P(log.dir) / "library.jsonl").exists() else []:
                r_ = _json.loads(line)
                if r_.get("graph"):
                    graphs[r_["key"]] = Subgraph.from_dict(r_["graph"])
                elif r_.get("subgraph"):
                    hh = Subgraph.from_describe(r_["subgraph"])
                    if hh.key() == r_["key"]:
                        graphs[r_["key"]] = hh
            ctx.library.load_jsonl(_P(log.dir) / "library.jsonl", graphs.get)
        for i_, h in enumerate(ensemble):  # rebuild library entries of members from the cache
            ctx.library.add(h, mode="ada", per_item=ev.run(h, train, y0_tr, occ=ev.occurrence(h, ensemble[:i_])),
                            source="resume")

    for m in range(start_round, cfg.rounds):
        if len(ensemble) >= cfg.max_members:
            break
        r = residual_weights(g_tr, cfg)
        # categories of the ensemble's current best solution per item (y0 when no member helps)
        cats = {it.item_id: ev.bench.score(it, y0_tr[it.item_id]).category for it in train}
        for i_, h in enumerate(ensemble):
            res = ev.run(h, train, y0_tr, occ=ev.occurrence(h, ensemble[:i_]))
            for i in cats:
                if res[i].g >= g_tr[i] and res[i].g > 0:
                    cats[i] = res[i].category
        report = residual_report(g_tr, r, cats)
        cands, branch_logs = [], []
        ens_keys = {h.key() for h in ensemble}
        for k in range(cfg.forest.k):
            bag, oob = ctx.sampler.split(train, weights=r, seed=sub_seed(cfg.seed, m, k))
            seed_h = pick_seed(ctx.library, r, g_tr, cfg, exclude=ens_keys)
            r_fit = None if resample else r

            def fit_fn(h, items, inputs, _r=r_fit, _tag=f"r{m}b{k}"):
                res = ev.run(h, items, inputs, occ=ev.occurrence(h, ensemble))
                pen = ev.cost_penalty(res)
                f = fit_ada(res, items, g_tr, _r) - pen
                ctx.library.add(h, mode="ada", per_item=res, fit=f, source=_tag)
                return f, pen

            gr = grow_subgraph(seed_h, bag, y0_tr, fit_fn, ctx.proposer, ctx.policy, cfg, limits=ctx.limits,
                               residual=report, stats=ctx.stats, rnd=m,
                               search_seed=sub_seed(cfg.seed, m, k, "search"), log=log, tag=f"r{m}b{k}")
            eval_items = oob or bag
            h_fit, _ = fit_fn(gr.h, eval_items, y0_tr)
            res = ev.run(gr.h, eval_items, y0_tr, occ=ev.occurrence(gr.h, ensemble))
            gains = {it.item_id: max(g_tr[it.item_id], res[it.item_id].g) - g_tr[it.item_id] for it in eval_items}
            cands.append((h_fit, gr.h, gains))
            branch_logs.append({"branch": k, "seed": seed_h.describe(), "subgraph": gr.h.describe(),
                                "bag_fit": gr.fit, "oob_fit": h_fit, "n_bag": len(bag), "n_oob": len(oob),
                                "kept": [str(e) for e in gr.kept], "n_evals": gr.n_evals,
                                "solo_mean_g": mean(res[it.item_id].g for it in eval_items)})
        _, best = pick_candidate(cands, g_tr, cfg.search.min_gain)

        occ_best = ev.occurrence(best, ensemble)
        res_ho = ev.run(best, heldout, y0_ho, occ=occ_best)
        new_g_ho = {i: max(g_ho[i], res_ho[i].g) for i in g_ho}
        gain = [new_g_ho[i] - g_ho[i] for i in sorted(g_ho)]
        lci = lower_ci(gain, cfg, seed=sub_seed(cfg.seed, m, "ci"))
        accepted = lci > cfg.accept_margin
        # was the chosen candidate not the best by solo mean g? (milestone M2 check)
        solo = {b["subgraph"]: b["solo_mean_g"] for b in branch_logs}
        rec = {"round": m, "mode": "adaboost", "residual": report, "weights_max": max(r.values()),
               "candidates": branch_logs, "chosen": best.describe(), "ho_gain_mean": mean(gain),
               "ho_gain_lci": lci, "accepted": accepted, "budget": ev.llm.budget.snapshot()}
        if accepted:
            stall = 0
            ensemble.append(best)
            res_tr = ev.run(best, train, y0_tr, occ=occ_best)  # complete its results on all training items
            ctx.library.add(best, mode="ada", per_item=res_tr, source=f"r{m}accepted")
            g_tr = {i: max(g_tr[i], res_tr[i].g) for i in g_tr}
            if cfg.get("fresh_heldout_eval", False):
                res_fr = ev.run_fresh(best, heldout, y0_ho, occ=occ_best)
                fresh_g = {i: max(g_ho[i], res_fr[i].g) for i in g_ho}
                rec["ho_gain_mean_fresh"] = mean(fresh_g[i] - g_ho[i] for i in sorted(g_ho))
                g_ho = fresh_g
            else:
                g_ho = new_g_ho
            rec["chosen_solo_train_g"] = mean(res_tr[i].g for i in g_tr)
        else:
            stall += 1
            n_rejected += 1
            ev.rep_base = n_rejected * 1000  # "retry with a new random seed": fresh LLM samples next round
        rec["train_g"] = mean(g_tr.values())
        rec["heldout_g"] = mean(g_ho.values())
        rec["ensemble"] = [h.describe() for h in ensemble]
        rec["solo_best_candidate"] = max(solo, key=solo.get) if solo else None
        history.append({k: rec[k] for k in ("round", "accepted", "ho_gain_mean", "ho_gain_lci", "train_g",
                                            "heldout_g", "chosen", "budget")})
        log.append("rounds.jsonl", rec)
        log.append("budget.jsonl", {"round": m, **ev.llm.budget.snapshot()})
        log.write("checkpoint.json", {"mode": "adaboost", "round": m, "ensemble": [h.to_dict() for h in ensemble],
                                      "g_tr": g_tr, "g_ho": g_ho, "stall": stall, "history": history,
                                      "n_rejected": n_rejected})
        if not accepted and stall >= cfg.patience:
            break

    ev.rep_base = 0
    extra = offline_selection(train, heldout, ensemble, ctx)
    if extra.get("replaced"):
        ensemble = [Subgraph.from_dict(d) for d in extra["offline_ensemble"]]
    return FitResult("adaboost", ensemble, history, extra)


def offline_selection(train, heldout, ensemble, ctx: Ctx) -> dict:
    """Complete the top library entries on all training items, select greedily without LLM calls, then verify
    once on heldout; replace the round-built ensemble only if strictly better there."""
    cfg, ev = ctx.cfg, ctx.ev
    y0_tr = {it.item_id: ev.y0[it.item_id] for it in train}
    y0_ho = {it.item_id: ev.y0[it.item_id] for it in heldout}
    ids = [it.item_id for it in train]
    top = sorted(ctx.library.entries("ada"), key=lambda e: -e.fit)[: cfg.library.complete_top]
    for e in top:
        if not e.covers(ids):
            ctx.library.add(e.subgraph, mode="ada", per_item=ev.run(e.subgraph, train, y0_tr), source="complete")
    chosen = select_ensemble(ctx.library, ids, cfg)
    if not chosen:
        return {"offline_ensemble": [], "replaced": False}
    g_off = mean(ensemble_g(ev, chosen, heldout, y0_ho).values())
    g_cur = mean(ensemble_g(ev, ensemble, heldout, y0_ho).values()) if ensemble else 0.0
    out = {"offline_ensemble": [h.to_dict() for h in chosen], "offline_desc": [h.describe() for h in chosen],
           "offline_heldout_g": g_off, "round_heldout_g": g_cur, "replaced": g_off > g_cur}
    ctx.log.append("offline_selection.jsonl", out)
    return out
