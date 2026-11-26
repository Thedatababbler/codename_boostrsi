"""GBDT mode: serial stages; each new subgraph refines the previous stage's solutions (doc section 6)."""

from __future__ import annotations

from statistics import mean

from ..config import sub_seed
from ..graph import Subgraph, single_writer
from ..search.grow import grow_subgraph
from .common import Ctx, FitResult, lower_ci, pick_candidate
from .residual import residual_report, residual_weights


def fit_gbdt(train, heldout, ctx: Ctx, resume: bool = True) -> FitResult:
    cfg, ev, log = ctx.cfg, ctx.ev, ctx.log

    def g_of(item, y):
        return ev.g(item.item_id, ev.score_of(item, y))

    def fit_fn(h: Subgraph, items, inputs):
        # a candidate is fitted at the position it would take: its occurrence among the current members
        res = ev.run(h, items, inputs, occ=ev.occurrence(h, ensemble))
        gains = [res[it.item_id].g - g_of(it, inputs[it.item_id]) for it in items]
        pen = ev.cost_penalty(res)
        return mean(gains) - pen, pen

    ensemble: list[Subgraph] = []
    cur_tr = {it.item_id: ev.y0[it.item_id] for it in train}
    cur_ho = {it.item_id: ev.y0[it.item_id] for it in heldout}
    stall, start_round, history = 0, 0, []
    n_rejected = 0
    ck = log.read("checkpoint.json") if resume else None
    if ck and ck.get("mode") == "gbdt":
        ensemble = [Subgraph.from_dict(d) for d in ck["ensemble"]]
        cur_tr, cur_ho, stall, start_round = ck["cur_tr"], ck["cur_ho"], ck["stall"], ck["round"] + 1
        history = ck.get("history", [])
        n_rejected = ck.get("n_rejected", 0)
        ev.rep_base = n_rejected * 1000
        if stall >= cfg.patience:  # the checkpointed run had already stopped
            start_round = cfg.rounds

    for m in range(start_round, cfg.rounds):
        if len(ensemble) >= cfg.max_members:
            break
        g_cur = {it.item_id: g_of(it, cur_tr[it.item_id]) for it in train}
        active = [it for it in train if g_cur[it.item_id] < 1.0]
        if not active:
            break
        # weights are reporting-only in GBDT (residual = the input solutions themselves)
        w = residual_weights({it.item_id: g_cur[it.item_id] for it in active}, cfg)
        cats = {it.item_id: ev.bench.score(it, cur_tr[it.item_id]).category for it in active}
        report = residual_report({i: g_cur[i] for i in w}, w, cats)
        cands, branch_logs = [], []
        for k in range(cfg.forest.k):
            bag, oob = ctx.sampler.split(active, weights=None, seed=sub_seed(cfg.seed, m, k))
            gr = grow_subgraph(single_writer(), bag, cur_tr, fit_fn, ctx.proposer, ctx.policy, cfg,
                               limits=ctx.limits, residual=report, stats=ctx.stats, rnd=m,
                               search_seed=sub_seed(cfg.seed, m, k, "search"), log=log, tag=f"r{m}b{k}")
            eval_items = oob or bag
            h_gain, _ = fit_fn(gr.h, eval_items, cur_tr)
            res = ev.run(gr.h, eval_items, cur_tr, occ=ev.occurrence(gr.h, ensemble))
            gains = {it.item_id: res[it.item_id].g - g_cur[it.item_id] for it in eval_items}
            cands.append((h_gain, gr.h, gains))
            ctx.library.add(gr.h, mode="gbdt", stage=m, fit=h_gain, source=f"r{m}b{k}")
            branch_logs.append({"branch": k, "subgraph": gr.h.describe(), "bag_fit": gr.fit, "oob_fit": h_gain,
                                "n_bag": len(bag), "n_oob": len(oob), "kept": [str(e) for e in gr.kept],
                                "n_evals": gr.n_evals})
        _, best = pick_candidate(cands, g_cur, cfg.search.min_gain)

        occ_best = ev.occurrence(best, ensemble)
        out_ho = ev.run(best, heldout, cur_ho, occ=occ_best)  # the only place heldout items are touched
        g_ho_old = {it.item_id: g_of(it, cur_ho[it.item_id]) for it in heldout}
        gain = [out_ho[i].g - g_ho_old[i] for i in sorted(g_ho_old)]
        lci = lower_ci(gain, cfg, seed=sub_seed(cfg.seed, m, "ci"))
        accepted = lci > cfg.accept_margin
        rec = {"round": m, "mode": "gbdt", "residual": report, "n_active": len(active), "candidates": branch_logs,
               "chosen": best.describe(), "ho_gain_mean": mean(gain), "ho_gain_lci": lci, "accepted": accepted,
               "budget": ev.llm.budget.snapshot()}
        if accepted:
            stall = 0
            ensemble.append(best)
            res_tr = ev.run(best, active, cur_tr, occ=occ_best)
            for it in active:
                cur_tr[it.item_id] = res_tr[it.item_id].solution
            if cfg.get("fresh_heldout_eval", False):
                out_fr = ev.run_fresh(best, heldout, cur_ho, occ=occ_best)
                rec["ho_gain_mean_fresh"] = mean(out_fr[i].g - g_ho_old[i] for i in sorted(g_ho_old))
                cur_ho = {i: out_fr[i].solution for i in cur_ho}
            else:
                cur_ho = {i: out_ho[i].solution for i in cur_ho}
        else:
            stall += 1
            n_rejected += 1
            ev.rep_base = n_rejected * 1000  # "retry with a new random seed": fresh LLM samples next round
        rec["train_g"] = mean(g_of(it, cur_tr[it.item_id]) for it in train)
        rec["heldout_g"] = mean(g_of(it, cur_ho[it.item_id]) for it in heldout)
        rec["ensemble"] = [h.describe() for h in ensemble]
        history.append({k: rec[k] for k in ("round", "accepted", "ho_gain_mean", "ho_gain_lci", "train_g",
                                            "heldout_g", "chosen", "budget")})
        log.append("rounds.jsonl", rec)
        log.append("budget.jsonl", {"round": m, **ev.llm.budget.snapshot()})
        log.write("checkpoint.json", {"mode": "gbdt", "round": m, "ensemble": [h.to_dict() for h in ensemble],
                                      "cur_tr": cur_tr, "cur_ho": cur_ho, "stall": stall, "history": history,
                                      "n_rejected": n_rejected})
        if not accepted and stall >= cfg.patience:
            break
    ev.rep_base = 0
    return FitResult("gbdt", ensemble, history)
