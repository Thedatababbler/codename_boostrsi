"""Inference on new items (doc sections 6.5, 7.5) and the baselines of section 11.1."""

from __future__ import annotations

from statistics import mean

from .boost.residual import bootstrap_lower
from .graph import IN, Node, Subgraph, static_scorer_calls
from .roles import role_prompt, y0_request


def writer_role() -> Subgraph:
    """[IN->writer]: the writer role (ledger prompt) reading the start solution y0."""
    return Subgraph((Node("n0", "writer"),), ((IN, "n0"),), "n0")


def hand_topology() -> Subgraph:
    """analyst -> writer -> repairer, the hand-written baseline. The writer reads IN and the analyst notes; the
    repairer reads only the writer, so it repairs (and is diagnosed on) the writer's plan, not y0."""
    return Subgraph((Node("n0", "analyst"), Node("n1", "writer"), Node("n2", "repairer")),
                    ((IN, "n0"), (IN, "n1"), ("n0", "n1"), ("n1", "n2")), "n2")


SEED_STRIDE = 1_000_000  # rep offset per run seed (Evaluator.rep_seed, best-of-N extra samples)


def stage_scorer_calls(h: Subgraph) -> int:
    """Scorer calls of a loop-free stage (kept for callers; loop stages report StageOutput.scorer_calls)."""
    return static_scorer_calls(h)


def _summary(rows: list[dict], ev, extra: dict | None = None) -> dict:
    """rows: one dict per (item, repeat) with g, score, tokens, scorer_calls, valid."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["item_id"], []).append(r)
    per_item = {iid: {k: mean(x[k] for x in xs) for k in ("g", "score", "tokens", "scorer_calls", "valid")}
                for iid, xs in by.items()}
    gs = [v["g"] for v in per_item.values()]
    lo = bootstrap_lower(gs, 0.95, 2000, 0) if gs else 0.0
    hi = 0.0 - bootstrap_lower([-x for x in gs], 0.95, 2000, 0) + 0.0 if gs else 0.0
    legal = [r["score"] for r in rows if r["valid"]]
    out = {"n": len(gs), "mean_g": mean(gs) if gs else 0.0, "g_ci95": [lo, hi],
           "mean_score": mean(v["score"] for v in per_item.values()) if gs else 0.0,
           "tokens_per_item": mean(v["tokens"] for v in per_item.values()) if gs else 0.0,
           "scorer_calls_per_item": mean(v["scorer_calls"] for v in per_item.values()) if gs else 0.0,
           "valid_rate": mean(r["valid"] for r in rows) if rows else 0.0,
           "legal_mean_score": mean(legal) if legal else None}
    out.update(extra or {})
    out["rows"] = rows
    return out


def y0_tokens(ev, item) -> int:
    _, t = getattr(ev.llm, "call_y0", ev.llm.call)(y0_request(item, 0))  # cached
    return t


def _chain(mode: str, ensemble, it, ev, r: int) -> dict:
    """One repeat of inference on one item (tokens and scorer calls include producing/scoring y0)."""
    y0 = ev.y0[it.item_id]
    tokens, calls = y0_tokens(ev, it), 1
    seen, n_stage, n_dup = set(), 0, 0

    def stage(i, h, y_in):
        """Run member i; count stage runs that exactly repeat an earlier one (same structure, input, repeat)."""
        nonlocal n_stage, n_dup
        occ = ev.occurrence(h, ensemble[:i])
        sig = (h.key(), sha_text(y_in), occ)
        n_stage += 1
        n_dup += sig in seen
        seen.add(sig)
        return ev._one(h, it, y_in, r, occ)

    if mode == "gbdt":
        y = y0
        for i, h in enumerate(ensemble):
            if ev.g(it.item_id, ev.score_of(it, y)) >= 1.0:
                break  # reached the reference: skip later stages
            so = stage(i, h, y)
            y, tokens, calls = so.solution, tokens + so.cost, calls + so.scorer_calls
        best = y
    else:
        best, best_s = y0, ev.score_of(it, y0)
        for i, h in enumerate(ensemble):
            so = stage(i, h, y0)
            tokens, calls = tokens + so.cost, calls + so.scorer_calls
            if so.score > best_s:
                best, best_s = so.solution, so.score
    sr = ev.bench.score(it, best)
    return {"item_id": it.item_id, "rep": r, "g": ev.g(it.item_id, sr.score), "score": sr.score,
            "tokens": tokens, "scorer_calls": calls, "valid": float(sr.valid), "category": sr.category,
            "stage_runs": n_stage, "dup_stage_runs": n_dup}


def sha_text(s: str) -> str:
    import hashlib

    return hashlib.sha256((s or "").encode()).hexdigest()


def infer(mode: str, ensemble: list[Subgraph], items, ev, n_repeats: int) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(it, r) for it in items for r in range(n_repeats)]
    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), ev.pool._max_workers))) as pool:
        rows = list(pool.map(lambda j: _chain(mode, ensemble, j[0], ev, j[1]), jobs))
    return _summary(rows, ev, {"mode": mode, "ensemble": [h.describe() for h in ensemble]})


def best_of_n(items, ev, n: int, seed: int = 0) -> dict:
    """Single writer (the benchmark prompt, as y0) sampled n times, the scorer picks the best.
    Sample 0 is y0 itself; samples 1..n-1 use repeat indices shifted by the run seed."""
    reps = [0] + [seed * SEED_STRIDE + r for r in range(1, n)]
    jobs = [(it, r) for it in items for r in reps]
    outs = list(ev.pool.map(lambda j: ev.llm.call(y0_request(j[0], j[1])), jobs))
    per_item: dict = {}
    for (it, r), (text, tok) in zip(jobs, outs):
        d = per_item.setdefault(it.item_id, {"best": None, "tokens": 0, "n": 0})
        sr = ev.bench.score(it, text)
        d["tokens"] += tok
        d["n"] += 1
        if d["best"] is None or sr.score > d["best"].score:
            d["best"] = sr
    rows = [{"item_id": i, "rep": 0, "g": ev.g(i, d["best"].score), "score": d["best"].score,
             "tokens": d["tokens"], "scorer_calls": d["n"], "valid": float(d["best"].valid),
             "category": d["best"].category} for i, d in per_item.items()]
    return _summary(rows, ev, {"mode": f"best_of_{n}", "n_samples": n})


def best_of_n_subgraph(h: Subgraph, items, ev, n: int) -> dict:
    """h run n times on y0 with repeat indices 0..n-1 (shifted by the run seed via ev.rep_seed); the scorer keeps
    the best of y0 and the n gated outputs. Tokens and scorer calls include y0."""
    jobs = [(it, r) for it in items for r in range(n)]
    outs = list(ev.pool.map(lambda j: ev._one(h, j[0], ev.y0[j[0].item_id], j[1]), jobs))
    per: dict = {}
    for (it, r), so in zip(jobs, outs):
        d = per.setdefault(it.item_id, {"it": it, "best": ev.y0[it.item_id], "best_s": ev.score_of(it, ev.y0[it.item_id]),
                                        "tokens": y0_tokens(ev, it), "calls": 1})
        d["tokens"] += so.cost
        d["calls"] += so.scorer_calls
        if so.score > d["best_s"]:
            d["best"], d["best_s"] = so.solution, so.score
    rows = []
    for i, d in per.items():
        sr = ev.bench.score(d["it"], d["best"])
        rows.append({"item_id": i, "rep": 0, "g": ev.g(i, sr.score), "score": sr.score, "tokens": d["tokens"],
                     "scorer_calls": d["calls"], "valid": float(sr.valid), "category": sr.category})
    return _summary(rows, ev, {"mode": f"best_of_{n}:{h.describe()}", "n_samples": n})


def _repair_loop_one(it, ev, r: int, budget: float, max_repairs: int) -> dict:
    """Writer role on y0, then the repairer repeatedly repairs the writer's own lineage (always the best-scoring
    plan of that lineage so far, newest on ties) with the scorer's diagnosis, until the token budget is used.
    The answer is the best of y0 and the lineage (the same gate every method has)."""
    rep = r + ev.rep_seed
    y0 = ev.y0[it.item_id]
    tokens, calls = y0_tokens(ev, it), 1
    s0 = ev.score_of(it, y0)
    text, t = ev.llm.call(role_prompt("writer", it, y0, [], None, rep=rep))  # = the writer_once sample
    tokens, calls = tokens + t, calls + 1
    cur, cur_sr = text, ev.bench.score(it, text)
    best, best_s = (cur, cur_sr.score) if cur_sr.score > s0 else (y0, s0)
    n_rep = 0
    while tokens < budget and n_rep < max_repairs and ev.g(it.item_id, best_s) < 1.0:
        text, t = ev.llm.call(role_prompt("repairer", it, cur, [], cur_sr.diag, rep=rep + 7919 * (n_rep + 1)))
        tokens, calls, n_rep = tokens + t, calls + 2, n_rep + 1  # diagnosis of cur + score of the new plan
        sr = ev.bench.score(it, text)
        if sr.score >= cur_sr.score:
            cur, cur_sr = text, sr
        if sr.score > best_s:
            best, best_s = text, sr.score
    sr = ev.bench.score(it, best)
    return {"item_id": it.item_id, "rep": r, "g": ev.g(it.item_id, sr.score), "score": sr.score, "tokens": tokens,
            "scorer_calls": calls, "valid": float(sr.valid), "category": sr.category, "n_repairs": n_rep}


def repair_loop(items, ev, budget: float, n_repeats: int, max_repairs: int = 40) -> dict:
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(it, r) for it in items for r in range(n_repeats)]
    with ThreadPoolExecutor(max_workers=max(1, min(len(jobs), ev.pool._max_workers))) as pool:
        rows = list(pool.map(lambda j: _repair_loop_one(j[0], ev, j[1], budget, max_repairs), jobs))
    out = _summary(rows, ev, {"mode": "repair_loop", "token_budget": budget})
    out["mean_repairs"] = mean(r["n_repairs"] for r in rows) if rows else 0.0
    return out
