"""Deterministic tools and checks (doc §3.2-3.4, revision §12).

Boundary (§3.3, revised): tools compute facts about the problem or about one existing plan; they never generate
or search for plans, and no tool output contains the reference profit or the relative score. The real
benchmark's simulator computes the reference profit internally (farm_planner.simulate_plan), so the boundary is
enforced on the outputs (tests/test_tools.py), not on imports.

diag_mode (ctx.diag_mode):
- "clean" (default): validate returns the benchmark diagnosis (FarmAdapter.score(...).diag) with the
  "(reference greedy profit=...)" part removed; the failing-day state is kept.
- "original": byte-identical to FarmAdapter.score(...).diag. Used only for cache replay of the multi-agent
  baselines, never in a new comparison.
"""
from __future__ import annotations

import re

MAX_CHARS = 400  # simulate / crop_stats outputs are kept under ~400 characters (§3.4)
_REF = re.compile(r" ?\(reference greedy profit=[^)]*\)")


def _clip(s: str, n: int = MAX_CHARS) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def clean_diag(diag: str) -> str:
    return _REF.sub("", diag or "")


# --------------------------------------------------------------------------- tools


def validate(st, ctx) -> str:
    """The benchmark's verdict on the current plan (legal or the first error plus the state on that day)."""
    d = ctx.bench.score(ctx.item, st.plan or "").diag
    return d if getattr(ctx, "diag_mode", "clean") == "original" else clean_diag(d)


def _ranges(days: list[int]) -> str:
    if not days:
        return "none"
    out, a, b = [], days[0], days[0]
    for x in days[1:]:
        if x == b + 1:
            b = x
        else:
            out.append(f"{a}-{b}" if a != b else str(a))
            a = b = x
    out.append(f"{a}-{b}" if a != b else str(a))
    return ",".join(out)


def simulate(st, ctx) -> str:
    """Profit, final cash, idle land by day, waste, and the plantings of a legal plan (compressed)."""
    sr = ctx.bench.score(ctx.item, st.plan or "")
    if not sr.valid:
        return _clip("simulation stopped: " + clean_diag(sr.diag).split("\n")[0])
    if not hasattr(ctx.bench, "simulate") or "crop_catalog" not in ctx.item.payload:
        return "simulation unavailable for this benchmark"
    sim = ctx.bench.simulate(ctx.item, st.plan)
    t = ctx.item.payload
    cat, D, L = t["crop_catalog"], t["duration_days"], t["farmland_count"]
    occ = [0] * (D + 2)
    plantings = []
    for ev in sim["daily_log"]:
        for p in ev.get("planted") or []:
            end = min(ev["day"] + cat[p["crop"]]["maturity_days"], D + 1)
            for d in range(ev["day"], end):
                occ[d] += p["quantity"]
            plantings.append(f"d{ev['day']} {p['quantity']}x{p['crop']}")
    free_days = [d for d in range(1, D + 1) if occ[d] < L]
    idle = sum(max(0, L - occ[d]) for d in range(1, D + 1))
    unsold = sum((sim.get("unsold_fruits") or {}).values())
    left = sum((sim.get("remaining_seeds") or {}).values())
    growing = sum(b["quantity"] for b in sim.get("growing_crops") or [])
    line = (f"profit={sim['profit']}; final cash={sim.get('final_money')}; idle plot-days={idle}/{L * D}; "
            f"days with free plots: {_ranges(free_days)}; unsold fruit={unsold}; leftover seeds={left}; "
            f"not harvested by the end={growing}; plantings: {' '.join(plantings)}")
    return _clip(line)


_PRICE_KEYS = ("fruit_price", "sell_price", "price")  # [inferred] for the real catalog; toy uses fruit_price
_YIELD_KEYS = ("yield", "fruit_yield", "harvest_yield")


def _first(d: dict, keys):
    return next((d[k] for k in keys if isinstance(d.get(k), (int, float))), None)


def crop_stats(st, ctx) -> str:
    """Per crop: days to mature, seed price, profit per plot per cycle and per day (when the catalog has
    price and yield), latest planting day that still matures."""
    t = ctx.item.payload
    if "crop_catalog" not in t:
        return "crop stats unavailable for this benchmark"
    cat, D = t["crop_catalog"], t["duration_days"]
    rows = []
    for name in t.get("available_crops") or sorted(cat):
        c = cat[name]
        m, seed = c["maturity_days"], c["seed_price"]
        price, yld = _first(c, _PRICE_KEYS), _first(c, _YIELD_KEYS)
        if price is not None and yld is not None:
            margin = yld * price - seed
            rows.append((-margin / m, f"{name}: grow={m}d, profit/plot/cycle={margin:g}, per-day={margin / m:.2f}, "
                                      f"last plant day={D - m}"))
        else:
            rows.append((0.0, f"{name}: grow={m}d, seed={seed}, last plant day={D - m}"))
    return _clip("; ".join(r for _, r in sorted(rows, key=lambda x: x[0])))


TOOLS = {"validate": validate, "simulate": simulate, "crop_stats": crop_stats}


# --------------------------------------------------------------------------- checks
# Both read the ScoreResult.category of the current plan, which the executor stores when the plan is written.
# category is computed by simulation only (FarmAdapter: idle rate and waste), never from the reference.

STATUS_OF = {"valid_ok": "legal_high", "valid_idle": "legal_low", "valid_waste": "legal_low"}

# FarmAdapter's 9 error categories (+ "empty") -> error_type labels (revision §12; see the notes)
ERROR_TYPE_OF = {
    "insufficient_funds": "funds",
    "farmland_exceeded": "farmland",
    "not_mature": "not_mature",
    "missing_seeds": "inventory",
    "missing_fruit": "inventory",
    "bad_day": "other",
    "unknown_crop": "other",
    "schema_error": "other",
    "parse_error": "other",
    "empty": "other",
}


def status(st, ctx=None) -> str:
    if not st.plan_legal:
        return "illegal"
    return STATUS_OF.get(st.plan_cat, "legal_low")


def error_type(st, ctx=None) -> str:
    if st.plan_legal:
        return "none"
    return ERROR_TYPE_OF.get(st.plan_cat, "other")


CHECKS = {"status": status, "error_type": error_type}
