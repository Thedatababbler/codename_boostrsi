"""Deterministic tools and checks (doc §3.2–3.4).

Boundary (§3.3): tools compute facts about the problem, or about one existing plan. They never
generate or search for plans and never touch the reference solution. This module may import
only the game *rules* — tests/test_tools.py asserts it does not import farm.reference.
"""
from __future__ import annotations

from farm.rules import parse_problem, simulate as _sim

MAX_CHARS = 400  # each rendered tool output is kept under ~400 characters (§3.4)


def _clip(s: str, n: int = MAX_CHARS) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------- tools


def validate(st, ctx=None) -> str:
    """Legal or not; if not, the first error and the state on that day."""
    r = _sim(st.problem, st.plan)
    if r.legal:
        return "LEGAL"
    s = r.error_state or {}
    planted = ", ".join(f"{p['plots']}x{p['crop']}@d{p['day']}" for p in s.get("planted", [])) or "none"
    tail = f" | state: cash={s.get('cash')}, idle={s.get('idle')}, planted=[{planted}]" if s else ""
    return _clip(f"ILLEGAL[{r.error}] {r.error_msg}{tail}")


def simulate(st, ctx=None) -> str:
    """Per-day cash / idle plots / harvests (compressed), final profit, most idle days."""
    r = _sim(st.problem, st.plan)
    if not r.legal:
        return _clip(f"simulation stopped: ILLEGAL[{r.error}] {r.error_msg}")
    P = parse_problem(st.problem)
    d = r.daily
    idle_days = [x["day"] for x in d if x["idle"] > 0]
    idle_plot_days = sum(x["idle"] for x in d)
    rich = sorted(d, key=lambda x: (-x["cash"] * (x["idle"] > 0), x["day"]))[:3]
    harv = [f"d{x['day']}+{int(x['harvest'])}" for x in d if x["harvest"]]
    line = (f"profit={int(r.profit)}; idle plot-days={idle_plot_days}/{P['plots'] * P['days']}; "
            f"days with idle land: {_ranges(idle_days)}; "
            f"idle cash+land: " + ", ".join(f"d{x['day']}(cash={x['cash']},idle={x['idle']})" for x in rich if x['idle'])
            + f"; harvests: {' '.join(harv[:8])}")
    return _clip(line)


def crop_stats(st, ctx=None) -> str:
    """Per crop: days to mature, profit per plot per cycle, daily profit, latest planting day."""
    P = parse_problem(st.problem)
    rows = []
    for name, c in sorted(P["crops"].items(),
                          key=lambda kv: -(kv[1]["price"] - kv[1]["cost"]) / kv[1]["grow"]):
        m = c["price"] - c["cost"]
        rows.append(f"{name}: grow={c['grow']}d, profit/plot/cycle={m}, "
                    f"per-day={m / c['grow']:.2f}, last plant day={P['days'] - c['grow']}")
    return _clip("; ".join(rows))


def reset_to_best(st, ctx=None) -> str | None:
    return st.best_plan


TOOLS = {"validate": validate, "simulate": simulate, "crop_stats": crop_stats,
         "reset_to_best": reset_to_best}


# --------------------------------------------------------------------------- checks


def status(st, ctx=None) -> str:
    """illegal / legal_low / legal_high by whether the current score reaches the threshold.

    mode "score" (doc default) uses the executor's score of the current plan.
    mode "bound" uses only inference-time info: simulated profit / a naive profit upper bound.
    """
    cfg = getattr(ctx, "check_cfg", None) or {}
    thr = cfg.get("status_threshold", 0.8)
    if not st.plan_legal:
        return "illegal"
    if cfg.get("status_mode", "score") == "bound":
        r = _sim(st.problem, st.plan)
        val = r.profit / max(_profit_bound(st.problem), 1e-9)
    else:
        val = st.plan_score
    return "legal_high" if val >= thr else "legal_low"


def _profit_bound(problem) -> float:
    P = parse_problem(problem)
    best = max(((c["price"] - c["cost"]) / c["grow"] for c in P["crops"].values()), default=0)
    return max(best, 0) * P["plots"] * P["days"]


def error_type(st, ctx=None) -> str:
    d = st.diag or ""
    for lab in ("funds", "not_mature", "farmland"):
        if f"[{lab}]" in d:
            return lab
    return "other"


CHECKS = {"status": status, "error_type": error_type}


def _ranges(days: list[int]) -> str:
    if not days:
        return "none"
    out, a, b = [], days[0], days[0]
    for x in days[1:]:
        if x == b + 1:
            b = x
        else:
            out.append(f"{a}-{b}" if a != b else str(a)); a = b = x
    out.append(f"{a}-{b}" if a != b else str(a))
    return ",".join(out)
