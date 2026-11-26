"""Toy Farm game rules (stand-in for the real benchmark's rules; see IMPLEMENTATION_NOTES).

Problem (JSON): {"days": D, "cash": C, "plots": L,
                 "crops": {name: {"cost": c, "grow": g, "price": p}}}
Plan (JSON list), actions applied in listed order within a day, days ascending:
    {"day": d, "action": "plant",   "crop": name, "plots": n}
    {"day": d, "action": "harvest", "crop": name, "plots": n}
A plot planted on day d with grow g can be harvested on any day >= d+g (and <= D).
Harvest frees the plots and pays price*n immediately. Unharvested crops are worth nothing.
Profit = final cash - initial cash.

Errors (first one aborts the simulation): funds, not_mature, farmland, other.
This module is pure rules: it never computes or searches for a plan.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class SimResult:
    legal: bool
    profit: float
    error: str | None = None          # funds | not_mature | farmland | other
    error_msg: str | None = None
    error_day: int | None = None
    error_state: dict | None = None   # state on the error day
    daily: list = field(default_factory=list)  # [{"day", "cash", "idle", "harvest"}]


def parse_problem(problem: str | dict) -> dict:
    return json.loads(problem) if isinstance(problem, str) else problem


def parse_plan(plan: str | list | None) -> list:
    if plan is None:
        raise ValueError("no plan")
    if isinstance(plan, list):
        return plan
    s = plan.strip()
    # tolerate surrounding prose / code fences: take the outermost JSON list
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b < a:
        raise ValueError("plan is not a JSON list")
    out = json.loads(s[a:b + 1])
    if not isinstance(out, list):
        raise ValueError("plan is not a JSON list")
    return out


def simulate(problem: str | dict, plan: str | list | None) -> SimResult:
    P = parse_problem(problem)
    D, cash0, L, crops = P["days"], P["cash"], P["plots"], P["crops"]
    cash = cash0
    fields: list[list] = []   # [crop, planted_day, n]
    daily = []

    def state(day):
        return {"day": day, "cash": cash, "idle": L - sum(f[2] for f in fields),
                "planted": [{"crop": f[0], "day": f[1], "plots": f[2]} for f in fields]}

    def fail(kind, msg, day):
        return SimResult(False, 0.0, kind, msg, day, state(day), daily)

    try:
        acts = parse_plan(plan)
    except Exception as e:  # noqa: BLE001
        return SimResult(False, 0.0, "other", f"unparseable plan: {e}", None, None, [])
    by_day: dict[int, list] = {}
    for a in acts:
        if not isinstance(a, dict) or not isinstance(a.get("day"), int):
            return fail("other", f"malformed action {a!r}", None)
        if not 1 <= a["day"] <= D:
            return fail("other", f"day {a.get('day')} outside 1..{D}", None)
        by_day.setdefault(a["day"], []).append(a)

    for day in range(1, D + 1):
        harvested = 0.0
        for a in by_day.get(day, []):
            kind, crop, n = a.get("action"), a.get("crop"), a.get("plots")
            if crop not in crops or not isinstance(n, int) or n <= 0:
                return fail("other", f"day {day}: bad crop/plots in {a!r}", day)
            c = crops[crop]
            if kind == "plant":
                idle = L - sum(f[2] for f in fields)
                if n > idle:
                    return fail("farmland", f"day {day}: plant {n} {crop} but only {idle} idle plots", day)
                if c["cost"] * n > cash:
                    return fail("funds", f"day {day}: plant {n} {crop} costs {c['cost'] * n} > cash {cash}", day)
                cash -= c["cost"] * n
                fields.append([crop, day, n])
            elif kind == "harvest":
                need = n
                mature = sum(f[2] for f in fields if f[0] == crop and f[1] + c["grow"] <= day)
                planted = sum(f[2] for f in fields if f[0] == crop)
                if planted < n:
                    return fail("other", f"day {day}: harvest {n} {crop} but only {planted} planted", day)
                if mature < n:
                    return fail("not_mature", f"day {day}: harvest {n} {crop} but only {mature} mature", day)
                for f in sorted((f for f in fields if f[0] == crop), key=lambda f: f[1]):
                    if need == 0:
                        break
                    if f[1] + c["grow"] > day:
                        continue
                    take = min(need, f[2])
                    f[2] -= take
                    need -= take
                fields[:] = [f for f in fields if f[2] > 0]
                cash += c["price"] * n
                harvested += c["price"] * n
            else:
                return fail("other", f"day {day}: unknown action {kind!r}", day)
        daily.append({"day": day, "cash": cash, "idle": L - sum(f[2] for f in fields),
                      "harvest": harvested})
    return SimResult(True, float(cash - cash0), daily=daily)
