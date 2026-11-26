"""Reference solver (greedy search) — used ONLY by the scorer to normalise scores.

workflow/tools.py must never import this module (§3.3, enforced by tests/test_tools.py).
"""
from __future__ import annotations

from .rules import parse_problem, simulate


def greedy_plan(problem) -> list:
    """Each day: harvest everything mature, then plant the best-daily-profit crop that can
    still mature before the horizon, as many plots as cash and land allow."""
    P = parse_problem(problem)
    D, cash, L, crops = P["days"], P["cash"], P["plots"], P["crops"]
    fields = []  # [crop, day, n]
    plan = []
    for day in range(1, D + 1):
        for f in list(fields):
            if f[1] + crops[f[0]]["grow"] <= day:
                plan.append({"day": day, "action": "harvest", "crop": f[0], "plots": f[2]})
                cash += crops[f[0]]["price"] * f[2]
                fields.remove(f)
        idle = L - sum(f[2] for f in fields)
        ranked = sorted(crops, key=lambda k: -(crops[k]["price"] - crops[k]["cost"]) / crops[k]["grow"])
        for name in ranked:
            c = crops[name]
            if day + c["grow"] > D or c["price"] <= c["cost"] or idle <= 0:
                continue
            n = min(idle, cash // c["cost"])
            if n <= 0:
                continue
            plan.append({"day": day, "action": "plant", "crop": name, "plots": int(n)})
            cash -= c["cost"] * n
            idle -= n
            fields.append([name, day, int(n)])
    return plan


def reference_profit(problem) -> float:
    r = simulate(problem, greedy_plan(problem))
    assert r.legal, r.error_msg
    return r.profit
