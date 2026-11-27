"""A deterministic fake model for the toy Farm game (tests and fake dry runs only).

It is an inner LLM for boosttopo.llm.CachedLLM: `call(req) -> (text, tokens)`. It reads the structured request
(role, item payload, current plan, diagnosis, upstream roles) and behaves like a noisy planner: drafts contain
injected mistakes, repairs fix the first reported error most of the time, improvements fill idle land, more
reliably when a simulation section or analyst notes are in the prompt. Outputs depend only on (prompt, rep).
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from .adapter import plan_list
from .rules import simulate


def _sort(plan: list) -> list:
    return sorted(plan, key=lambda a: (a["day"], 0 if a["action"] == "harvest" else 1))


def _fence(plan: list) -> str:
    return "Ledger: (fake)\n```json\n" + json.dumps(_sort(plan)) + "\n```"


class FakeFarmLLM:
    name = "fake-farm-v2"
    temperature = 0.7

    def __init__(self, err_rate: float = 0.12, think_tokens: tuple = (1500, 5000)):
        self.err_rate, self.think_tokens = err_rate, think_tokens
        self.n_calls = 0

    def call(self, req) -> tuple[str, int]:
        text, tokens, _ = self.call_meta(req)
        return text, tokens

    def call_meta(self, req):
        self.n_calls += 1
        rng = random.Random(int(hashlib.sha256(f"{req.rep}|{req.prompt}".encode()).hexdigest()[:12], 16))
        P = req.item.payload["toy"]
        role = req.role
        up = {r for r, _ in req.upstream}
        informed = "analyst" in up or "=== CROP STATS" in req.prompt
        cur = _load(req.y_cur)
        if role == "y0":
            out = _fence(self._draft(P, rng, False))
        elif role in ("analyst", "critic"):
            crops = P["crops"]
            best = max(crops, key=lambda k: (crops[k]["price"] - crops[k]["cost"]) / crops[k]["grow"])
            out = f"Prefer {best}; harvest on maturity and replant immediately; keep cash for seeds."
        elif role == "repairer":
            out = _fence(self._repair(P, cur, req.diag or "", rng) if not simulate(P, cur).legal or not cur
                         else self._improve(P, cur, rng, False))
        elif role == "improve":
            out = _fence(self._improve(P, cur, rng, "=== SIMULATION" in req.prompt))
        else:  # writer: a new plan, starting from the current one when it is legal
            out = _fence(self._improve(P, cur, rng, informed) if cur and simulate(P, cur).legal
                         else self._draft(P, rng, informed))
        ct = rng.randint(*self.think_tokens)
        pt = len(req.prompt) // 4
        return out, pt + ct, {"prompt_tokens": pt, "completion_tokens": ct}

    # ------------------------------------------------------------------ skills
    def _draft(self, P, rng, informed: bool) -> list:
        D, cash, L, crops = P["days"], P["cash"], P["plots"], P["crops"]
        by_rate = sorted(crops, key=lambda k: -(crops[k]["price"] - crops[k]["cost"]) / crops[k]["grow"])
        fields, plan = [], []
        p_best = 0.85 if informed else 0.35
        for day in range(1, D + 1):
            for f in list(fields):
                g = crops[f[0]]["grow"]
                if f[1] + g == day + 1 and rng.random() < self.err_rate * 0.5:
                    plan.append({"day": day, "action": "harvest", "crop": f[0], "plots": f[2]})  # too early
                    fields.remove(f)
                    cash += crops[f[0]]["price"] * f[2]
                elif f[1] + g <= day and rng.random() > 0.25:
                    plan.append({"day": day, "action": "harvest", "crop": f[0], "plots": f[2]})
                    fields.remove(f)
                    cash += crops[f[0]]["price"] * f[2]
            idle = L - sum(f[2] for f in fields)
            if idle <= 0 or rng.random() > 0.6:
                continue
            name = by_rate[0] if rng.random() < p_best else rng.choice(list(crops))
            c = crops[name]
            if day + c["grow"] > D and rng.random() > 0.1:
                continue
            n = min(idle, cash // c["cost"])
            if rng.random() < self.err_rate:
                n += 1  # overspend / overplant
            if n <= 0:
                continue
            plan.append({"day": day, "action": "plant", "crop": name, "plots": int(n)})
            cash -= c["cost"] * n
            fields.append([name, day, int(n)])
        return plan

    def _repair(self, P, plan, diag, rng) -> list:
        plan = _sort(plan)
        if not plan:
            return self._draft(P, rng, False)
        idx = _first_bad(P, plan)
        if idx is None or rng.random() < 0.15:
            return plan  # nothing found / failed to fix
        a = dict(plan[idx])
        crops = P["crops"]
        kind = ("funds" if "costs" in diag else "farmland" if "idle plots" in diag
                else "not_mature" if "mature" in diag else "other")
        if rng.random() < 0.2:
            del plan[idx]  # crude fix: drop the action (may cascade)
            return plan
        if kind == "not_mature":
            plants = [x for x in plan if x["action"] == "plant" and x["crop"] == a["crop"] and x["day"] <= a["day"]]
            if plants:
                a["day"] = min(P["days"], plants[0]["day"] + crops[a["crop"]]["grow"])
            plan[idx] = a
        elif kind in ("funds", "farmland") and a["action"] == "plant":
            st = simulate(P, plan[: idx + 1]).error_state or {}
            c = crops[a["crop"]]
            n = min(a["plots"], st.get("idle", 0), st.get("cash", 0) // c["cost"])
            # keep the matching harvest consistent with the new plot count
            for j in range(idx + 1, len(plan)):
                h = plan[j]
                if h["action"] == "harvest" and h["crop"] == a["crop"] and h["day"] >= a["day"] + c["grow"]:
                    h = dict(h)
                    h["plots"] -= a["plots"] - n
                    plan[j] = h
                    break
            plan[idx] = dict(a, plots=n)
            plan = [x for x in plan if x["plots"] > 0]
        else:
            del plan[idx]
        return plan

    def _improve(self, P, plan, rng, informed: bool) -> list:
        plan = _sort(plan)
        base = simulate(P, plan)
        if not base.legal:
            return self._repair(P, plan, f"[{base.error}]", rng)
        crops, D = P["crops"], P["days"]
        careful = 0.8 if informed else 0.3
        tries = 6 if informed else 3
        best, best_profit = plan, base.profit
        by_rate = sorted(crops, key=lambda k: -(crops[k]["price"] - crops[k]["cost"]) / crops[k]["grow"])
        for _ in range(tries):
            day = rng.randint(1, D - 1)
            name = by_rate[0] if (informed and rng.random() < 0.7) else rng.choice(list(crops))
            g = crops[name]["grow"]
            if day + g > D:
                continue
            n = rng.randint(1, 2)
            cand = _sort(best + [{"day": day, "action": "plant", "crop": name, "plots": n},
                                 {"day": day + g, "action": "harvest", "crop": name, "plots": n}])
            r = simulate(P, cand)
            if rng.random() < careful:
                if r.legal and r.profit > best_profit:
                    best, best_profit = cand, r.profit
            else:
                best = cand  # careless edit, may break legality
                if not r.legal:
                    break
                best_profit = r.profit
        return best


def _load(s: str) -> list:
    try:
        return [x for x in plan_list(s or "") if isinstance(x, dict) and isinstance(x.get("day"), int)]
    except Exception:  # noqa: BLE001
        return []


def _first_bad(P, plan) -> int | None:
    for i in range(len(plan)):
        if not simulate(P, plan[: i + 1]).legal:
            return i
    return None
