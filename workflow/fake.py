"""A deterministic fake model for the toy Farm task (tests, dry runs, toy boosting experiments).

It reads the skill name and slots from the rendered prompt and behaves like a noisy planner:
drafts contain injected mistakes, repair fixes the first reported error most of the time,
improve fills idle land — more reliably when simulation feedback is in the prompt, and drafts
pick better crops when crop_stats is in the prompt. Outputs depend only on (prompt, seed).
"""
from __future__ import annotations

import json
import random
import re

from farm.rules import simulate

from .llm import LLM, Usage


def _slot(prompt: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n(.*?)\n?</{tag}>", prompt, re.S)
    return m.group(1).strip() if m else ""


def _sort(plan: list) -> list:
    return sorted(plan, key=lambda a: (a["day"], 0 if a["action"] == "harvest" else 1))


def _fence(plan: list) -> str:
    return "```json\n" + json.dumps(_sort(plan)) + "\n```"


class FakeFarmLLM(LLM):
    model = "fake-farm-v1"

    def __init__(self, err_rate: float = 0.12, think_tokens: tuple = (1500, 5000)):
        self.err_rate, self.think_tokens = err_rate, think_tokens
        self.calls = 0

    def generate(self, prompt: str, seed: int) -> tuple[str, Usage]:
        self.calls += 1
        rng = random.Random(seed)
        skill = re.search(r"### SKILL: (\w+)", prompt).group(1)
        P = json.loads(_slot(prompt, "problem"))
        if skill == "draft":
            out = _fence(self._draft(P, rng, bool(_slot(prompt, "crop_stats"))))
        elif skill == "repair":
            out = _fence(self._repair(P, _load(_slot(prompt, "plan")), _slot(prompt, "diagnosis"), rng))
        elif skill == "improve":
            out = _fence(self._improve(P, _load(_slot(prompt, "plan")), rng, bool(_slot(prompt, "simulation"))))
        else:
            best = max(P["crops"], key=lambda k: (P["crops"][k]["price"] - P["crops"][k]["cost"]) / P["crops"][k]["grow"])
            out = f"Prefer {best}; harvest on maturity and replant immediately; keep cash for seeds."
        ct = rng.randint(*self.think_tokens)
        return f"<think>{'.' * 20}</think>{out}", Usage(len(prompt) // 4, ct)

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
        kind = re.search(r"\[(\w+)\]", diag)
        kind = kind.group(1) if kind else "other"
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
        a, b = s.find("["), s.rfind("]")
        v = json.loads(s[a:b + 1])
        return [x for x in v if isinstance(x, dict) and isinstance(x.get("day"), int)]
    except Exception:  # noqa: BLE001
        return []


def _first_bad(P, plan) -> int | None:
    for i in range(len(plan)):
        if not simulate(P, plan[: i + 1]).legal:
            return i
    return None
