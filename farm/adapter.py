"""Toy Farm benchmark behind boosttopo's BenchAdapter interface (tests and fake dry runs only).

The real benchmark is boosttopo.bench.farm.FarmAdapter (SelfImprove-Bench). This adapter mimics the parts of it
the workflow code relies on:
- `score(item, text) -> ScoreResult(score, valid, diag, category)`, with FarmAdapter's diagnosis format, including
  the "(reference greedy profit=...)" part that the clean diagnosis mode must remove;
- `simulate(item, text) -> dict` with the keys of farm_planner.simulate_plan that FarmAdapter itself reads
  (daily_log[].day / planted[].crop, quantity; profit; final_money; unsold_fruits; remaining_seeds;
  growing_crops[].crop, quantity, harvest_day; reference_profit; relative_score);
- `item.payload` with the task keys FarmAdapter reads (crop_catalog[].maturity_days / seed_price, available_crops,
  duration_days, farmland_count, initial_money), plus `toy` (the toy rules' own problem).

The toy plan format is a JSON list of plant / harvest actions (see rules.py), not the real {"days": [...]}.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import threading

from boosttopo.bench.base import Item, ScoreResult

from .reference import reference_profit
from .rules import simulate as toy_simulate

_CAT = {"funds": "insufficient_funds", "farmland": "farmland_exceeded", "not_mature": "not_mature",
        "other": "schema_error"}


def gen_problem(rng: random.Random, difficulty: int = 5) -> dict:
    n_crops = 2 + difficulty // 2
    crops = {}
    names = ["wheat", "corn", "bean", "melon", "berry", "rice", "squash", "tomato"]
    for name in names[:n_crops]:
        grow = rng.randint(2, 3 + difficulty)
        cost = rng.randint(2, 10)
        price = cost + rng.randint(1, 4 + 2 * grow)
        crops[name] = {"cost": cost, "grow": grow, "price": price}
    return {"days": 10 + 3 * difficulty, "cash": rng.randint(15, 40), "plots": 2 + difficulty, "crops": crops}


def payload_of(P: dict) -> dict:
    return {"toy": P, "duration_days": P["days"], "initial_money": P["cash"], "farmland_count": P["plots"],
            "available_crops": sorted(P["crops"]),
            "crop_catalog": {k: {"maturity_days": c["grow"], "seed_price": c["cost"], "fruit_price": c["price"],
                                 "yield": 1} for k, c in P["crops"].items()}}


def task_prompt(P: dict) -> str:
    return ("Farm planning task (toy rules). Days 1..D. Each plot holds one crop. `plant` costs cost*n cash now and "
            "needs n free plots; a crop planted on day d can be harvested on any day >= d+grow (and <= D), paying "
            "price*n and freeing the plots. Goal: maximise final cash - initial cash.\n"
            "Plan format: a JSON list of actions, e.g. [{\"day\": 1, \"action\": \"plant\", \"crop\": \"corn\", "
            "\"plots\": 2}, {\"day\": 5, \"action\": \"harvest\", \"crop\": \"corn\", \"plots\": 2}]\n"
            f"PROBLEM: {json.dumps(P, sort_keys=True)}\n")


def make_item(P: dict, item_id: str, difficulty: int = 5, seed: int = 0) -> Item:
    return Item(item_id, "farm-toy", difficulty, seed, task_prompt(P), payload_of(P))


def plan_list(text: str):
    """The toy plan inside a model answer: the last ```json block, else the outermost [...]."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", t, flags=re.S)
    for b in reversed(blocks):
        try:
            v = json.loads(b.strip())
            if isinstance(v, list):
                return v
        except json.JSONDecodeError:
            pass
    a, b = t.find("["), t.rfind("]")
    if a < 0 or b <= a:
        raise ValueError("no JSON list found")
    v = json.loads(t[a:b + 1])
    if not isinstance(v, list):
        raise ValueError("not a list")
    return v


class ToyFarmAdapter:
    game = "farm-toy"

    def __init__(self):
        self._ref: dict[str, float] = {}
        self._memo: dict = {}
        self._lock = threading.Lock()

    def make_items(self, game: str, difficulty: int, seeds: list[int]) -> list[Item]:
        out = []
        for s in seeds:
            P = gen_problem(random.Random(s), difficulty)
            out.append(make_item(P, f"farmtoy-d{difficulty}-s{s}", difficulty, s))
        return out

    def ref(self, item: Item) -> float:
        with self._lock:
            if item.item_id not in self._ref:
                self._ref[item.item_id] = max(reference_profit(item.payload["toy"]), 1e-9)
            return self._ref[item.item_id]

    def simulate(self, item: Item, solution: str) -> dict:
        P = item.payload["toy"]
        acts = plan_list(solution)
        r = toy_simulate(P, acts)
        if not r.legal:
            raise ValueError(r.error_msg)
        crops = P["crops"]
        log = {}
        for a in acts:
            d = log.setdefault(a["day"], {"day": a["day"], "planted": []})
            if a["action"] == "plant":
                d["planted"].append({"crop": a["crop"], "quantity": a["plots"]})
        growing, fields = [], []
        for a in sorted(acts, key=lambda a: (a["day"], a["action"] != "harvest")):
            if a["action"] == "plant":
                fields.append([a["crop"], a["day"], a["plots"]])
            else:
                need = a["plots"]
                for f in sorted((f for f in fields if f[0] == a["crop"]), key=lambda f: f[1]):
                    if need and f[1] + crops[f[0]]["grow"] <= a["day"]:
                        t = min(need, f[2]); f[2] -= t; need -= t
                fields = [f for f in fields if f[2] > 0]
        for f in fields:
            growing.append({"crop": f[0], "quantity": f[2], "harvest_day": f[1] + crops[f[0]]["grow"]})
        ref = self.ref(item)
        return {"daily_log": [log[k] for k in sorted(log)], "profit": r.profit, "final_money": P["cash"] + r.profit,
                "unsold_fruits": {}, "remaining_seeds": {}, "growing_crops": growing, "reference_profit": ref,
                "relative_score": r.profit / ref}

    def score(self, item: Item, solution: str) -> ScoreResult:
        k = (item.item_id, hashlib.sha256((solution or "").encode()).hexdigest())
        with self._lock:
            if k in self._memo:
                return self._memo[k]
        res = self._score(item, solution)
        with self._lock:
            self._memo[k] = res
        return res

    def _score(self, item: Item, solution: str) -> ScoreResult:
        if not (solution or "").strip():
            return ScoreResult(0.0, False, "empty solution", "empty")
        P = item.payload["toy"]
        try:
            acts = plan_list(solution)
        except (ValueError, json.JSONDecodeError) as exc:
            return ScoreResult(0.0, False, f"JSON parse failed: {exc}", "parse_error")
        r = toy_simulate(P, acts)
        if not r.legal:
            msg = f"illegal plan: {r.error_msg}"
            s = r.error_state
            if s and r.error_day:
                g = ", ".join(f"{p['plots']} {p['crop']} (harvestable from day {p['day'] + P['crops'][p['crop']]['grow']})"
                              for p in s.get("planted", []))
                msg += (f"\nDays 1..{r.error_day - 1} of the plan are legal. State at the start of day {r.error_day} "
                        f"(before its operations): cash={s['cash']}; free plots={s['idle']}; growing={g or 'nothing'}. "
                        f"Fix day {r.error_day} and re-check every later day the same way.")
            return ScoreResult(0.0, False, msg, _CAT.get(r.error, "schema_error"))
        sim = self.simulate(item, solution)
        score = max(0.0, float(sim["relative_score"]))
        D, L = P["days"], P["plots"]
        used = sum(p["quantity"] * (min(ev["day"] + P["crops"][p["crop"]]["grow"], D + 1) - ev["day"])
                   for ev in sim["daily_log"] for p in ev["planted"])
        idle = 1.0 - used / float(L * D)
        growing = sum(b["quantity"] for b in sim["growing_crops"])
        immature = sum(p["quantity"] for ev in sim["daily_log"] for p in ev["planted"]
                       if ev["day"] + P["crops"][p["crop"]]["grow"] > D)
        diag = (f"valid plan; profit={sim['profit']} (reference greedy profit={sim['reference_profit']}); "
                f"idle plot-day rate={idle:.2f}; unsold fruit=0; leftover seeds=0; "
                f"plants not harvested by the end={growing}; plantings that cannot mature in time={immature}")
        cat = "valid_idle" if idle > 0.3 else ("valid_waste" if growing else "valid_ok")
        return ScoreResult(score, True, diag, cat)

    def features(self, item: Item) -> dict:
        return {"days": item.payload["duration_days"]}


def standard_splits(difficulty: int = 5, n_train: int = 12) -> dict[str, list[Item]]:
    """Doc §8.4 split sizes on the toy game. The final split exists only to test that nothing uses it."""
    b = ToyFarmAdapter()
    return {"train": b.make_items("farm-toy", difficulty, list(range(100, 100 + n_train))),
            "heldout": b.make_items("farm-toy", difficulty, list(range(200, 230))),
            "test": b.make_items("farm-toy", difficulty, list(range(300, 340))),
            "final": b.make_items("farm-toy", difficulty, list(range(400, 440)))}


def gen_items(n: int, seed: int, difficulty: int = 5) -> list[Item]:
    return ToyFarmAdapter().make_items("farm-toy", difficulty, list(range(seed * 1000, seed * 1000 + n)))
