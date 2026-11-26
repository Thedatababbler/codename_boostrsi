"""SelfImprove-Bench Farm adapter: generation and scoring use the benchmark's own code
(``benches/farm/farm_planner.py`` and ``game.build_prompt``), imported from the cloned repo."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from pathlib import Path

from .base import Item, ScoreResult


def extract_json(text: str):
    """Parse the first JSON object in a model reply (tolerates fences, prose and <think> blocks)."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, flags=re.S)
    if m:
        t = m.group(1)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object found")
    return json.loads(t[i : j + 1])


_CATS = [
    ("insufficient_funds", r"Insufficient funds"),
    ("farmland_exceeded", r"Not enough farmland"),
    ("missing_seeds", r"Not enough \S+ seeds"),
    ("not_mature", r"Not enough mature"),
    ("missing_fruit", r"Not enough \S+ fruit"),
    ("bad_day", r"day must be|Every days entry|must not repeat"),
    ("unknown_crop", r"Cannot (buy|sell|plant)"),
]


def categorize_error(msg: str) -> str:
    for cat, pat in _CATS:
        if re.search(pat, msg):
            return cat
    return "schema_error"


class FarmAdapter:
    game = "farm"

    def __init__(self, repo: str | Path = "third_party/SelfImprove-Bench"):
        repo = str(Path(repo).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from benches.farm import farm_planner, game  # noqa: E402

        self.fp = farm_planner
        self.game_mod = game
        self._memo: dict[tuple[str, str], ScoreResult] = {}
        self._lock = threading.Lock()

    def make_items(self, game: str, difficulty: int, seeds: list[int]) -> list[Item]:
        assert game == "farm"
        items = []
        for s in seeds:
            tid = f"farm-d{difficulty}-s{s}"
            task = self.fp.generate_task(difficulty, seed=s, task_id=tid)
            items.append(Item(tid, "farm", difficulty, s, self.game_mod.build_prompt(task), task))
        return items

    def simulate(self, item: Item, solution: str) -> dict:
        plan = extract_json(solution)
        if not isinstance(plan, dict):
            raise self.fp.FarmError("plan must be a JSON object")
        if isinstance(plan.get("days"), list) is False:
            raise self.fp.FarmError("farm plan must contain a days array")
        return self.fp.simulate_plan(dict(item.payload), plan)

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
        try:
            sim = self.simulate(item, solution)
        except (json.JSONDecodeError, ValueError) as exc:
            if isinstance(exc, self.fp.FarmError):
                msg = f"illegal plan: {exc}" + self._state_before_failure(item, solution, str(exc))
                return ScoreResult(0.0, False, msg, categorize_error(str(exc)))
            return ScoreResult(0.0, False, f"JSON parse failed: {exc}", "parse_error")
        except Exception as exc:  # malformed structures inside the plan
            return ScoreResult(0.0, False, f"illegal plan: {type(exc).__name__}: {exc}", "schema_error")
        score = max(0.0, float(sim["relative_score"] or 0.0))
        t = item.payload
        cat_ = t["crop_catalog"]
        duration, farmland = t["duration_days"], t["farmland_count"]
        used = 0
        immature = 0
        for ev in sim["daily_log"]:
            for p in ev["planted"]:
                mat = cat_[p["crop"]]["maturity_days"]
                end = ev["day"] + mat
                used += p["quantity"] * (min(end, duration + 1) - ev["day"])
                if end > duration:
                    immature += p["quantity"]
        idle = 1.0 - used / float(farmland * duration)
        unsold = sum(sim["unsold_fruits"].values())
        left = sum(sim["remaining_seeds"].values())
        growing = sum(b["quantity"] for b in sim["growing_crops"])
        diag = (f"valid plan; profit={sim['profit']} (reference greedy profit={sim['reference_profit']}); "
                f"idle plot-day rate={idle:.2f}; unsold fruit={unsold}; leftover seeds={left}; "
                f"plants not harvested by the end={growing}; plantings that cannot mature in time={immature}")
        if idle > 0.3:
            cat = "valid_idle"
        elif unsold or left or growing:
            cat = "valid_waste"
        else:
            cat = "valid_ok"
        return ScoreResult(score, True, diag, cat)

    def _state_before_failure(self, item: Item, solution: str, err: str) -> str:
        """Simulator state at the start of the failing day, from replaying the legal prefix of the plan."""
        m = re.search(r"on day (\d+)", err)
        if not m:
            return ""
        day = int(m.group(1))
        try:
            plan = extract_json(solution)
            prefix = {"days": [d for d in plan["days"] if isinstance(d, dict) and isinstance(d.get("day"), int)
                               and d["day"] < day]}
            sim = self.fp.simulate_plan(dict(item.payload), prefix)
        except Exception:
            return ""
        growing = {}
        for b in sim["growing_crops"]:
            growing.setdefault((b["crop"], b["harvest_day"]), 0)
            growing[(b["crop"], b["harvest_day"])] += b["quantity"]
        g = ", ".join(f"{q} {c} (harvestable from day {d})" for (c, d), q in sorted(growing.items(), key=lambda t: t[0][1]))
        occupied = sum(growing.values())
        return (f"\nDays 1..{day - 1} of the plan are legal. State at the start of day {day} (before its operations): "
                f"cash={sim['final_money']}; free plots={item.payload['farmland_count'] - occupied}; "
                f"seeds in inventory={sim['remaining_seeds'] or {}}; unsold fruit in inventory={sim['unsold_fruits'] or {}}; "
                f"growing={g or 'nothing'}. Fix day {day} and re-check every later day the same way.")

    def features(self, item: Item) -> dict:
        t = item.payload
        cheapest = min(c["seed_price"] for c in t["crop_catalog"].values())
        return {"n_crops": len(t["available_crops"]), "farmland": t["farmland_count"],
                "days": t["duration_days"], "money_over_cheapest_seed": t["initial_money"] / cheapest}
