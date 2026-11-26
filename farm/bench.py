"""Benchmark adapter: items, scorer, validator diagnostics.

The executor only talks to a `Bench` with this interface, so the real Farm benchmark from the
original repo can be plugged in by implementing the same three methods.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from .reference import reference_profit
from .rules import simulate


@dataclass(frozen=True)
class Item:
    id: str
    problem: str            # JSON text
    y0_score: float | None = None   # cached y0 score (start point of g, §4.2)


@dataclass
class ScoreResult:
    score: float            # capped to [0, 1]
    raw: float              # uncapped (profit / reference profit)
    legal: bool
    profit: float
    error: str | None = None
    error_msg: str | None = None


class FarmBench:
    def __init__(self):
        self._ref: dict[str, float] = {}

    def ref(self, item: Item) -> float:
        if item.id not in self._ref:
            self._ref[item.id] = max(reference_profit(item.problem), 1e-9)
        return self._ref[item.id]

    def score(self, item: Item, plan: str | None) -> ScoreResult:
        r = simulate(item.problem, plan)
        if not r.legal:
            return ScoreResult(0.0, 0.0, False, 0.0, r.error, r.error_msg)
        raw = r.profit / self.ref(item)
        return ScoreResult(max(0.0, min(1.0, raw)), raw, True, r.profit)


def gen_problem(rng: random.Random, difficulty: int = 5) -> dict:
    n_crops = 2 + difficulty // 2
    crops = {}
    names = ["wheat", "corn", "bean", "melon", "berry", "rice", "squash", "tomato"]
    for name in names[:n_crops]:
        grow = rng.randint(2, 3 + difficulty)
        cost = rng.randint(2, 10)
        price = cost + rng.randint(1, 4 + 2 * grow)
        crops[name] = {"cost": cost, "grow": grow, "price": price}
    return {"days": 10 + 3 * difficulty, "cash": rng.randint(15, 40), "plots": 2 + difficulty,
            "crops": crops}


def gen_items(n: int, seed: int, difficulty: int = 5, prefix: str = "d") -> list[Item]:
    rng = random.Random(seed)
    return [Item(f"{prefix}{difficulty}-{seed}-{i}", json.dumps(gen_problem(rng, difficulty)))
            for i in range(n)]


def standard_splits(difficulty: int = 5, n_train: int = 12) -> dict[str, list[Item]]:
    """Doc §8.4 split sizes. Final-test is generated but must not be used."""
    return {
        "train": gen_items(n_train, 101, difficulty, "tr"),
        "heldout": gen_items(30, 202, difficulty, "ho"),
        "test": gen_items(40, 303, difficulty, "te"),
        "final": gen_items(40, 404, difficulty, "fi"),
    }
