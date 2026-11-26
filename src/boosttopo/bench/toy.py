"""Toy benchmark for unit tests (doc section 10.4).

A solution is a JSON list of integers; the item hides a target vector. score = 1 - L1/scale,
and any negative entry makes the solution invalid (score 0). ``toy_rule`` is the FakeLLM:
deterministic role rules, where only some role combinations fix some item kinds.

Item kinds:
- ``easy``: the writer alone steps toward the target;
- ``broken``: y0 contains negatives -- only a ``repairer`` makes it valid;
- ``deep``: the writer only moves when it has an ``analyst`` message upstream.
"""

from __future__ import annotations

import json
import random

from ..llm import Request
from .base import Item, ScoreResult

DIM = 6
SCALE = 60.0


class ToyBench:
    game = "toy"

    def make_items(self, game: str, difficulty: int, seeds: list[int]) -> list[Item]:
        items = []
        kinds = ["easy", "broken", "deep"]
        for s in seeds:
            rng = random.Random(s)
            target = [rng.randint(3, 12) for _ in range(DIM)]
            kind = kinds[s % 3]
            payload = {"target": target, "kind": kind}
            items.append(Item(f"toy-d{difficulty}-s{s}", "toy", difficulty, s,
                              f"toy task {s}: find the hidden vector", payload))
        return items

    def parse(self, solution: str) -> list[int]:
        v = json.loads(solution)
        if not isinstance(v, list) or len(v) != DIM or not all(isinstance(x, int) for x in v):
            raise ValueError("bad shape")
        return v

    def score(self, item: Item, solution: str) -> ScoreResult:
        try:
            v = self.parse(solution)
        except (ValueError, TypeError, json.JSONDecodeError):
            return ScoreResult(0.0, False, "JSON parse failed", "parse_error")
        if any(x < 0 for x in v):
            return ScoreResult(0.0, False, "illegal: negative entry (insufficient funds)", "insufficient_funds")
        d = sum(abs(a - b) for a, b in zip(v, item.payload["target"]))
        return ScoreResult(max(0.0, 1.0 - d / SCALE), True, f"valid; distance={d}", "valid_ok")

    def features(self, item: Item) -> dict:
        return {"dim": DIM}


def toy_y0(item: Item) -> str:
    t = item.payload["target"]
    if item.payload["kind"] == "broken":
        return json.dumps([-1] + [max(0, x - 4) for x in t[1:]])
    return json.dumps([max(0, x - 6) for x in t])


def toy_rule(req: Request) -> str:
    item: Item = req.item
    t = item.payload["target"]
    kind = item.payload["kind"]
    if req.role == "y0":
        return toy_y0(item)
    if req.role in ("analyst", "critic"):
        return f"{req.role}-note"
    try:
        v = json.loads(req.y_cur)
        assert isinstance(v, list) and len(v) == DIM
    except Exception:
        v = [0] * DIM
    roles_up = {r for r, _ in req.upstream}
    if req.role == "repairer":
        return json.dumps([max(0, x) for x in v])
    # writer
    step = 1
    if "analyst" in roles_up:
        step = 3
    if kind == "deep" and "analyst" not in roles_up:
        return json.dumps(v)  # cannot improve deep items without analysis
    out = []
    for x, y in zip(v, t):
        if x < 0:
            out.append(x)  # writer never fixes negatives
        elif x < y:
            out.append(min(y, x + step))
        elif x > y:
            out.append(max(y, x - step))
        else:
            out.append(x)
    if "critic" in roles_up:  # critic: fix the worst coordinate fully (but not negatives)
        j = max(range(DIM), key=lambda k: abs(out[k] - t[k]) if out[k] >= 0 else -1)
        if out[j] >= 0:
            out[j] = t[j]
    return json.dumps(out)
