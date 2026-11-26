"""Search policy f (doc section 9.3): ranks candidate edits; only the top_k are evaluated."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Protocol

from ..config import sub_seed
from ..graph import AnyEdit, IllegalEdit, Limits, MacroEdit, Subgraph, apply_edit, edit_type
from ..roles import READERS


@dataclass
class Obs:
    subgraph: dict
    residual: dict
    edit_stats: dict
    round: int
    step: int
    budget_left: float = float("inf")
    leaf: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["budget_left"] == float("inf"):
            d["budget_left"] = None
        return d


class Policy(Protocol):
    def rank(self, h: Subgraph, cands: list[AnyEdit], obs: Obs) -> list[AnyEdit]: ...


class EditStats:
    """Running per-edit-type outcomes (count, mean gain, positive rate), also per dominant failure type."""

    def __init__(self):
        self.d: dict[str, dict] = {}

    def add(self, etype: str, fail: str, gain: float) -> None:
        for k in (etype, f"{etype}|{fail}"):
            s = self.d.setdefault(k, {"n": 0, "sum": 0.0, "pos": 0})
            s["n"] += 1
            s["sum"] += gain
            s["pos"] += int(gain > 0)

    def view(self) -> dict:
        return {k: {"n": v["n"], "mean_gain": v["sum"] / v["n"], "pos_rate": v["pos"] / v["n"]}
                for k, v in self.d.items()}


def _is_removal(e: AnyEdit) -> bool:
    steps = e.steps if isinstance(e, MacroEdit) else (e,)
    return any(s.kind in ("remove_node", "remove_edge") for s in steps)


def _adds_repairer(h: Subgraph, e: AnyEdit) -> bool:
    """True iff applying e leaves more repairer nodes than before (adds one, or turns a node into one)."""
    try:
        h2 = apply_edit(h, e, Limits(max_nodes=99, max_indegree=99))
    except (IllegalEdit, KeyError, StopIteration, ValueError):
        return False
    return h2.role_counts().get("repairer", 0) > h.role_counts().get("repairer", 0)


def _adds_reader(e: AnyEdit) -> str | None:
    steps = e.steps if isinstance(e, MacroEdit) else (e,)
    for s in steps:
        if s.kind == "add_reader":
            return s.a["role"]
    return None


class F0Policy:
    """Hand-written initial policy: the four rules of section 9.3."""

    def __init__(self, explore: float = 0.02, repair_share: float = 0.5):
        self.explore = explore
        self.repair_share = repair_share

    def _score(self, h: Subgraph, e: AnyEdit, obs: Obs) -> float:
        s = 0.0
        removal = _is_removal(e)
        # rule 1: invalid plans dominate high-weight items -> edits that ADD a repairer first
        # (not edits that remove one or turn a repairer into another role)
        if obs.residual.get("invalid_share_hi", 0.0) > self.repair_share and _adds_repairer(h, e):
            s += 100.0
        # rule 2: no reader yet -> adding readers first, analyst before critic
        if not any(h.role_counts()[r] for r in READERS):
            r = _adds_reader(e)
            if r == "analyst":
                s += 50.0
            elif r == "critic":
                s += 40.0
        # rule 3: historical mean gain + exploration bonus
        st = obs.edit_stats.get(edit_type(e), {"n": 0, "mean_gain": 0.0})
        s += st["mean_gain"] + self.explore / math.sqrt(1 + st["n"])
        # rule 4: removals last, unless the cost penalty already exceeds the subgraph's gain
        sg = obs.subgraph
        if removal and not (sg.get("cost_penalty", 0.0) > sg.get("fit", 0.0)):
            s -= 1000.0
        return s

    def rank(self, h, cands, obs):
        return sorted(cands, key=lambda e: (-self._score(h, e, obs), str(e)))


class RandomPolicy:
    """Baseline: f replaced by a random order (seeded by round/step)."""

    def __init__(self, root_seed: int = 0):
        self.root = root_seed

    def rank(self, h, cands, obs):
        rng = random.Random(sub_seed(self.root, "policy", obs.round, obs.step, h.key()))
        c = sorted(cands, key=str)
        rng.shuffle(c)
        return c


def make_policy(cfg) -> Policy:
    name = cfg.search.policy
    if name == "f0":
        return F0Policy()
    if name == "random":
        return RandomPolicy(cfg.seed)
    raise ValueError(name)
