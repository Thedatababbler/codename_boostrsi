"""Benchmark adapter contract (doc section 4.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Item:
    item_id: str
    game: str
    difficulty: int
    seed: int
    prompt: str  # full task text shown to the agent
    payload: dict  # raw instance, used for scoring (contains the reference)

    def __hash__(self) -> int:
        return hash(self.item_id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Item) and other.item_id == self.item_id


@dataclass
class ScoreResult:
    score: float  # benchmark raw score; invalid -> 0
    valid: bool
    diag: str  # human-readable reason, kept as close to the benchmark's message as possible
    category: str = ""  # coarse failure type used for residual clustering


class BenchAdapter(Protocol):
    game: str

    def make_items(self, game: str, difficulty: int, seeds: list[int]) -> list[Item]: ...

    def score(self, item: Item, solution: str) -> ScoreResult: ...

    def features(self, item: Item) -> dict: ...


def make_bench(cfg) -> BenchAdapter:
    game = cfg.bench.game
    if game == "farm":
        from .farm import FarmAdapter

        return FarmAdapter(cfg.bench.repo)
    if game == "toy":
        from .toy import ToyBench

        return ToyBench()
    raise ValueError(f"unknown game {game}")
