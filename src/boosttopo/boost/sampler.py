"""Item samplers (doc section 8.4): all | subsample | weighted_bootstrap."""

from __future__ import annotations

import random

from ..config import sub_seed


class AllSampler:
    kind = "all"

    def split(self, items, weights=None, seed=0):
        return list(items), []


class Subsample:
    kind = "subsample"

    def __init__(self, frac: float, root_seed: int = 0):
        self.frac, self.root = frac, root_seed

    def split(self, items, weights=None, seed=0):
        rng = random.Random(sub_seed(self.root, "sampler", seed))
        items = sorted(items, key=lambda it: it.item_id)
        k = max(1, round(len(items) * self.frac))
        bag = rng.sample(items, k)
        ids = {it.item_id for it in bag}
        return sorted(bag, key=lambda it: it.item_id), [it for it in items if it.item_id not in ids]


class WeightedBootstrap:
    """Draw len(items)*frac items with replacement, proportional to the residual weights."""

    kind = "weighted_bootstrap"

    def __init__(self, frac: float, root_seed: int = 0):
        self.frac, self.root = frac, root_seed

    def split(self, items, weights=None, seed=0):
        rng = random.Random(sub_seed(self.root, "sampler", seed))
        items = sorted(items, key=lambda it: it.item_id)
        w = [(weights or {}).get(it.item_id, 1.0) for it in items]
        k = max(1, round(len(items) * self.frac))
        bag = rng.choices(items, weights=w, k=k)
        ids = {it.item_id for it in bag}
        return bag, [it for it in items if it.item_id not in ids]


def make_sampler(cfg):
    kind = cfg.sampler.kind
    if kind == "all":
        return AllSampler()
    if kind == "subsample":
        return Subsample(cfg.sampler.frac, cfg.seed)
    if kind == "weighted_bootstrap":
        return WeightedBootstrap(cfg.sampler.frac, cfg.seed)
    raise ValueError(kind)
