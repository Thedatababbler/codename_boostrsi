"""Edit proposers (doc section 8): the default atomic proposer and the random-subspace (forest) plugin."""

from __future__ import annotations

import math
import random
from typing import Protocol

from ..graph import EDIT_KINDS, AnyEdit, Subgraph, edit_roles, expand_macro, legal_edits
from ..roles import ROLES

PROPOSERS: dict[str, type] = {}


def register_proposer(name: str):
    def deco(cls):
        PROPOSERS[name] = cls
        cls.registry_name = name
        return cls
    return deco


class EditProposer(Protocol):
    def start_search(self, seed: int) -> None: ...

    def propose(self, h: Subgraph, ctx) -> list[AnyEdit]: ...


@register_proposer("atomic")
class AtomicProposer:
    """All legal atomic edits; with macros=(...) (cfg search.macro_slot) also every legal expansion of them."""

    def __init__(self, macros=()):
        self.macros = list(macros)

    def start_search(self, seed):
        pass

    def propose(self, h, ctx):
        return legal_edits(h, ctx.limits) + [m for name in self.macros for m in expand_macro(name, h, ctx.limits)]


@register_proposer("random_subspace")
class RandomSubspaceProposer:
    def __init__(self, role_frac=0.5, edit_type_frac=0.67, resample="per_search", macro_rate=0.5,
                 macros=("critique_and_rewrite", "analyze_first", "repair_tail")):
        self.role_frac, self.edit_type_frac = role_frac, edit_type_frac
        self.resample, self.macro_rate, self.macros = resample, macro_rate, list(macros)
        self.rng = random.Random(0)
        self._draw()

    def _draw(self):
        nr = max(1, math.ceil(len(ROLES) * self.role_frac))
        ne = max(1, math.ceil(len(EDIT_KINDS) * self.edit_type_frac))
        self.roles = set(self.rng.sample(sorted(ROLES), nr))
        self.edit_types = set(self.rng.sample(sorted(EDIT_KINDS), ne))

    def start_search(self, seed):
        self.rng = random.Random(seed)
        self._draw()

    def propose(self, h, ctx):
        if self.resample == "per_step":
            self._draw()
        atomic = [e for e in legal_edits(h, ctx.limits)
                  if e.kind in self.edit_types and edit_roles(h, e) <= self.roles]
        macro = [m for name in self.macros for m in expand_macro(name, h, ctx.limits)
                 if edit_roles(h, m) <= self.roles | {"writer"}]
        k = round(len(macro) * self.macro_rate)
        return atomic + (self.rng.sample(macro, k=k) if k else [])


def make_proposer(cfg):
    name = cfg.search.proposer
    if name == "atomic":
        return AtomicProposer(cfg.search.random_subspace.macros if cfg.search.get("macro_slot") else ())
    if name == "random_subspace":
        rs = cfg.search.random_subspace
        return RandomSubspaceProposer(rs.role_frac, rs.edit_type_frac, rs.resample, rs.macro_rate, rs.macros)
    return PROPOSERS[name]()
