"""Subgraph library and offline ensemble selection (doc section 7.4)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from statistics import mean

from ..graph import Subgraph, single_writer


@dataclass
class Entry:
    subgraph: Subgraph
    mode: str
    g: dict = field(default_factory=dict)  # item_id -> g (on y0 inputs; ada mode only)
    cost: float = 0.0  # mean tokens per item
    costs: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)
    fit: float = float("-inf")

    def covers(self, ids) -> bool:
        return all(i in self.g for i in ids)


class Library:
    def __init__(self, log=None):
        self.entries_: dict[tuple[str, str], Entry] = {}
        self.log = log
        self._lock = threading.RLock()

    def add(self, h: Subgraph, mode: str, stage: int | None = None, per_item: dict | None = None,
            fit: float | None = None, source: str = "") -> Entry:
        with self._lock:
            return self._add(h, mode, stage, per_item, fit, source)

    def _add(self, h, mode, stage, per_item, fit, source) -> Entry:
        k = (h.key(), mode)
        e = self.entries_.get(k)
        if e is None:
            e = self.entries_[k] = Entry(h, mode)
        if stage is not None:
            e.stages.append(stage)
        if per_item:
            for iid, r in per_item.items():
                e.g[iid] = r.g
                e.costs[iid] = r.cost
            e.cost = mean(e.costs.values())
        if fit is not None:
            e.fit = max(e.fit, fit)
        if self.log is not None:
            self.log.append("library.jsonl", {"key": h.key(), "mode": mode, "stage": stage, "source": source,
                                              "graph": h.to_dict(),
                                              "subgraph": h.describe(), "fit": fit, "cost": e.cost,
                                              "g": {i: r.g for i, r in (per_item or {}).items()}})
        return e

    def load_jsonl(self, path, subgraph_of) -> int:
        """Rebuild entries from a run's library.jsonl (used on resume). ``subgraph_of(key)`` -> Subgraph|None."""
        import json
        from pathlib import Path

        n = 0
        if not Path(path).exists():
            return 0
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            h = subgraph_of(r["key"])
            if h is None:
                continue
            with self._lock:
                e = self.entries_.setdefault((r["key"], r["mode"]), Entry(h, r["mode"]))
                e.g.update(r.get("g") or {})
                if r.get("fit") is not None:
                    e.fit = max(e.fit, r["fit"])
                if r.get("cost"):
                    e.cost = r["cost"]
            n += 1
        return n

    def entries(self, mode: str | None = None) -> list[Entry]:
        with self._lock:
            return [e for (k, m), e in self.entries_.items() if mode is None or m == mode]


def weighted_gain(e: Entry, g_ens: dict, r: dict) -> float:
    ids = [i for i in r if i in e.g]
    if not ids:
        return float("-inf")
    z = sum(r[i] for i in ids)
    return sum(r[i] * (max(g_ens[i], e.g[i]) - g_ens[i]) for i in ids) / z if z > 0 else 0.0


def pick_seed(library: Library, r: dict, g_ens: dict, cfg, exclude: set = ()) -> Subgraph:
    """Start from the library entry with the highest weighted gain over the current ensemble (else single writer)."""
    best, best_gain = single_writer(), 0.0
    for e in library.entries("ada"):
        if e.subgraph.key() in exclude:
            continue
        gn = weighted_gain(e, g_ens, r)
        if gn > best_gain + 1e-12:
            best, best_gain = e.subgraph, gn
    return best


def select_ensemble(library: Library, train_ids, cfg) -> list[Subgraph]:
    """Greedy, LLM-free selection on recorded per-item g vectors."""
    pool = [e for e in library.entries(mode="ada") if e.covers(train_ids)]
    chosen, g = [], {i: 0.0 for i in train_ids}
    lam = cfg.get("lam", 0.0)
    while len(chosen) < cfg.max_members and pool:
        def gain(e):
            return mean(max(g[i], e.g[i]) - g[i] for i in train_ids) - lam * e.cost / 1000.0
        best = max(pool, key=lambda e: (gain(e), -e.cost, e.subgraph.key()))
        if gain(best) <= 0:
            break
        chosen.append(best.subgraph)
        g = {i: max(g[i], best.g[i]) for i in train_ids}
        pool = [e for e in pool if e is not best]
    return chosen
