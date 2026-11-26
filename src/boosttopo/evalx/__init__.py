"""Evaluator: runs a subgraph on items, caches per-(subgraph, item, input, repeat) results on disk,
computes gap closure g and cost (doc sections 4.4 and 5.4)."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from ..graph import StageOutput, Subgraph, run_subgraph, static_scorer_calls
from ..llm import atomic_write_json, sha
from ..roles import prompt_version, y0_request


def gap_closure(s: float, s0: float) -> float:
    b = max(1.0, s0 + 0.05)
    return min(1.0, max(0.0, (s - s0) / (b - s0)))


@dataclass
class ItemResult:
    item_id: str
    score: float  # gated score, mean over repeats
    raw_score: float  # ungated output score, mean over repeats
    g: float  # mean over repeats
    cost: float  # mean tokens
    diag: str
    category: str
    solution: str  # repeat 0's gated solution (carried forward in GBDT mode)
    sol_score: float
    sol_g: float
    scores: list  # per-repeat gated scores

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("solution")
        return d


class Evaluator:
    def __init__(self, bench, llm, cache_dir: str | Path, n_repeats: int = 2, lam: float = 0.0,
                 max_workers: int = 32):
        self.bench = bench
        self.llm = llm
        self.dir = Path(cache_dir) / "eval"
        self.n_repeats = n_repeats
        self.lam = lam
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.y0: dict[str, str] = {}
        self.s0: dict[str, float] = {}
        self.items: dict[str, object] = {}
        self._lock = threading.Lock()
        self.rep_base = 0  # shifted after each rejected round so the next attempt draws fresh samples
        # Run seed: shifts every subgraph-stage sample (fit and inference) so runs with different seeds draw
        # independent LLM samples. seed 0 -> 0, i.e. identical keys to the earlier runs. y0 is not shifted.
        self.rep_seed = 0
        self.n_stage_runs = 0
        self.n_cached_stage_runs = 0
        # Round 4 (cfg fresh_duplicate_stages): the occurrence index of a member among earlier members with the
        # same structure shifts its repeat index (and so its cache key), so a repeated member draws fresh
        # samples. Occurrence 0 (every non-duplicate) keeps its earlier key. Set by the driver from the config.
        self.dedup_stages = False

    FRESH_OFFSET = 500_000  # repeat shift for the post-acceptance heldout re-evaluation (< seed stride 1e6)
    DUP_STRIDE = 20_011  # repeat shift per occurrence of a duplicated member (prime; see tests for collisions)

    def occurrence(self, h: Subgraph, earlier) -> int:
        """How many of the earlier members have h's structure (0 when dedup is off)."""
        if not self.dedup_stages:
            return 0
        k = h.key()
        return sum(1 for e in earlier if e.key() == k)

    def run_fresh(self, h: Subgraph, items, inputs: dict[str, str], n_repeats: int | None = None, occ: int = 0):
        """ev.run with repeat indices no other evaluation uses: an independent sample of h on these items."""
        old = self.rep_base
        self.rep_base = old + self.FRESH_OFFSET
        try:
            return self.run(h, items, inputs, n_repeats, occ=occ)
        finally:
            self.rep_base = old

    # ---------------------------------------------------------------- start solutions
    def make_start(self, items) -> dict[str, str]:
        """Single-writer baseline y0, generated once per item (repeat 0) and frozen."""
        def one(it):
            text, tokens = getattr(self.llm, "call_y0", self.llm.call)(y0_request(it, rep=0))
            return it, text
        for it, text in self.pool.map(one, items):
            self.items[it.item_id] = it
            self.y0[it.item_id] = text
            self.s0[it.item_id] = self.bench.score(it, text).score
        return {it.item_id: self.y0[it.item_id] for it in items}

    def g(self, item_id: str, score: float) -> float:
        return gap_closure(score, self.s0[item_id])

    def score_of(self, item, solution: str) -> float:
        return self.bench.score(item, solution).score

    # ---------------------------------------------------------------- running
    def _one(self, h: Subgraph, item, y_in: str, rep: int, occ: int = 0) -> StageOutput:
        rep = rep + self.rep_base + self.rep_seed + occ * self.DUP_STRIDE
        k = sha(json.dumps([h.key(), item.item_id, sha(y_in), rep, self.llm.name, prompt_version()]))
        path = self.dir / k[:2] / f"{k}.json"
        with self._lock:
            self.n_stage_runs += 1
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                # still charge the LLM budget for the logical calls this stage represents
                b = self.llm.budget
                with self.llm._lock:
                    b.logical_calls += len(d.get("trace", []))
                    b.logical_tokens += int(d["cost"])
                with self._lock:
                    self.n_cached_stage_runs += 1
                so = StageOutput(**d)
                if so.scorer_calls < 0:  # cached before stages counted their scorer calls (loop-free then)
                    so.scorer_calls = static_scorer_calls(h)
                return so
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        iid = item.item_id
        g_of = (lambda sc: self.g(iid, sc)) if iid in self.s0 else None
        so = run_subgraph(h, item, y_in, self.llm, self.bench, rep=rep, g_of=g_of)
        atomic_write_json(path, asdict(so))
        return so

    def run(self, h: Subgraph, items, inputs: dict[str, str], n_repeats: int | None = None,
            occ: int = 0) -> dict[str, "ItemResult"]:
        n = n_repeats or self.n_repeats
        uniq = list({it.item_id: it for it in items}.values())
        for it in uniq:
            self.items.setdefault(it.item_id, it)
        jobs = [(it, r) for it in uniq for r in range(n)]
        outs = list(self.pool.map(lambda j: self._one(h, j[0], inputs[j[0].item_id], j[1], occ), jobs))
        res: dict[str, ItemResult] = {}
        by: dict[str, list[StageOutput]] = {}
        for (it, r), so in zip(jobs, outs):
            by.setdefault(it.item_id, []).append(so)
        for iid, sos in by.items():
            gs = [self.g(iid, s.score) for s in sos]
            res[iid] = ItemResult(
                item_id=iid,
                score=sum(s.score for s in sos) / len(sos),
                raw_score=sum(s.raw_score for s in sos) / len(sos),
                g=sum(gs) / len(gs),
                cost=sum(s.cost for s in sos) / len(sos),
                diag=sos[0].diag, category=sos[0].category,
                solution=sos[0].solution, sol_score=sos[0].score, sol_g=gs[0],
                scores=[s.score for s in sos],
            )
        return res

    def mean_cost(self, res: dict[str, ItemResult]) -> float:
        return sum(r.cost for r in res.values()) / max(1, len(res))

    def cost_penalty(self, res: dict[str, ItemResult]) -> float:
        return self.lam * self.mean_cost(res) / 1000.0
