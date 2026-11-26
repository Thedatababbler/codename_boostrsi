"""Run configuration: YAML file deep-merged over defaults, with attribute access."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """A dict with attribute access; nested dicts become Cfg too."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return v

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self))


DEFAULTS: dict[str, Any] = {
    "run_id": None,
    "seed": 0,
    "mode": "gbdt",  # gbdt | adaboost
    "bench": {
        "game": "farm",  # farm | toy
        "repo": "third_party/SelfImprove-Bench",
        "train": {"difficulty": 3, "seeds": [1000, 1040]},  # [start, stop)
        "heldout": {"difficulty": 3, "seeds": [2000, 2020]},
        # Development test splits: set per preset (no defaults, so nothing is merged in unplanned).
        "dev_tests": {},
        # One-time final test (design doc section 4.3: G1 d3 x20, G2 d4/d5 x10), fresh seeds never used
        # before. Only built and evaluated by `run --final-test`.
        "final_tests": {
            "final_G1_d3": {"difficulty": 3, "seeds": [7000, 7020]},
            "final_G2_d4": {"difficulty": 4, "seeds": [8000, 8010]},
            "final_G2_d5": {"difficulty": 5, "seeds": [9000, 9010]},
        },
    },
    "llm": {
        "kind": "openai",  # openai | fake
        "api_base": "http://127.0.0.1:8401/v1",
        "api_key_env": "BOOSTTOPO_API_KEY",
        "model": "qwen3-14b",
        "temperature": 0.7,
        "max_tokens": 4096,
        "timeout": 600,
        "max_concurrency": 48,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        # Round 4: tag for server-side settings that change outputs (part of the cache-key model name), and the
        # tags whose cached y0 records are reused as fixed start solutions (null = untagged).
        "cache_tag": None,
        "y0_cache_tags": [],
    },
    "cache_dir": "cache",
    "rounds": 6,
    "patience": 2,
    "accept_margin": 0.01,
    "max_members": 4,
    "lam": 0.0,  # per 1k tokens
    "utility": "log",
    "eps": 0.1,
    "tau": 0.8,
    "alpha": 3.0,
    "weight_cap": 5.0,
    "uniform_weights": False,  # baseline "no residual"
    "ci_level": 0.9,
    # After a round is accepted on its heldout sample, re-run the accepted subgraph on heldout with fresh
    # sampling seeds and use THAT sample for the reported heldout g and (GBDT) the heldout solutions carried
    # to the next stage. The accept decision itself is unchanged. Deviation from the design doc (section 6/7).
    "fresh_heldout_eval": True,
    # Round 4: a member that occurs more than once in an ensemble draws fresh samples (its occurrence index among
    # same-structure members enters the repeat index and cache key). Runs before round 4 had this off; reproduce
    # them with fresh_duplicate_stages=false (only runs whose ensembles contain duplicates differ).
    "fresh_duplicate_stages": True,
    # Round 4 loop attribute (off by default): the set_loop edit gives a writer-type node "repeat up to k times"
    # with a stop rule. reserve_slot: one evaluation slot per step goes to the best-ranked loop edit.
    # slot_pick (with search.slot_mode = one_per_kind): "rank" = f0's order, "random" = a seeded uniform draw.
    "loop": {"enabled": False, "ks": [2, 4, 8], "stops": ["until_legal", "keep_best"], "reserve_slot": True,
             "slot_pick": "rank"},
    # Baseline budget matching: best-of-N sample count rounded "ceil" (rounds 2-3) or "floor" (round 4 presets)
    "baselines": {"bon_round": "ceil"},
    "n_boot": 2000,
    "limits": {"max_nodes": 3, "max_indegree": 2},
    "search": {
        "proposer": "atomic",
        "policy": "f0",  # f0 | random
        "top_k": 3,
        "max_steps": 3,
        "min_gain": 0.005,
        # Round 4 macro switch (off by default): the atomic proposer also emits the three macros, and one
        # evaluation slot per step is reserved for the best-ranked macro.
        "macro_slot": False,
        # "top_k" (default) | "one_per_kind": evaluate exactly one loop edit, one macro and one atomic edit per step
        "slot_mode": "top_k",
        "random_subspace": {
            "role_frac": 0.5,
            "edit_type_frac": 0.67,
            "resample": "per_search",
            "macro_rate": 0.5,
            "macros": ["critique_and_rewrite", "analyze_first", "repair_tail"],
        },
    },
    "macro": {"ablate_on_accept": False},
    "forest": {"k": 1},
    "sampler": {"kind": "all", "frac": 0.6},
    "eval": {"n_repeats": 2},
    "library": {"complete_top": 3},
}


# Maps of named splits: a config that sets one replaces the default map instead of merging into it
# (merging silently added default splits that the preset never asked for).
REPLACE_KEYS = {"dev_tests", "final_tests"}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k not in REPLACE_KEYS:
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Cfg:
    raw: dict = {}
    if path is not None:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "tests" in (raw.get("bench") or {}) or "tests" in ((overrides or {}).get("bench") or {}):
        raise ValueError("bench.tests was renamed: use bench.dev_tests (development) / bench.final_tests")
    merged = deep_merge(DEFAULTS, raw)
    if overrides:
        merged = deep_merge(merged, overrides)
    return Cfg.wrap(merged)


def parse_override(s: str) -> dict:
    """'a.b.c=1' -> {'a': {'b': {'c': 1}}}; value parsed as YAML."""
    key, _, val = s.partition("=")
    node: Any = yaml.safe_load(val)
    for part in reversed(key.split(".")):
        node = {part: node}
    return node


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def sub_seed(root: int, *parts: Any) -> int:
    """Derive a deterministic sub-seed from the root seed and (round, branch, step, ...)."""
    h = hashlib.sha256(json.dumps([root, *parts]).encode()).hexdigest()
    return int(h[:12], 16)
