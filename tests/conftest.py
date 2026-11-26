import pytest

from boosttopo.config import load_config
from boosttopo.run import build, make_splits


def toy_cfg(tmp_path, **over):
    base = {
        "bench": {"game": "toy", "train": {"difficulty": 1, "seeds": [0, 12]},
                  "heldout": {"difficulty": 1, "seeds": [100, 109]},
                  "dev_tests": {"G1": {"difficulty": 1, "seeds": [200, 206]}}},
        "llm": {"kind": "fake", "max_concurrency": 4},
        "cache_dir": str(tmp_path / "cache"),
        "rounds": 4, "patience": 1, "accept_margin": 0.0, "n_boot": 200,
        "eval": {"n_repeats": 1},
    }
    from boosttopo.config import deep_merge
    return load_config(None, deep_merge(base, over))


@pytest.fixture
def toy(tmp_path):
    def make(run_dir=None, **over):
        cfg = toy_cfg(tmp_path, **over)
        bench, llm, ev, log = build(cfg, run_dir)
        splits = make_splits(cfg, bench)
        ev.make_start(splits["train"] + splits["heldout"])
        return cfg, bench, llm, ev, log, splits
    return make
