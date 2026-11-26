import json

import pytest

from boosttopo.bench.farm import FarmAdapter


@pytest.fixture(scope="module")
def farm():
    return FarmAdapter("third_party/SelfImprove-Bench")


def test_generation_matches_rules(farm):
    it = farm.make_items("farm", 3, [7])[0]
    t = it.payload
    assert len(t["available_crops"]) == 8 and t["farmland_count"] == 16 and t["duration_days"] == 16
    assert "reference_profit" not in it.prompt.split("## Current task")[1]


def test_handwritten_legal_plan_positive_and_illegal_zero(farm):
    it = farm.make_items("farm", 3, [7])[0]
    cat = it.payload["crop_catalog"]
    c = min(cat.values(), key=lambda c: c["maturity_days"])
    q = min(it.payload["farmland_count"], it.payload["initial_money"] // c["seed_price"])
    d = 1 + c["maturity_days"]
    plan = {"days": [{"day": 1, "buy_seeds": [{"crop": c["id"], "quantity": q}], "plant": [{"crop": c["id"], "quantity": q}]},
                     {"day": d, "harvest": [{"crop": c["id"], "quantity": q}],
                      "sell": [{"crop": c["id"], "quantity": q * c["yield_per_harvest"]}]}]}
    r = farm.score(it, json.dumps(plan))
    assert r.valid and r.score > 0, r
    bad = {"days": [{"day": 1, "buy_seeds": [{"crop": c["id"], "quantity": 1000}]}]}
    r2 = farm.score(it, json.dumps(bad))
    assert not r2.valid and r2.score == 0 and r2.category == "insufficient_funds"
    assert farm.score(it, "not json").category == "parse_error"
    assert farm.score(it, "```json\n" + json.dumps(plan) + "\n```").valid
