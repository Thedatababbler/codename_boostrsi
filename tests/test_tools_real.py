"""Tools on the real benchmark (SelfImprove-Bench Farm, d5 train seeds): crop_stats reads the real catalog fields,
simulate's profit equals the scorer's on hand-written legal plans, and no tool output contains the reference
profit or the relative score. Skipped when third_party/SelfImprove-Bench is not cloned."""
import json
import re
from pathlib import Path

import pytest

from workflow.exec import Ctx
from workflow.model import State
from workflow.tools import crop_stats, simulate, validate

REPO = Path("third_party/SelfImprove-Bench")
pytestmark = pytest.mark.skipif(not (REPO / "benches/farm/farm_planner.py").exists(), reason="benchmark not cloned")
SEEDS = [1000, 1003, 1007, 1011]  # train split (never test / heldout / final-test seeds)


@pytest.fixture(scope="module")
def items():
    from boosttopo.bench.farm import FarmAdapter
    fa = FarmAdapter(REPO)
    return fa, fa.make_items("farm", 5, SEEDS)


def cycle_plan(t: dict, crop: str, fill: float = 1.0) -> dict:
    """Plant `crop` on as many free plots as cash allows (times `fill`), harvest and sell everything at maturity,
    repeat while a new planting still matures before the end."""
    c, D, L = t["crop_catalog"][crop], t["duration_days"], t["farmland_count"]
    money, growing, days = t["initial_money"], {}, []
    for d in range(1, D + 1):
        ops = {"day": d, "harvest": [], "sell": [], "buy_seeds": [], "plant": []}
        q_h = growing.pop(d, 0)
        if q_h:
            ops["harvest"].append({"crop": crop, "quantity": q_h})
            ops["sell"].append({"crop": crop, "quantity": q_h * c["yield_per_harvest"]})
            money += q_h * c["yield_per_harvest"] * c["fruit_price"]
        free = L - sum(growing.values())
        q = int(min(free, money // c["seed_price"]) * fill)
        if q > 0 and d + c["maturity_days"] <= D:
            ops["buy_seeds"].append({"crop": crop, "quantity": q})
            ops["plant"].append({"crop": crop, "quantity": q})
            money -= q * c["seed_price"]
            growing[d + c["maturity_days"]] = growing.get(d + c["maturity_days"], 0) + q
        if any(ops[k] for k in ("harvest", "sell", "buy_seeds", "plant")):
            days.append(ops)
    return {"days": days}


def plans(t):
    crops = sorted(t["available_crops"], key=lambda k: t["crop_catalog"][k]["seed_price"])
    for crop in crops[:3]:
        for fill in (1.0, 0.5):
            yield crop, fill, json.dumps(cycle_plan(t, crop, fill))


def _num_in(x, s: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(str(x))}(?![\d])", s) is not None


def test_crop_stats_reads_real_catalog(items):
    fa, its = items
    for it in its:
        t = it.payload
        out = crop_stats(State(problem=it.prompt), Ctx(None, fa, it))
        assert "profit/plot/cycle" in out and "unavailable" not in out
        c = t["crop_catalog"]
        best = max(t["available_crops"], key=lambda k: (c[k]["yield_per_harvest"] * c[k]["fruit_price"]
                                                        - c[k]["seed_price"]) / c[k]["maturity_days"])
        assert out.startswith(best + ":"), out


def test_simulate_profit_equals_scorer_and_no_reference(items):
    fa, its = items
    n_legal = 0
    for it in its:
        for crop, fill, p in plans(it.payload):
            sr = fa.score(it, p)
            assert sr.valid, (it.item_id, crop, fill, sr.diag)
            n_legal += 1
            sim = fa.simulate(it, p)
            ctx = Ctx(None, fa, it)  # diag_mode clean
            st = State(problem=it.prompt, plan=p)
            outs = [validate(st, ctx), simulate(st, ctx), crop_stats(st, ctx)]
            got = float(re.search(r"profit=(-?[\d.]+)", outs[1]).group(1))
            want = float(re.search(r"profit=(-?[\d.]+)", sr.diag).group(1))
            assert got == want == float(sim["profit"]), (it.item_id, crop, got, want)
            assert float(re.search(r"profit=(-?[\d.]+)", outs[0]).group(1)) == want
            ref, rel = sim["reference_profit"], sim["relative_score"]
            assert ref != sim["profit"]
            for o in outs:
                assert "reference" not in o
                assert not _num_in(ref, o), (it.item_id, ref, o)
                # 3+ decimals: at 2 decimals an idle rate can coincide with the score (seen: 0.57 on s1007)
                for r in {str(rel), f"{rel:.3f}", f"{rel:.4f}", f"{sr.score:.3f}"}:
                    assert not _num_in(r, o), (it.item_id, r, o)
    assert n_legal == 6 * len(SEEDS)


def test_illegal_plan_tools(items):
    fa, its = items
    it = its[0]
    c = min(it.payload["crop_catalog"].values(), key=lambda c: c["seed_price"])
    bad = json.dumps({"days": [{"day": 1, "buy_seeds": [{"crop": c["id"], "quantity": 2}]},
                               {"day": 2, "plant": [{"crop": c["id"], "quantity": 3}]}]})
    sr = fa.score(it, bad)
    assert not sr.valid and sr.category == "missing_seeds"
    st, ctx = State(problem=it.prompt, plan=bad), Ctx(None, fa, it)
    v = validate(st, ctx)
    assert v == sr.diag and "State at the start of day 2" in v  # no reference in an illegal diag: clean = original
    assert simulate(st, ctx).startswith("simulation stopped:")
