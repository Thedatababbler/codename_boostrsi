import json
from statistics import mean

from boosttopo.boost.adaboost import fit_ada
from boosttopo.boost.library import Library, select_ensemble
from boosttopo.boost.residual import residual_weights
from boosttopo.config import Cfg
from boosttopo.evalx import ItemResult
from boosttopo.graph import single_writer
from boosttopo.run import fit
from boosttopo.runlog import RunLog
from boosttopo.search.grow import grow_subgraph
from boosttopo.search.proposers import AtomicProposer
from boosttopo.policy import EditStats, F0Policy
from boosttopo.graph import Limits


def _ir(iid, g, cost=0.0):
    return ItemResult(iid, g, g, g, cost, "", "", "", g, g, [g])


class _It:
    def __init__(self, i):
        self.item_id = i


# ---------------- T5
def test_grow_monotone(toy):
    cfg, bench, llm, ev, log, sp = toy()
    tr = sp["train"]
    inp = {it.item_id: ev.y0[it.item_id] for it in tr}
    hist = []

    def fit_fn(h, items, inputs):
        res = ev.run(h, items, inputs)
        f = mean(res[it.item_id].g for it in items)
        return f, 0.0

    gr = grow_subgraph(single_writer(), tr, inp, fit_fn, AtomicProposer(), F0Policy(), cfg, limits=Limits(),
                       residual={"invalid_share_hi": 0.0}, stats=EditStats(), rnd=0, search_seed=0)
    f0, _ = fit_fn(single_writer(), tr, inp)
    assert gr.fit >= f0
    assert len(gr.h.nodes) >= 2  # the toy needs an analyst or repairer to fix deep/broken items


# ---------------- T6
def test_gbdt_improves_heldout(toy, tmp_path):
    cfg, bench, llm, ev, log, sp = toy(run_dir=str(tmp_path / "run"))
    fr = fit(cfg, sp, ev, log)
    assert fr.ensemble
    ho = fr.history[-1]["heldout_g"]
    assert ho > 0.0
    tr_g = [h["train_g"] for h in fr.history]
    assert all(b >= a - 1e-12 for a, b in zip(tr_g, tr_g[1:]))  # train g monotone over stages


# ---------------- T7
def test_weights_decrease_with_g_and_capped():
    cfg = Cfg.wrap({"utility": "log", "eps": 0.1, "weight_cap": 5.0})
    g = {"i1": 0.9, "i2": 0.8, "i3": 0.7, "i4": 0.1, "i5": 0.0, "i6": 0.3}
    r = residual_weights(g, cfg)
    order = sorted(g, key=g.get)
    assert all(r[a] >= r[b] for a, b in zip(order, order[1:]))
    # section 9.2 table: i4 + i5 hold ~72% of the weight
    assert abs(r["i4"] + r["i5"] - 0.72) < 0.02
    cfg2 = Cfg.wrap({"utility": "log", "eps": 0.01, "weight_cap": 2.0})
    g2 = {f"i{k}": (0.0 if k == 0 else 0.9) for k in range(10)}
    r2 = residual_weights(g2, cfg2)
    raw_mean = mean(1 / (0.01 + v) for v in g2.values())
    z = sum(min(1 / (0.01 + v), 2.0 * raw_mean) for v in g2.values())
    assert abs(r2["i0"] - 2.0 * raw_mean / z) < 1e-9  # capped at weight_cap x mean


def test_specialist_beats_generalist_section_9_2():
    cfg = Cfg.wrap({"utility": "log", "eps": 0.1, "weight_cap": 5.0})
    g_ens = {"i1": 0.9, "i2": 0.8, "i3": 0.7, "i4": 0.1, "i5": 0.0, "i6": 0.3}
    r = residual_weights(g_ens, cfg)
    items = [_It(i) for i in g_ens]
    A = {i: _ir(i, v) for i, v in zip(g_ens, [0.95, 0.85, 0.75, 0.15, 0.05, 0.35])}
    B = {i: _ir(i, v) for i, v in zip(g_ens, [0.4, 0.3, 0.3, 0.6, 0.7, 0.2])}
    fa, fb = fit_ada(A, items, g_ens, r), fit_ada(B, items, g_ens, r)
    assert abs(fa - 0.05) < 1e-6
    assert abs(fb - 0.46) < 0.02
    assert fb > 8 * fa
    assert mean(x.g for x in B.values()) < mean(x.g for x in A.values())


def test_offline_selection_matches_hand_computation():
    from boosttopo.graph import Node, Subgraph, IN
    mk = lambda r: Subgraph((Node("n0", r),), ((IN, "n0"),), "n0")
    lib = Library()
    ids = ["a", "b", "c", "d"]
    lib.add(mk("writer"), "ada", per_item={i: _ir(i, v) for i, v in zip(ids, [0.6, 0.6, 0.6, 0.0])})
    lib.add(mk("repairer"), "ada", per_item={i: _ir(i, v) for i, v in zip(ids, [0.0, 0.0, 0.2, 0.9])})
    cfg = Cfg.wrap({"max_members": 4, "lam": 0.0})
    chosen = select_ensemble(lib, ids, cfg)
    # step 1: writer gain .45 > repairer .275; step 2: repairer adds (0+0+0+.9)/4 = .225; then nothing
    assert [h.role("n0") for h in chosen] == ["writer", "repairer"]


def test_adaboost_runs_and_heldout_not_worse(toy, tmp_path):
    cfg, bench, llm, ev, log, sp = toy(run_dir=str(tmp_path / "ada"), mode="adaboost")
    fr = fit(cfg, sp, ev, log)
    assert fr.ensemble
    assert fr.history[-1]["heldout_g"] > 0.0


# ---------------- T8 + leak
def _run_dump(toy, path, **over):
    cfg, bench, llm, ev, log, sp = toy(run_dir=str(path), **over)
    fr = fit(cfg, sp, ev, log)
    return [h.describe() for h in fr.ensemble], fr.history, (path / "edit_outcomes.jsonl").read_text()


def test_forest_switch_off_is_bitwise_identical(toy, tmp_path):
    for mode in ("gbdt", "adaboost"):
        a = _run_dump(toy, tmp_path / f"{mode}_a", mode=mode)
        b = _run_dump(toy, tmp_path / f"{mode}_b", mode=mode,
                      search={"proposer": "atomic", "random_subspace": {"role_frac": 0.25}}, forest={"k": 1},
                      sampler={"kind": "all"})
        strip = lambda hist: [{k: v for k, v in h.items() if k != "budget"} for h in hist]
        assert a[0] == b[0] and strip(a[1]) == strip(b[1]) and a[2] == b[2]


def test_forest_on_runs(toy, tmp_path):
    for mode, kind in (("gbdt", "subsample"), ("adaboost", "weighted_bootstrap")):
        ens, hist, _ = _run_dump(toy, tmp_path / f"forest_{mode}", mode=mode,
                                 search={"proposer": "random_subspace"}, forest={"k": 3},
                                 sampler={"kind": kind, "frac": 0.6})
        rec = [json.loads(l) for l in (tmp_path / f"forest_{mode}" / "rounds.jsonl").read_text().splitlines()]
        assert len(rec[0]["candidates"]) == 3


def test_heldout_and_test_ids_never_in_search_logs(toy, tmp_path):
    for mode in ("gbdt", "adaboost"):
        cfg, bench, llm, ev, log, sp = toy(run_dir=str(tmp_path / f"leak_{mode}"), mode=mode)
        fit(cfg, sp, ev, log)
        forbidden = [it.item_id for k in sp if k != "train" for it in sp[k]]
        for name in ("edit_outcomes.jsonl", "library.jsonl", "rounds.jsonl"):
            p = tmp_path / f"leak_{mode}" / name
            if p.exists():
                txt = p.read_text()
                assert not any(f'"{i}"' in txt for i in forbidden), name
