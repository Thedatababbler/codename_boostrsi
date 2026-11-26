from boosttopo.evalx import gap_closure
from boosttopo.graph import single_writer


def test_g_clipped():
    assert gap_closure(0.5, 0.5) == 0.0
    assert gap_closure(2.0, 0.2) == 1.0
    assert gap_closure(0.0, 0.5) == 0.0
    assert abs(gap_closure(0.6, 0.2) - 0.5) < 1e-9
    assert abs(gap_closure(1.0, 0.98) - (0.02 / 0.05)) < 1e-9  # b = s0 + 0.05 near the reference


def test_repeat_eval_hits_cache(toy):
    cfg, bench, llm, ev, log, sp = toy()
    tr = sp["train"]
    inp = {it.item_id: ev.y0[it.item_id] for it in tr}
    r1 = ev.run(single_writer(), tr, inp)
    n = llm.inner.n_calls
    r2 = ev.run(single_writer(), tr, inp)
    assert llm.inner.n_calls == n
    assert {k: v.g for k, v in r1.items()} == {k: v.g for k, v in r2.items()}
    assert all(0.0 <= v.g <= 1.0 for v in r1.values())
