from boosttopo.llm import CachedLLM, FakeLLM, Request


def test_cache_hit_no_second_call(tmp_path):
    fake = FakeLLM(lambda r: "out:" + r.prompt)
    llm = CachedLLM(fake, tmp_path)
    a = llm.call(Request("hello", rep=0))
    b = llm.call(Request("hello", rep=0))
    assert a == b and fake.n_calls == 1
    llm.call(Request("hello", rep=1))  # a different repeat index is a different sample
    assert fake.n_calls == 2
    assert llm.budget.logical_calls == 3 and llm.budget.new_calls == 2
    # a fresh client over the same directory also hits
    fake2 = FakeLLM(lambda r: "x")
    assert CachedLLM(fake2, tmp_path).call(Request("hello"))[0] == "out:hello" and fake2.n_calls == 0
