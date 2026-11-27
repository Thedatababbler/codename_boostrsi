"""boosttopo.llm.CachedLLM: identical in-flight requests are merged, the first write is kept (revision §12)."""
import json

from boosttopo.llm import CachedLLM, FakeLLM, Request


def test_identical_inflight_requests_merged_and_first_write_kept(tmp_path):
    """Concurrent identical requests: one model call, every caller gets the same text; an existing record is
    never overwritten; the cache key is unchanged."""
    import threading
    import time

    from boosttopo.llm import sha

    calls = []
    lock = threading.Lock()

    def rule(r):
        with lock:
            calls.append(1)
            n = len(calls)
        time.sleep(0.2)  # keep the first call in flight while the others arrive
        return f"sample-{n}"  # a nondeterministic server: every generation differs

    llm = CachedLLM(FakeLLM(rule), tmp_path)
    req = Request("same prompt", rep=3)
    out = []
    ts = [threading.Thread(target=lambda: out.append(llm.call(req)[0])) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(calls) == 1 and set(out) == {"sample-1"}
    assert llm.budget.new_calls == 1 and llm.budget.logical_calls == 8
    # key format unchanged: sha of [model name, sha(system + sep + prompt), temperature, rep]
    k = sha(json.dumps(["fake", sha("" + "\n\x00\n" + "same prompt"), 0.0, 3]))
    assert llm.key(req) == k and (tmp_path / "llm" / k[:2] / f"{k}.json").exists()
    # another process wrote the record first: the stored record wins and is returned
    other = Request("other prompt", rep=0)
    k2 = llm.key(other)
    p2 = tmp_path / "llm" / k2[:2] / f"{k2}.json"
    p2.parent.mkdir(parents=True, exist_ok=True)
    assert llm._write_first(p2, {"text": "first", "tokens": 1}) == {"text": "first", "tokens": 1}
    assert llm._write_first(p2, {"text": "second", "tokens": 2})["text"] == "first"
    assert json.loads(p2.read_text())["text"] == "first"
    assert llm.call(other)[0] == "first"
