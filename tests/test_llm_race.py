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


# ----------------------------------------------------------------------------- across processes

def _proc(cache_dir, barrier, q, no_link):
    import os
    import time

    from boosttopo.llm import CachedLLM, FakeLLM, Request
    if no_link:  # a file system without hard links: only the shard lock protects the store
        def _nolink(*a, **k):
            raise OSError("no hard links")
        os.link = _nolink

    def rule(r):
        time.sleep(0.05)
        return f"from-{os.getpid()}"

    llm = CachedLLM(FakeLLM(rule), cache_dir)
    barrier.wait()
    outs = [llm.call(Request(f"prompt {i}", rep=0))[0] for i in range(6)]
    q.put(outs)


import pytest  # noqa: E402


@pytest.mark.parametrize("where", ["local", "repo_fs"])
@pytest.mark.parametrize("no_link", [False, True])
def test_processes_sharing_one_cache_first_write_wins(tmp_path, where, no_link):
    """Several processes miss the same keys at once (worst case; the runs split work by item so this should not
    happen): each may generate, but every process gets the same record, the one stored first."""
    import multiprocessing as mp
    import shutil
    import uuid
    from pathlib import Path

    d = tmp_path if where == "local" else Path("runs") / f".race_{uuid.uuid4().hex[:8]}"  # repo fs = NFS here
    d.mkdir(parents=True, exist_ok=True)
    try:
        ctx = mp.get_context("fork")
        n = 6
        barrier, q = ctx.Barrier(n), ctx.Queue()
        ps = [ctx.Process(target=_proc, args=(str(d), barrier, q, no_link)) for _ in range(n)]
        for p in ps:
            p.start()
        res = [q.get(timeout=60) for _ in ps]
        for p in ps:
            p.join(timeout=60)
            assert p.exitcode == 0
        for i in range(6):
            got = {r[i] for r in res}
            assert len(got) == 1, got  # every process returned the same record
            llm = CachedLLM(FakeLLM(lambda r: "unused"), d)
            k = llm.key(Request(f"prompt {i}", rep=0))
            assert json.loads((d / "llm" / k[:2] / f"{k}.json").read_text())["text"] == got.pop()
        assert not list((d / "llm").rglob("*.tmp"))
    finally:
        if where != "local":
            shutil.rmtree(d, ignore_errors=True)
