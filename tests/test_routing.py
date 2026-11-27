"""Several identical model servers: each item's requests always go to one server (balanced round-robin), so no
request is generated twice; the cache key does not depend on the server; CallLog records every real call.
Uses two stub OpenAI-compatible HTTP servers on localhost (no model)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from boosttopo.config import load_config
from boosttopo.llm import Request
from workflow.llm import CallLog, RoutedLLM, make_llm


def _stub(tag: str, seen: list):
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((tag, body["messages"][-1]["content"], body["seed"], body["max_tokens"],
                         body.get("chat_template_kwargs")))
            out = {"choices": [{"message": {"content": f"<think>r</think>answer from {tag}"},
                                "finish_reason": "length" if "long" in body["messages"][-1]["content"] else "stop"}],
                   "usage": {"total_tokens": 7, "prompt_tokens": 3, "completion_tokens": 4}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


class _It:
    def __init__(self, iid):
        self.item_id = iid


def test_routed_servers_item_affinity_and_call_log(tmp_path):
    seen: list = []
    s1, s2 = _stub("A", seen), _stub("B", seen)
    try:
        bases = f"http://127.0.0.1:{s1.server_port}/v1,http://127.0.0.1:{s2.server_port}/v1"
        cfg = load_config(None, {"llm": {"kind": "openai", "api_base": bases, "model": "qwen3-14b",
                                         "temperature": 0.7, "max_tokens": 30000, "max_concurrency": 4,
                                         "timeout": 30, "extra_body": {"chat_template_kwargs":
                                                                       {"enable_thinking": True}},
                                         "cache_tag": "kvfp8-h200"},
                                 "cache_dir": str(tmp_path / "cache")})
        log = tmp_path / "calls.jsonl"
        llm = make_llm(cfg, call_log=str(log))
        single = make_llm(load_config(None, {**cfg.to_dict(), "llm": {**cfg.llm.to_dict(),
                                                                      "api_base": bases.split(",")[0]}}))
        assert isinstance(llm.inner, CallLog) and isinstance(llm.inner.inner, RoutedLLM)
        assert llm.name == single.name  # which / how many servers does not enter the cache key
        r = llm.inner.inner
        items = [_It(f"farm-d5-s{s}") for s in range(6000, 6010)]
        r.assign([it.item_id for it in items])
        assert sorted(list(r.owner.values()).count(k) for k in (0, 1)) == [5, 5]
        for it in items:
            for rep in range(3):
                text, _ = llm.call(Request(f"prompt {it.item_id} {rep}" + (" long" if rep == 2 else ""),
                                           item=it, rep=rep))
                assert text == f"answer from {'AB'[r.owner[it.item_id]]}"
        by_item: dict = {}
        for tag, prompt, seed, mt, ctk in seen:
            by_item.setdefault(prompt.split()[1], set()).add(tag)
            assert mt == 30000 and ctk == {"enable_thinking": True}
        assert all(len(v) == 1 for v in by_item.values())  # one server per item
        recs = [json.loads(x) for x in log.read_text().splitlines()]
        assert len(recs) == 30 and sum(x["finish_reason"] == "length" for x in recs) == 10
        assert {x["server"] for x in recs} == {0, 1}
        n = len(seen)
        llm.call(Request(f"prompt {items[0].item_id} 0", item=items[0], rep=0))  # cache hit: no call, no log
        assert len(seen) == n and len(log.read_text().splitlines()) == 30
    finally:
        s1.shutdown()
        s2.shutdown()
