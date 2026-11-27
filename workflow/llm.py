"""LLM access = boosttopo's client and cache (revision §12): boosttopo.llm.CachedLLM over OpenAICompatLLM.

Cache keys are boosttopo's (model name incl. generation settings and llm.cache_tag, prompt hash, temperature,
Request.rep), so workflow steps that send a multi-agent request hit the multi-agent cache. No credentials are
looked up: the client is built with api_key=None (vLLM on localhost needs none); unlike boosttopo.llm.build_llm,
no environment variable is read.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

from boosttopo.llm import CachedLLM, OpenAICompatLLM


class CacheMiss(RuntimeError):
    pass


class CacheOnly:
    """Inner client that refuses to generate: wrapping makes a CachedLLM serve the cache or raise CacheMiss.
    Keeps the wrapped client's name and generation settings, so the cache-key model name is unchanged."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.temperature = inner.temperature
        self.max_tokens = getattr(inner, "max_tokens", None)
        self.extra_body = getattr(inner, "extra_body", None)
        self.misses = 0

    def call(self, req):
        self.misses += 1
        raise CacheMiss(f"not in cache: role={req.role} rep={req.rep} item={getattr(req.item, 'item_id', None)}")


class RoutedLLM:
    """Several identical model servers (same weights, settings and GPU type): each item's requests always go to the
    same server, chosen by a hash of the item id. Work is split by item, so no two servers (or jobs) ever generate
    the same request. The cache-key name is the shared one, so which server answered does not enter the key."""

    def __init__(self, clients: list):
        assert clients and len({(c.name, c.temperature, c.max_tokens, repr(c.extra_body)) for c in clients}) == 1
        self.clients = clients
        self.owner: dict[str, int] = {}
        c0 = clients[0]
        self.name, self.temperature = c0.name, c0.temperature
        self.max_tokens, self.extra_body = c0.max_tokens, c0.extra_body

    def assign(self, item_ids) -> None:
        """Round-robin over the sorted item ids (balanced); items not assigned fall back to a hash."""
        for j, iid in enumerate(sorted(set(item_ids) - set(self.owner))):
            self.owner[iid] = j % len(self.clients)

    def server_of(self, req) -> int:
        iid = getattr(req.item, "item_id", None) or str(req.item)
        if iid in self.owner:
            return self.owner[iid]
        return int(hashlib.sha256(iid.encode()).hexdigest(), 16) % len(self.clients)

    def call(self, req):
        return self.clients[self.server_of(req)].call(req)

    def call_meta(self, req):
        text, tokens, meta = self.clients[self.server_of(req)].call_meta(req)
        return text, tokens, {**meta, "server": self.server_of(req)}


class CallLog:
    """Inner-client wrapper: one JSON line per real model call (cache hits never reach it): role, item, rep,
    finish_reason, prompt / completion tokens, seconds, server, wall-clock end time. Used for the truncated share
    and the measured throughput. Writes nothing into the cache."""

    def __init__(self, inner, path: str, keep_text: bool = False):
        self.inner, self.path = inner, path
        self.keep_text = keep_text  # also store the prompt and the raw output (smoke-test file)
        self.name, self.temperature = inner.name, inner.temperature
        self.max_tokens = getattr(inner, "max_tokens", None)
        self.extra_body = getattr(inner, "extra_body", None)
        self._lock = threading.Lock()

    def call(self, req):
        text, tokens, _ = self.call_meta(req)
        return text, tokens

    def call_meta(self, req):
        if hasattr(self.inner, "call_meta"):
            text, tokens, meta = self.inner.call_meta(req)
        else:
            (text, tokens), meta = self.inner.call(req), {}
        rec = {"role": req.role, "item": getattr(req.item, "item_id", None), "rep": req.rep, "tokens": tokens,
               **meta, "t_end": round(time.time(), 2)}
        if self.keep_text:
            rec.update(system=req.system, prompt=req.prompt, output=text)
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return text, tokens, meta


def route(env, items) -> None:
    """Assign items to servers round-robin (no-op with one server)."""
    inner = env.llm.inner
    while inner is not None and not isinstance(inner, RoutedLLM):
        inner = getattr(inner, "inner", None)
    if inner is not None:
        inner.assign([it.item_id for it in items])
        print("item -> server:", {i: inner.owner[i] for i in sorted(inner.owner)}, flush=True)


def make_llm(cfg, cache_only: bool = False, call_log: str | None = None) -> CachedLLM:
    """cfg: a boosttopo Cfg (configs/*.yaml + --set). llm.kind: openai | fake (the toy farm model)."""
    lc = cfg.llm
    if lc.kind == "fake" and cfg.bench.game == "toy":  # boosttopo's own toy benchmark and fake model
        from boosttopo.bench.toy import toy_rule
        from boosttopo.llm import FakeLLM
        inner = FakeLLM(toy_rule)
    elif lc.kind == "fake":
        from farm.fake import FakeFarmLLM
        inner = FakeFarmLLM()
    else:
        bases = lc.api_base if isinstance(lc.api_base, (list, tuple)) else str(lc.api_base).split(",")
        clients = [OpenAICompatLLM(b, lc.model, api_key=None, temperature=lc.temperature, max_tokens=lc.max_tokens,
                                   timeout=lc.timeout, max_concurrency=lc.max_concurrency,  # per server
                                   extra_body=lc.get("extra_body"))
                   for b in bases]
        inner = clients[0] if len(clients) == 1 else RoutedLLM(clients)
    if cache_only:
        inner = CacheOnly(inner)
    elif call_log:
        inner = CallLog(inner, call_log)
    return CachedLLM(inner, cfg.cache_dir, tag=lc.get("cache_tag"), y0_tags=tuple(lc.get("y0_cache_tags") or ()))
