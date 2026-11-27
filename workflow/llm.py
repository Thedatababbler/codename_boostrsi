"""LLM access = boosttopo's client and cache (revision §12): boosttopo.llm.CachedLLM over OpenAICompatLLM.

Cache keys are boosttopo's (model name incl. generation settings and llm.cache_tag, prompt hash, temperature,
Request.rep), so workflow steps that send a multi-agent request hit the multi-agent cache. No credentials are
looked up: the client is built with api_key=None (vLLM on localhost needs none); unlike boosttopo.llm.build_llm,
no environment variable is read.
"""
from __future__ import annotations

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


def make_llm(cfg, cache_only: bool = False) -> CachedLLM:
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
        inner = OpenAICompatLLM(lc.api_base, lc.model, api_key=None, temperature=lc.temperature,
                                max_tokens=lc.max_tokens, timeout=lc.timeout, max_concurrency=lc.max_concurrency,
                                extra_body=lc.get("extra_body"))
    if cache_only:
        inner = CacheOnly(inner)
    return CachedLLM(inner, cfg.cache_dir, tag=lc.get("cache_tag"), y0_tags=tuple(lc.get("y0_cache_tags") or ()))
