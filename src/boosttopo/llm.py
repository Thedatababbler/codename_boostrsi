"""LLM clients: an OpenAI-compatible HTTP client, a deterministic FakeLLM, and a disk cache.

Every call carries a ``Request``: the rendered prompt text (what a real model sees and what
the cache key hashes) plus the structured fields (role, item, current solution, upstream
messages, diagnostics) that FakeLLM rules use instead of parsing text.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


@dataclass
class Request:
    prompt: str
    role: str = ""
    item: Any = None
    y_cur: str = ""
    upstream: list = field(default_factory=list)  # list[(role, text)]
    diag: str | None = None
    rep: int = 0
    system: str | None = None


class LLM(Protocol):
    name: str
    temperature: float

    def call(self, req: Request) -> tuple[str, int]: ...


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False))


def strip_thinking(text: str) -> tuple[str, int]:
    """Drop a leading <think>...</think> block (thinking mode); returns (answer, reasoning chars).
    An unterminated block (output truncated inside the reasoning) leaves an empty answer.
    Outputs without the tag are returned unchanged (all non-thinking cache records)."""
    if "<think>" not in text and "</think>" not in text:
        return text, 0
    head, sep, tail = text.rpartition("</think>")
    if not sep:
        return "", len(text)
    return tail.lstrip(), len(head) + len(sep)


class OpenAICompatLLM:
    """Chat-completions client over plain urllib (works with vLLM, proxies, OpenAI)."""

    def __init__(self, api_base: str, model: str, api_key: str | None = None, temperature: float = 0.7,
                 max_tokens: int = 4096, timeout: float = 600, max_concurrency: int = 32,
                 extra_body: dict | None = None, retries: int = 6):
        self.api_base = api_base.rstrip("/")
        self.name = model
        self.api_key = api_key or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = dict(extra_body or {})
        self.retries = retries
        self._sem = threading.Semaphore(max_concurrency)

    def call(self, req: Request) -> tuple[str, int]:
        text, tokens, _ = self.call_meta(req)
        return text, tokens

    def call_meta(self, req: Request) -> tuple[str, int, dict]:
        """As call(), plus {finish_reason, completion_tokens, prompt_tokens, seconds} of the final attempt."""
        msgs = []
        if req.system:
            msgs.append({"role": "system", "content": req.system})
        msgs.append({"role": "user", "content": req.prompt})
        body = {"model": self.name, "messages": msgs, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "seed": req.rep, **self.extra_body}
        data = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                with self._sem:
                    t0 = time.time()
                    r = urllib.request.Request(
                        f"{self.api_base}/chat/completions", data=data,
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
                    with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                        out = json.loads(resp.read())
                text = out["choices"][0]["message"].get("content") or ""
                usage = out.get("usage") or {}
                tokens = int(usage.get("total_tokens") or (len(req.prompt) + len(text)) // 4)
                meta = {"finish_reason": out["choices"][0].get("finish_reason"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "prompt_tokens": usage.get("prompt_tokens"), "seconds": round(time.time() - t0, 2)}
                return text, tokens, meta
            except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last}")


class FakeLLM:
    """Deterministic rule-based LLM for tests: ``rule(req) -> text``; tokens = len(prompt)//4 + 10."""

    def __init__(self, rule: Callable[[Request], str], name: str = "fake", temperature: float = 0.0):
        self.rule = rule
        self.name = name
        self.temperature = temperature
        self.n_calls = 0
        self._lock = threading.Lock()

    def call(self, req: Request) -> tuple[str, int]:
        with self._lock:
            self.n_calls += 1
        return self.rule(req), len(req.prompt) // 4 + 10


@dataclass
class Budget:
    logical_calls: int = 0
    logical_tokens: int = 0
    new_calls: int = 0
    new_tokens: int = 0
    score_calls: int = 0

    def snapshot(self) -> dict:
        return dict(self.__dict__)


# generation settings of the first runs; keys made with them keep the bare model name (cache stays valid)
_DEFAULT_GEN = {"max_tokens": 4096, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


class CachedLLM:
    """Disk cache keyed by (model, prompt hash, temperature, repeat index).

    Logical counters include cache hits (they are the budget a fresh run would spend);
    ``new_*`` counters only count real calls.
    """

    def __init__(self, inner: LLM, cache_dir: str | Path, budget: Budget | None = None, tag: str | None = None,
                 y0_tags: tuple = ()):
        self.inner = inner
        self.name = self.name_of(inner, tag)
        self.temperature = inner.temperature
        self.dir = Path(cache_dir) / "llm"
        self.budget = budget or Budget()
        self._lock = threading.Lock()
        # Round 4: names whose cached y0 (start solution) records are reused read-only before generating y0 under
        # this client's own name (e.g. y0 generated before a server-setting change, kept as fixed inputs).
        self.y0_names = [self.name_of(inner, t) for t in y0_tags]

    @staticmethod
    def name_of(inner, tag: str | None = None) -> str:
        """The model name used in cache keys: carries every generation setting that changes outputs, plus an
        optional tag for server-side settings (cfg llm.cache_tag, e.g. the KV-cache dtype). Default settings and
        no tag keep the bare model name (the earliest cache stays valid)."""
        gen = {"max_tokens": getattr(inner, "max_tokens", None), "extra_body": getattr(inner, "extra_body", None)}
        if tag:
            gen["cache_tag"] = tag
        if not any(gen.values()) or gen == _DEFAULT_GEN:
            return inner.name
        return f"{inner.name}|{sha(json.dumps(gen, sort_keys=True))[:8]}"

    def key(self, req: Request, name: str | None = None) -> str:
        sys_part = req.system or ""
        return sha(json.dumps([name or self.name, sha(sys_part + "\n\x00\n" + req.prompt), self.temperature,
                               req.rep]))

    def call_y0(self, req: Request) -> tuple[str, int]:
        """A y0 request: a record cached under one of self.y0_names if there is one, else self.call(req)."""
        for name in self.y0_names:
            k = self.key(req, name)
            path = self.dir / k[:2] / f"{k}.json"
            if path.exists():
                rec = json.loads(path.read_text(encoding="utf-8"))
                with self._lock:
                    self.budget.logical_calls += 1
                    self.budget.logical_tokens += rec["tokens"]
                return strip_thinking(rec["text"])[0], rec["tokens"]
        return self.call(req)

    def call(self, req: Request) -> tuple[str, int]:
        text, tokens, _ = self.call_meta(req)
        return text, tokens

    def call_meta(self, req: Request) -> tuple[str, int, dict]:
        """As call(), plus the call metadata stored with the cache record (empty for records written before
        metadata was kept, or for inner LLMs without call_meta). The cache key is unchanged."""
        k = self.key(req)
        path = self.dir / k[:2] / f"{k}.json"
        if path.exists():
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                with self._lock:
                    self.budget.logical_calls += 1
                    self.budget.logical_tokens += rec["tokens"]
                text, think = strip_thinking(rec["text"])
                return text, rec["tokens"], {**rec.get("meta", {}), "thinking_chars": think}
            except (json.JSONDecodeError, KeyError):
                pass
        if hasattr(self.inner, "call_meta"):
            text, tokens, meta = self.inner.call_meta(req)
        else:
            (text, tokens), meta = self.inner.call(req), {}
        rec = {"text": text, "tokens": tokens, "model": self.name}
        if meta:
            rec["meta"] = meta
        atomic_write_json(path, rec)
        with self._lock:
            self.budget.logical_calls += 1
            self.budget.logical_tokens += tokens
            self.budget.new_calls += 1
            self.budget.new_tokens += tokens
        text, think = strip_thinking(text)  # the raw output (with the reasoning) stays in the cache record
        return text, tokens, {**meta, "thinking_chars": think}


def build_llm(cfg, cache_dir: str | Path, fake_rule: Callable | None = None) -> CachedLLM:
    lc = cfg.llm
    if lc.kind == "fake":
        if fake_rule is None:
            raise ValueError("llm.kind=fake needs a fake_rule")
        inner: LLM = FakeLLM(fake_rule)
    else:
        key = os.environ.get(lc.get("api_key_env") or "", None)
        inner = OpenAICompatLLM(lc.api_base, lc.model, api_key=key, temperature=lc.temperature,
                                max_tokens=lc.max_tokens, timeout=lc.timeout,
                                max_concurrency=lc.max_concurrency, extra_body=lc.get("extra_body"))
    return CachedLLM(inner, cache_dir, tag=lc.get("cache_tag"), y0_tags=tuple(lc.get("y0_cache_tags") or ()))
