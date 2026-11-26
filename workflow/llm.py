"""LLM clients: disk cache, concurrency limit, OpenAI-compatible endpoint (e.g. vLLM Qwen3-14B).

No credentials are looked up anywhere: the endpoint URL / key are passed in explicitly.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0   # includes thinking tokens
    cached: bool = False


class LLM:
    model = "base"

    def generate(self, prompt: str, seed: int) -> tuple[str, Usage]:
        raise NotImplementedError


def cache_key(model: str, prompt: str, seed: int) -> str:
    return hashlib.sha256(json.dumps([model, prompt, seed]).encode()).hexdigest()


class CachedLLM(LLM):
    """Wraps an LLM with a JSONL disk cache and a concurrency limit.

    The key is (model, prompt, seed); the seed is derived from the step's tree path and loop /
    fanout indices (exec.Ctx.seed), so the same step at a different position or iteration is a
    different key (§4.1). Cache hits return the original usage so cost accounting is identical
    to a live run; `new_calls` counts actual model invocations.
    """

    def __init__(self, inner: LLM, path: str | None = None, max_concurrency: int = 4):
        self.inner, self.path, self.model = inner, path, inner.model
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(max_concurrency)
        self.new_calls = 0
        self.hits = 0
        if path and os.path.exists(path):
            with open(path) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self._mem[r["key"]] = r

    def generate(self, prompt: str, seed: int) -> tuple[str, Usage]:
        k = cache_key(self.model, prompt, seed)
        with self._lock:
            if k in self._mem:
                self.hits += 1
                r = self._mem[k]
                return r["text"], Usage(r["pt"], r["ct"], cached=True)
        with self._sem:
            text, u = self.inner.generate(prompt, seed)
        rec = {"key": k, "seed": seed, "text": text, "pt": u.prompt_tokens, "ct": u.completion_tokens}
        with self._lock:
            self._mem[k] = rec
            self.new_calls += 1
            if self.path:
                os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
        return text, u


class OpenAICompatLLM(LLM):
    """Chat-completions client for an OpenAI-compatible server (vLLM, SGLang, ...)."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.6, top_p: float = 0.95, max_tokens: int = 32768,
                 enable_thinking: bool = True, timeout: float = 3600, retries: int = 3):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.kw = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.enable_thinking, self.timeout, self.retries = enable_thinking, timeout, retries

    def generate(self, prompt: str, seed: int) -> tuple[str, Usage]:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "seed": seed, **self.kw,
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        err = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                             data=json.dumps(body).encode(), headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    out = json.loads(resp.read())
                msg = out["choices"][0]["message"]
                text = msg.get("content") or ""
                if msg.get("reasoning_content"):
                    text = f"<think>{msg['reasoning_content']}</think>{text}"
                u = out.get("usage", {})
                return text, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
            except Exception as e:  # noqa: BLE001
                err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {err}")
