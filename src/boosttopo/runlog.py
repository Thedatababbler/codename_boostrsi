"""Run directory and JSONL logs (doc section 10.3)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .llm import atomic_write_json


class RunLog:
    def __init__(self, run_dir: str | Path | None):
        self.dir = Path(run_dir) if run_dir else None
        self._lock = threading.Lock()
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)

    def append(self, name: str, rec: dict) -> None:
        if not self.dir:
            return
        with self._lock, open(self.dir / name, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def write(self, name: str, obj) -> None:
        if self.dir:
            atomic_write_json(self.dir / name, obj)

    def read(self, name: str):
        if not self.dir or not (self.dir / name).exists():
            return None
        return json.loads((self.dir / name).read_text(encoding="utf-8"))
