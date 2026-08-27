from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


class AppendOnlyTelemetry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def read(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        with self.path.open(encoding="utf-8") as handle:
            return tuple(json.loads(line) for line in handle if line.strip())

    def append_new(self, events: Iterable[dict[str, Any]]) -> int:
        existing = {
            (event.get("event_index"), event.get("event_type")) for event in self.read()
        }
        count = 0
        for event in events:
            key = (event.get("event_index"), event.get("event_type"))
            if key not in existing:
                self.append(event)
                existing.add(key)
                count += 1
        return count
