"""
Offline journal + connectivity probe.

When the PWA is offline, diagnoses are journaled as JSON dicts to a local
store (a JSON file on desktop; browser localStorage through Pyodide's `js`
module in stlite).  The SyncEngine later flushes the journal to the SQL
database.  This is the "write to local storage + SyncQueue" half of the
offline architecture.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import DATA


def is_online(probe_url: str = "https://api.open-meteo.com/v1/forecast") -> bool:
    """Cheap reachability probe.  Fail-closed: offline on any exception."""
    try:
        import urllib.request
        req = urllib.request.Request(probe_url, headers={"Range": "bytes=0-0"}, method="GET")
        with urllib.request.urlopen(req, timeout=3) as _:
            return True
    except Exception:
        return False


class OfflineStore:
    """Append-only JSON journal of offline-created diagnoses."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DATA.offline_journal
        self._use_local_storage = False
        try:
            if __import__("sys").platform == "emscripten":   # browser/PWA
                import js  # type: ignore
                # stlite runs the kernel in a Web Worker where `localStorage`
                # does not exist; fall back to the JSON file journal there.
                self._js = js
                self._use_local_storage = hasattr(js, "localStorage") and js.localStorage is not None
        except Exception:
            pass

    # -- low-level -------------------------------------------------------
    def _read_all(self) -> List[Dict[str, Any]]:
        if self._use_local_storage:
            raw = self._js.localStorage.getItem("agrovision_offline_journal")
            if raw:
                try:
                    return json.loads(str(raw))
                except Exception:
                    return []
            return []
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_all(self, entries: List[Dict[str, Any]]) -> None:
        payload = json.dumps(entries, ensure_ascii=False)
        if self._use_local_storage:
            self._js.localStorage.setItem("agrovision_offline_journal", payload)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(payload, encoding="utf-8")

    # -- public API --------------------------------------------------------
    def add_diagnosis(self, payload: Dict[str, Any]) -> str:
        """Journal a diagnosis.  Returns the local id (uuid)."""
        import uuid
        local_id = f"off:{uuid.uuid4().hex[:12]}"
        entry = {
            "local_id": local_id,
            "created_offline_at": time.time(),
            "data": payload,
        }
        entries = self._read_all()
        entries.append(entry)
        self._write_all(entries)
        return local_id

    def pending(self) -> List[Dict[str, Any]]:
        return self._read_all()

    def remove(self, local_id: str) -> None:
        entries = [e for e in self._read_all() if e["local_id"] != local_id]
        self._write_all(entries)

    def clear(self) -> None:
        self._write_all([])

    @property
    def pending_count(self) -> int:
        return len(self._read_all())

    def is_local_storage(self) -> bool:
        return self._use_local_storage