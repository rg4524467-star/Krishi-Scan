"""
Sync engine - moves offline journal entries into the central SQL database.

Policy (no manual sync button required):
  * Record path: if the SQL DB is reachable -> write straight through with
    synced=True.  Otherwise journal to OfflineStore.
  * Flush path: whenever the app is online (checked at page render and on
    demand), push each journal entry into Diagnosis + SyncQueue(->synced_at),
    then drop it from the journal.
  * Never silently drop: a failed flush keeps the journal untouched so the
    record is not lost.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from ..config import DATA
from . import models
from .db import get_session
from .offline_store import OfflineStore, is_online

try:
    from sqlalchemy.exc import SQLAlchemyError as _DBError
except Exception:  # pragma: no cover
    _DBError = Exception


@dataclass
class SyncStatus:
    pending: int = 0
    last_synced_at: Optional[datetime] = None
    online: bool = False
    last_error: str = ""


class SyncEngine:
    def __init__(self, store: Optional[OfflineStore] = None):
        self.store = store or OfflineStore()

    # -- record -----------------------------------------------------------
    def record_diagnosis(self, *, image_ref: str, predicted_disease: str,
                         confidence: float, crop: Optional[str] = None,
                         farmer_id: Optional[int] = None,
                         soil_data_json: Optional[str] = None,
                         weather_data_json: Optional[str] = None,
                         gatekeeper_failed: bool = False) -> dict:
        """
        Persist a diagnosis.  If the backend DB is unavailable, journal offline.
        Returns {'id':..., 'synced':bool}.
        """
        try:
            with get_session() as session:
                diag = models.Diagnosis(
                    image_ref=image_ref, crop=crop, predicted_disease=predicted_disease,
                    confidence=confidence, farmer_id=farmer_id,
                    soil_data_json=soil_data_json, weather_data_json=weather_data_json,
                    gatekeeper_failed=gatekeeper_failed,
                    synced=True,
                )
                session.add(diag)
                session.flush()
                row_id = diag.id
            return {"id": row_id, "synced": True}
        except Exception:
            # Journal fallback - the record must survive an offline moment.
            payload = {
                "image_ref": image_ref, "crop": crop,
                "predicted_disease": predicted_disease, "confidence": confidence,
                "farmer_id": farmer_id, "soil_data_json": soil_data_json,
                "weather_data_json": weather_data_json,
                "gatekeeper_failed": gatekeeper_failed,
            }
            local_id = self.store.add_diagnosis(payload)
            return {"id": local_id, "synced": False}

    # -- flush --------------------------------------------------------------
    def sync_pending(self, force: bool = False) -> SyncStatus:
        online = is_online() if force else is_online()
        status = SyncStatus(online=online, pending=self.store.pending_count)
        if not online or self.store.pending_count == 0:
            return status

        remaining = []
        for entry in self.store.pending():
            try:
                with get_session() as session:
                    payload = entry["data"]
                    diag = self._to_diagnosis(payload)
                    session.add(diag)
                    session.flush()
                    session.add(models.SyncQueue(
                        diagnosis_id=diag.id,
                        created_offline_at=datetime.fromtimestamp(
                            entry["created_offline_at"], tz=timezone.utc),
                        synced_at=datetime.now(timezone.utc),
                    ))
                # Only remove after a committed insert succeeded.
            except Exception as exc:
                remaining.append(entry)   # never drop on error
                status.last_error = str(exc)
            finally:
                pass

        if remaining:
            # Re-write only the failed entries back to the journal.
            self.store._write_all(remaining)
            status.pending = len(remaining)
        else:
            self.store.clear()
            status.pending = 0
        status.last_synced_at = datetime.now(timezone.utc)
        return status

    @staticmethod
    def _to_diagnosis(payload: dict) -> "models.Diagnosis":
        import json
        # Normalise optional JSON text or dict payloads.
        def js(v):
            return v if v is None or isinstance(v, str) else json.dumps(v)
        return models.Diagnosis(
            farmer_id=payload.get("farmer_id"),
            image_ref=payload.get("image_ref"),
            crop=payload.get("crop"),
            predicted_disease=payload.get("predicted_disease", "unknown"),
            confidence=float(payload.get("confidence", 0.0)),
            soil_data_json=js(payload.get("soil_data_json")),
            weather_data_json=js(payload.get("weather_data_json")),
            gatekeeper_failed=bool(payload.get("gatekeeper_failed", False)),
            synced=True,
        )

    def status(self) -> SyncStatus:
        return SyncStatus(online=is_online(), pending=self.store.pending_count)