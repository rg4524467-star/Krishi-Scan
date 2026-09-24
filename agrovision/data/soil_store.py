"""
Persist/load the farmer's saved soil values.

Reference: SQLite/PG `soil_records` table, keyed to a `Farmer` by phone.
`save_soil` upserts the farmer (so a phone-only profile works) and inserts a
new record row; the latest row wins on load.  DB errors degrade to None so an
unreachable backend never blocks the UI.

Callers keep using SoilValues for the session; this module is the durable copy.
"""
from __future__ import annotations

from typing import Optional

from .db import get_session
from . import models


def _find_farmer(session, phone: str):
    if not phone:
        return None
    return session.query(models.Farmer).filter(
        models.Farmer.phone == phone).first()


def _upsert_farmer(session, phone: str, name: str = "", location: str = "") -> Optional[int]:
    if not phone:
        return None
    farmer = _find_farmer(session, phone)
    if farmer is None:
        farmer = models.Farmer(phone=phone, name=name or None, location=location or None)
        session.add(farmer)
        session.flush()
    return farmer.id


def save_soil(phone: str, n=None, p=None, k=None, ph=None,
              source: str = "manual", name: str = "", location: str = "") -> bool:
    """Insert a SoilRecord for the farmer.  Returns True when persisted."""
    if not phone:
        return False
    try:
        with get_session() as session:
            farmer_id = _upsert_farmer(session, phone, name, location)
            if farmer_id is None:
                return False
            session.add(models.SoilRecord(
                farmer_id=farmer_id, n=n, p=p, k=k, ph=ph, source=source))
        return True
    except Exception:
        return False


def load_latest_soil(phone: str):
    """Latest SoilRecord as a plain dict (n/p/k/ph/source) or None."""
    if not phone:
        return None
    try:
        with get_session() as session:
            farmer = _find_farmer(session, phone)
            if farmer is None:
                return None
            record = (session.query(models.SoilRecord)
                      .filter(models.SoilRecord.farmer_id == farmer.id)
                      .order_by(models.SoilRecord.timestamp.desc())
                      .first())
            if record is None:
                return None
            return {"n": record.n, "p": record.p, "k": record.k,
                    "ph": record.ph, "source": record.source}
    except Exception:
        return None


def has_saved_soil(phone: str) -> bool:
    """Cheap existence probe used by the diagnosis page to offer 'use saved'."""
    return load_latest_soil(phone) is not None