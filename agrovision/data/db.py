"""
Database engine/session management.

Uses the configured DATABASE_URL (Postgres in production, SQLite file for
local dev/tests).  The OFFLINE runtime does NOT open this - it uses the
JSON OfflineStore and only syncs when a cloud endpoint is reachable.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import DATA
from .models import Base


def _make_engine():
    connect_args = {}
    if DATA.database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    return create_engine(DATA.database_url, connect_args=connect_args, future=True)


engine = _make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    # The browser/Pyodide runtime never opens SQLite - it journals to the JSON
    # OfflineStore (see data/offline_store.py) and only syncs when a cloud
    # endpoint is reachable.  Skipping here avoids creating files under a
    # Windows-style path that does not exist in the in-memory filesystem.
    from ..util.runtime import is_pyodide
    if is_pyodide():
        return
    Base.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:
    # Browser/Pyodide: no SQLite connection; callers (history, soil_store) wrap
    # this in try/except and degrade to the offline journal / guided entry.
    from ..util.runtime import is_pyodide
    if is_pyodide():
        raise RuntimeError("SQLite unavailable in the browser runtime")
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
