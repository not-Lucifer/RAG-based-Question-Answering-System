"""SQLAlchemy engine and session factory (SQLite by default)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_lock = threading.Lock()
_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_engine_url: str | None = None


def _enable_sqlite_fk(dbapi_conn, _record) -> None:  # noqa: ANN001 - DB-API types
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_engine() -> Engine:
    """Return the engine for ``DATABASE_URL`` (rebuilt if the URL changes, e.g. in tests)."""
    global _engine, _session_factory, _engine_url
    url = get_settings().database_url
    with _lock:
        if _engine is None or _engine_url != url:
            if _engine is not None:
                _engine.dispose()
            kwargs = {}
            if url.startswith("sqlite"):
                kwargs["connect_args"] = {"check_same_thread": False}
                db_path = url.split("sqlite:///", 1)[-1]
                if db_path and db_path != ":memory:":
                    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            _engine = create_engine(url, **kwargs)
            if url.startswith("sqlite"):
                event.listen(_engine, "connect", _enable_sqlite_fk)
            _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
            _engine_url = url
        return _engine


def SessionLocal() -> Session:  # noqa: N802 - conventional name from the blueprint
    """Open a new ORM session."""
    get_engine()
    assert _session_factory is not None
    return _session_factory()


def get_db() -> Iterator[Session]:
    """Yield a session and always close it (FastAPI dependency / scripts)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they do not exist."""
    from app.db import models  # noqa: F401 - registers models on Base.metadata

    Base.metadata.create_all(bind=get_engine())


def dispose_engine() -> None:
    """Close all pooled connections (tests)."""
    global _engine, _session_factory, _engine_url
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = _session_factory = _engine_url = None
