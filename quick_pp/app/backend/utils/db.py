import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Generator

from quick_pp.database.db_connector import DBConnector

log = logging.getLogger(__name__)


def _allow_sqlite_fallback(db_url: str) -> bool:
    """Only fall back from an explicit non-SQLite database when explicitly opted in.

    Silently rewriting QPP_DATABASE_URL to local SQLite inside a single Gunicorn worker
    causes multi-worker split-brain where some workers read/write PostgreSQL while others
    read/write a worker-local SQLite file.
    """
    if not db_url or db_url.startswith("sqlite"):
        return True
    return os.environ.get("QPP_ALLOW_SQLITE_FALLBACK", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _init_connector(db_url: str) -> tuple[str, DBConnector]:
    """Initialize DBConnector and run schema setup with retry backoff for network DBs."""
    if not db_url or db_url.startswith("sqlite:///./data"):
        os.makedirs("./data", exist_ok=True)
    connector = DBConnector(db_url)
    max_attempts = 1 if db_url.startswith("sqlite") else 3
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            connector.setup_db()
            return db_url, connector
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                log.warning(
                    "Database setup attempt %d/%d for %s failed (%s); retrying...",
                    attempt,
                    max_attempts,
                    db_url,
                    exc,
                )
                time.sleep(0.75 * attempt)
                connector.dispose()
                connector = DBConnector(db_url)

    if _allow_sqlite_fallback(db_url):
        log.warning(
            "Primary database (%s) unreachable (%s); falling back to local SQLite database.",
            db_url,
            last_exc,
        )
        os.makedirs("./data", exist_ok=True)
        connector.dispose()
        fallback_url = _FALLBACK_SQLITE_URL
        os.environ["QPP_DATABASE_URL"] = fallback_url
        connector = DBConnector(fallback_url)
        connector.setup_db()
        return fallback_url, connector

    log.error(
        "Configured database (%s) failed setup after %d attempts (%s). "
        "Refusing silent SQLite fallback to prevent multi-worker split-brain.",
        db_url,
        max_attempts,
        last_exc,
    )
    if last_exc is not None:
        raise last_exc
    return db_url, connector


# Initialize application-wide DBConnector singleton
_FALLBACK_SQLITE_URL = "sqlite:///./data/quick_pp.db"
_INITIAL_DB_URL = os.environ.get("QPP_DATABASE_URL") or _FALLBACK_SQLITE_URL
_DB_URL, _connector = _init_connector(_INITIAL_DB_URL)


@contextmanager
def get_db() -> Generator[Any, None, None]:
    """Yield a SQLAlchemy session from the app DBConnector.

    Example usage in endpoints:
        with get_db() as db:
            ...
    """
    global _DB_URL, _connector
    env_url = os.environ.get("QPP_DATABASE_URL") or _FALLBACK_SQLITE_URL
    if (
        env_url != _DB_URL
        or DBConnector._engine is None
        or DBConnector._engine_url != _DB_URL
    ):
        try:
            _connector.dispose()
        except Exception:
            pass
        _DB_URL, _connector = _init_connector(env_url)
    with _connector.get_session() as session:
        yield session


def dispose() -> None:
    """Dispose underlying engine/resources managed by the connector."""
    try:
        _connector.dispose()
    except Exception:
        log.exception("Failed to dispose DBConnector")


__all__ = ["get_db", "dispose", "_connector"]
