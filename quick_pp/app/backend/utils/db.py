import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

from quick_pp.database.db_connector import DBConnector


log = logging.getLogger(__name__)


# Initialize application-wide DBConnector singleton
_FALLBACK_SQLITE_URL = "sqlite:///./data/quick_pp.db"
_DB_URL = os.environ.get("QPP_DATABASE_URL") or _FALLBACK_SQLITE_URL
if not _DB_URL or _DB_URL.startswith("sqlite:///./data"):
    os.makedirs("./data", exist_ok=True)
_connector = DBConnector(_DB_URL)
try:
    _connector.setup_db()
except Exception as exc:
    log.warning(
        "Primary database (%s) unreachable (%s); falling back to local SQLite database.",
        _DB_URL,
        exc,
    )
    os.makedirs("./data", exist_ok=True)
    _connector.dispose()
    _DB_URL = _FALLBACK_SQLITE_URL
    os.environ["QPP_DATABASE_URL"] = _DB_URL
    _connector = DBConnector(_DB_URL)
    _connector.setup_db()


@contextmanager
def get_db() -> Generator[Any, None, None]:
    """Yield a SQLAlchemy session from the app DBConnector.

    Example usage in endpoints:
        with get_db() as db:
            ...
    """
    global _DB_URL, _connector
    env_url = os.environ.get("QPP_DATABASE_URL") or _FALLBACK_SQLITE_URL
    if env_url != _DB_URL or DBConnector._engine is None or DBConnector._engine_url != _DB_URL:
        try:
            _connector.dispose()
        except Exception:
            pass
        _DB_URL = env_url
        _connector = DBConnector(_DB_URL)
        try:
            _connector.setup_db()
        except Exception as exc:
            log.warning(
                "Database (%s) unreachable (%s); falling back to local SQLite database.",
                _DB_URL,
                exc,
            )
            os.makedirs("./data", exist_ok=True)
            _connector.dispose()
            _DB_URL = _FALLBACK_SQLITE_URL
            os.environ["QPP_DATABASE_URL"] = _DB_URL
            _connector = DBConnector(_DB_URL)
            _connector.setup_db()
    with _connector.get_session() as session:
        yield session


def dispose() -> None:
    """Dispose underlying engine/resources managed by the connector."""
    try:
        _connector.dispose()
    except Exception:
        log.exception("Failed to dispose DBConnector")


__all__ = ["get_db", "dispose", "_connector"]
