import os
import logging
from contextlib import contextmanager
from typing import Any, Generator

from quick_pp.database.db_connector import DBConnector

log = logging.getLogger(__name__)


# Initialize application-wide DBConnector singleton
_DB_URL = os.environ.get("QPP_DATABASE_URL", None)
_connector = DBConnector(_DB_URL)


@contextmanager
def get_db() -> Generator[Any, None, None]:
    """Yield a SQLAlchemy session from the app DBConnector.

    Example usage in endpoints:
        with get_db() as db:
            ...
    """
    with _connector.get_session() as session:
        yield session


def dispose() -> None:
    """Dispose underlying engine/resources managed by the connector."""
    try:
        _connector.dispose()
    except Exception:
        log.exception("Failed to dispose DBConnector")


__all__ = ["get_db", "dispose", "_connector"]
