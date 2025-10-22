import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


class DBConnector:
    """
    Manages database connections and sessions using SQLAlchemy.

    This class follows a singleton-like pattern for the engine and session maker
    to ensure that there is only one engine instance per application, which is
    a recommended practice for SQLAlchemy.
    """

    _engine: Engine | None = None
    _Session: sessionmaker[Session] | None = None

    def __init__(self, db_url: str | None = None):
        """
        Initializes the DBConnector. If an engine already exists, it does nothing.

        Args:
            db_url (str | None): The database connection URL. If not provided,
                                 it will try to use the DATABASE_URL environment
                                 variable. Defaults to a local SQLite database
                                 if the environment variable is not set.
        """
        if DBConnector._engine is not None:
            return

        if db_url is None:
            # Default to a local SQLite DB if no URL is provided.
            db_url = os.environ.get("QPP_DATABASE_URL", "sqlite:///./data/quick_pp.db")

        DBConnector._engine = create_engine(db_url, pool_pre_ping=True)
        DBConnector._Session = sessionmaker(
            autocommit=False, autoflush=False, bind=DBConnector._engine
        )

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Provides a transactional scope around a series of operations.

        This context manager ensures that the session is properly closed.
        It will also roll back the transaction if an exception occurs.
        """
        if DBConnector._Session is None:
            raise RuntimeError("DBConnector not initialized. Call __init__ first.")

        session: Session = DBConnector._Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def run_sql_script(self, script_path: str):
        """
        Executes an SQL script file.

        Args:
            script_path (str): The path to the SQL script file.
        """
        if DBConnector._engine is None:
            raise RuntimeError("DBConnector not initialized. Call __init__ first.")

        with open(script_path, 'r') as f:
            sql_script = f.read()

        with DBConnector._engine.connect() as connection:
            raw_connection = connection.connection
            raw_connection.executescript(sql_script)

    def setup_db(self):
        """
        Setup quick_pp database.
        """
        script_dir = os.path.dirname(__file__)
        self.run_sql_script(os.path.join(script_dir, 'setup_db.sql'))
