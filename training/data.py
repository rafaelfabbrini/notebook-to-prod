"""
Utilities for loading training data from CSV files or SQL databases.

The module exposes a single public class, :class:`DataLoader`, which abstracts
away the two supported data sources and returns a pandas
:class:`~pandas.DataFrame` ready for downstream validation and model training.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from core.config import settings


class DataLoader:
    """
    Load tabular data either from a CSV file or a SQL database.

    Args:
        data_path: Optional path to a CSV file containing the dataset. When
            provided, the CSV source takes precedence over the SQL fallback.
        sql_connection: SQLAlchemy-compatible connection string. Ignored when
            *data_path* is set.
        sql_query: Query text used to retrieve the dataset from the database.
    """

    def __init__(
        self,
        data_path: str | None = None,
        sql_connection: str = settings.DEFAULT_SQL_CONNECTION,
        sql_query: str = settings.DEFAULT_SQL_QUERY,
    ):
        self._data_path = data_path
        self._sql_connection = sql_connection
        self._sql_query = sql_query

    def load(self) -> pd.DataFrame:
        """
        Return the dataset as a :class:`pandas.DataFrame`.

        The method first attempts to read *data_path* (CSV). If the path is not
        supplied, it falls back to executing *sql_query* against
        *sql_connection*.

        Raises:
            FileNotFoundError: When *data_path* is provided but the file does
                not exist.
            ConnectionError: When unable to connect to the database.
            RuntimeError: When the SQL query fails.
            ValueError: When neither a CSV path nor valid SQL details are
                provided.
        """
        if self._data_path:
            return self._load_from_csv()
        if self._sql_connection is not None and self._sql_query is not None:
            return self._load_from_db()
        raise ValueError("Either data_path or SQL connection details must be provided.")

    def _load_from_csv(self) -> pd.DataFrame:
        """
        Read a CSV file from *data_path*.

        Returns:
            Parsed :class:`pandas.DataFrame`.

        Raises:
            FileNotFoundError: If the file is missing.
        """
        path = Path(self._data_path)
        if path.exists():
            return pd.read_csv(path)
        raise FileNotFoundError(f"CSV file not found at: {path}")

    def _load_from_db(self) -> pd.DataFrame:
        """
        Execute *sql_query* against *sql_connection*.

        Returns:
            Query result as a :class:`pandas.DataFrame`.

        Raises:
            ConnectionError: If the connection test fails.
            RuntimeError: If the query execution fails.
        """
        engine = create_engine(self._sql_connection)
        self._validate_db_connection(engine)
        return self._query_db(engine)

    def _validate_db_connection(self, engine: Engine) -> None:
        """
        Ensure the database connection can be established.

        Args:
            engine: SQLAlchemy :class:`~sqlalchemy.engine.Engine` instance.

        Raises:
            ConnectionError: When the test query cannot be executed.
        """
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:  # noqa: BLE001
            raise ConnectionError("Failed to connect to database") from e

    def _query_db(self, engine: Engine) -> pd.DataFrame:
        """
        Run *sql_query* and return the result.

        Args:
            engine: Validated SQLAlchemy engine.

        Returns:
            Result set as a :class:`pandas.DataFrame`.

        Raises:
            RuntimeError: When the query execution fails.
        """
        try:
            with engine.connect() as connection:
                return pd.read_sql_query(self._sql_query, connection)
        except SQLAlchemyError as e:
            raise RuntimeError("Database query failed") from e
