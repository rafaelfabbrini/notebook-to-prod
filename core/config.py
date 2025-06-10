"""
Runtime configuration for the notebook-to-prod project.

The module defines a :class:`Settings` dataclass-style model that centralises
environment-driven configuration. All public constants should be accessed via
the module-level singleton :data:`settings`.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Container for application-wide settings.

    Field values are resolved in the following order of precedence — explicit
    keyword arguments, environment variables (loaded from ``.env``), and finally
    the default literals defined below.

    Attributes:
        API_KEY: Shared secret for authenticating requests to the FastAPI
            service.
        DEFAULT_MODEL_NAME: Name assigned to newly trained MLflow models.
        DEFAULT_MODEL_REGISTRY_URI: URI of the MLflow Model Registry.
        DEFAULT_MODEL_TRACKING_URI: URI of the MLflow Tracking backend.
        DEFAULT_SQL_CONNECTION: SQLAlchemy connection string for feature
            retrieval.
        DEFAULT_SQL_QUERY: Default SQL query executed by the data layer.
        LOG_LEVEL: Root logging level for the application.

    Notes:
        *Environment variables* are automatically mapped by Pydantic using the
        field names listed above. The :pyattr:`model_config` section instructs
        Pydantic to read them from a ``.env`` file encoded as UTF-8 when they
        are not present in the process environment.
    """

    API_KEY: str
    DEFAULT_MODEL_NAME: str = "property-valuation-model"
    DEFAULT_MODEL_REGISTRY_URI: str = "file:./mlruns"
    DEFAULT_MODEL_TRACKING_URI: str = "file:./mlruns"
    DEFAULT_SQL_CONNECTION: str = "sqlite:///./data.db"
    DEFAULT_SQL_QUERY: str = "SELECT * FROM my_table"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
"""Public, eagerly-instantiated singleton holding the resolved configuration."""
