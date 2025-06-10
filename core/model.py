"""
MLflow-backed model registry utilities.

This module defines :class:`ModelStore`, a small wrapper around MLflow Tracking
and Model Registry that makes it easy to persist, version and retrieve
scikit-learn estimators either locally (``file:`` URIs) or in a remote MLflow
service.
"""

import mlflow
from config import settings
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


class ModelStore:
    """
    Wrapper around MLflow for storing and retrieving models.

    The class offers two public methods:

    * :meth:`save` — log a fitted estimator together with optional metrics and
      artefacts, then register the model; and
    * :meth:`load` — fetch a specific or the latest registered version.

    Attributes:
        _model_name: Name under which models are registered.
        _tracking_uri: MLflow Tracking URI (runs and artefacts).
        _registry_uri: MLflow Model Registry URI (may differ from tracking).
        _client: Low-level :class:`mlflow.tracking.MlflowClient` instance.
    """

    def __init__(
        self,
        model_name: str = settings.DEFAULT_MODEL_NAME,
        tracking_uri: str = settings.DEFAULT_MODEL_TRACKING_URI,
        registry_uri: str = settings.DEFAULT_MODEL_REGISTRY_URI,
    ):
        """
        Instantiate a :class:`ModelStore`.

        Args:
            model_name: MLflow *registered model* name.
            tracking_uri: URI of the MLflow Tracking backend.
            registry_uri: URI of the Model Registry.
        """
        self._model_name = model_name
        self._tracking_uri = tracking_uri
        self._registry_uri = registry_uri
        self._client = MlflowClient(tracking_uri=self._tracking_uri)
        mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_registry_uri(self._registry_uri)

    def save(
        self,
        model: Pipeline | BaseEstimator,
        artifact_path: str | None = None,
        metrics: dict[str, float] | None = None,
        artifact_files: list[str] | None = None,
    ) -> None:
        """
        Log a fitted estimator and register it.

        Args:
            model: Trained scikit-learn estimator or pipeline.
            artifact_path: Sub-directory inside the MLflow run where the model is
                logged. Defaults to the model name.
            metrics: Optional mapping of metric names to float values.
            artifact_files: Extra files (for example plots) to log alongside the
                model.

        Raises:
            TypeError: If *model* is not a scikit-learn estimator.
        """
        if not isinstance(model, Pipeline | BaseEstimator):
            raise TypeError("Only scikit-learn models are supported.")

        if not artifact_path:
            artifact_path = self._model_name

        with mlflow.start_run(run_name=f"{self._model_name}_train") as run:
            model_uri = f"runs:/{run.info.run_id}/{artifact_path}"
            mlflow.sklearn.log_model(model, artifact_path=artifact_path)
            mlflow.register_model(model_uri, self._model_name)

            if metrics:
                mlflow.log_metrics(metrics)

            if artifact_files:
                for file in artifact_files:
                    mlflow.log_artifact(file)

    def load(self, version: str | None = None) -> Pipeline | BaseEstimator:
        """
        Load a registered model.

        Args:
            version: Explicit version number to retrieve. When ``None`` the
                latest registered version is loaded.

        Returns:
            The deserialised scikit-learn estimator.

        Raises:
            ValueError: If the requested model (or version) does not exist.
        """
        if not self._exists(version=version):
            error_message = (
                f"Model '{self._model_name}' version '{version}' does not exist."
                if version is not None
                else f"No versions found for model '{self._model_name}'"
            )
            raise ValueError(error_message)

        if version is not None:
            model_uri = f"models:/{self._model_name}/{version}"
        else:
            latest_version = self._get_latest_model_version()
            model_uri = f"models:/{self._model_name}/{latest_version}"

        return mlflow.sklearn.load_model(model_uri)

    def _exists(self, version: str | None = None) -> bool:
        """
        Check whether the model (optionally a specific version) exists.

        Args:
            version: Version identifier to look up. If ``None`` any version will
                satisfy the existence check.

        Returns:
            ``True`` if the model—or the requested version—is present in the
            registry, ``False`` otherwise.
        """
        try:
            if version is not None:
                self._client.get_model_version(name=self._model_name, version=version)
                return True
            versions = self._client.search_model_versions(f"name='{self._model_name}'")
            return len(versions) > 0
        except mlflow.exceptions.RestException:
            return False

    def _get_latest_model_version(self) -> str:
        """
        Return the most recent registered model version.

        Returns:
            Latest version identifier as a string.

        Raises:
            ValueError: If the model has no registered versions.
        """
        versions = self._client.get_latest_versions(self._model_name)
        if not versions:
            raise ValueError(f"No versions found for model '{self._model_name}'")

        sorted_versions = sorted(
            versions,
            key=lambda m: m.creation_timestamp,
            reverse=True,
        )
        return sorted_versions[0].version
