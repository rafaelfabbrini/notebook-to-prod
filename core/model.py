"""
MLflow-backed model registry utilities.

This module defines :class:`ModelStore`, a small wrapper around MLflow Tracking
and Model Registry that makes it easy to persist, version and retrieve
scikit-learn estimators either locally (``file:`` URIs) or in a remote MLflow
service.
"""

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from core.config import settings


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
        artifacts_dir: Directory for storing model artifacts.
        metrics_path: Path to the metrics JSON file.
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
        self.metrics_path = Path("metrics.json")
        self.artifacts_dir = Path("artifacts")
        self._client = MlflowClient(
            tracking_uri=self._tracking_uri, registry_uri=self._registry_uri
        )
        self.experiment_id = self._setup_experiment()

    def save(
        self,
        model: Pipeline | BaseEstimator,
        input_example: dict | None = None,
    ) -> None:
        """
        Log a fitted estimator and register it.

        Args:
            model: A fitted scikit-learn Pipeline or BaseEstimator to be logged and
            registered.
            input_example: Optional dictionary containing a sample input for MLflow
                model signature inference.

        Raises:
            TypeError: If *model* is not a scikit-learn estimator.
        """
        if not isinstance(model, Pipeline | BaseEstimator):
            raise TypeError("Only scikit-learn models are supported.")

        with mlflow.start_run(
            run_name=f"{self._model_name}-train", experiment_id=self.experiment_id
        ):
            model_path = "model"
            mlflow.sklearn.log_model(
                model, artifact_path=model_path, input_example=input_example
            )
            model_uri = f"{mlflow.get_artifact_uri()}/{model_path}"
            mlflow.register_model(model_uri, self._model_name)

            if self.metrics_path.exists():
                with open(self.metrics_path) as file:
                    metrics = json.load(file)
                mlflow.log_metrics(metrics)

            if self.artifacts_dir.exists() and any(self.artifacts_dir.iterdir()):
                mlflow.log_artifacts(str(self.artifacts_dir), artifact_path="plots")

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

    def _setup_experiment(self, name: str = "PropertyValuation") -> str:
        """
        Ensures a single persistent MLflow experiment exists, and sets it for current
        run context. This is only done once across the pipeline's lifetime — not per
        run.

        Parameters
        ----------
        name : str
            The name of the MLflow experiment to create or reuse.

        Notes
        -----
        This function resolves the path to an absolute URI and ensures consistent
        behavior across environments, including local and Dockerized setups.

        Returns
        -------
        str
            The experiment ID of the created or existing experiment.
        """
        experiment = self._client.get_experiment_by_name(name)
        if experiment is None:
            experiment_id = self._client.create_experiment(
                name, artifact_location=self._tracking_uri
            )
        else:
            experiment_id = experiment.experiment_id

        return experiment_id

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

    def _get_latest_model_version(self) -> int:
        """
        Return the most recent registered model version.

        Returns:
            Latest version identifier as a integer.

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
