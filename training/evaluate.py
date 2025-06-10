"""
Model evaluation utilities for the property-valuation pipeline.

This module defines :class:`ModelEvaluator`, which automates post-training
assessment of a fitted scikit-learn :class:`~sklearn.pipeline.Pipeline`.  The
class computes cross-validated metrics and produces two diagnostic figures:

* **True vs. predicted scatter plot** - visualises overall fit.
* **Feature-importance bar chart** - ranks model drivers.

Both plots are saved to disk so they can be logged as artefacts alongside the
model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from core.schemas import PipelineInput
from core.validation import DataValidator


class ModelEvaluator:
    """
    Run cross-validation and generate basic diagnostics for a pipeline.

    Args:
        pipeline: Fitted scikit-learn pipeline.
        data: Raw training dataset containing both features and target.

    Attributes:
        _pipeline: The fitted pipeline passed at construction.
        _data: Original, unvalidated training frame.
        _feature_data: Features validated by :class:`DataValidator`.
        _target_data: Target column validated by :class:`DataValidator`.
        _predictions: Pipeline predictions on *_feature_data*.
    """

    def __init__(self, pipeline: Pipeline, data: pd.DataFrame):
        self._pipeline: Pipeline = pipeline
        self._data = data
        self._feature_data, self._target_data = DataValidator.validate_training_data(
            self._data
        )
        self._predictions = self._pipeline.predict(self._feature_data)

    def evaluate(self) -> tuple[dict[str, float], list[str]]:
        """
        Execute all evaluation steps.

        Returns:
            Tuple ``(metrics, artefact_files)`` where

            * **metrics** - aggregated cross-validation scores.
            * **artefact_files** - file paths to the generated plots.
        """
        metrics = self._cross_validate()
        artifact_files: list[str] = []
        artifact_files.append(self._prediction_plot())
        artifact_files.append(self._feature_importance())

        return metrics, artifact_files

    def _cross_validate(self, cv: int = 5) -> dict[str, float]:
        """
        Run *k*-fold CV and aggregate RMSE and R².

        Args:
            cv: Number of folds for cross-validation.

        Returns:
            Dictionary with mean and standard deviation of train/test RMSE and
            R² across folds.
        """
        scoring = {
            "neg_rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        }
        scores = cross_validate(
            self._pipeline,
            self._feature_data,
            self._target_data,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
        )

        def _aggregate(arr: np.ndarray) -> tuple[float, float]:
            """Return k-fold mean and sample standard deviation."""
            return arr.mean().item(), arr.std(ddof=1).item()

        train_rmse_mean, train_rmse_std = _aggregate(-scores["train_neg_rmse"])
        test_rmse_mean, test_rmse_std = _aggregate(-scores["test_neg_rmse"])
        train_r2_mean, train_r2_std = _aggregate(scores["train_r2"])
        test_r2_mean, test_r2_std = _aggregate(scores["test_r2"])

        return {
            "train_rmse_mean": train_rmse_mean,
            "train_rmse_std": train_rmse_std,
            "train_r2_mean": train_r2_mean,
            "train_r2_std": train_r2_std,
            "test_rmse_mean": test_rmse_mean,
            "test_rmse_std": test_rmse_std,
            "test_r2_mean": test_r2_mean,
            "test_r2_std": test_r2_std,
        }

    def _prediction_plot(
        self, out_file: str | Path = "plots/true_vs_predicted.png"
    ) -> str:
        """
        Save a scatter plot of true vs. predicted values.

        Args:
            out_file: Destination PNG path. Parent directories are created if
                necessary.

        Returns:
            Absolute path to the saved plot.
        """
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.scatter(self._target_data, self._predictions, alpha=0.5)
        plt.plot(
            [self._target_data.min(), self._target_data.max()],
            [self._target_data.min(), self._target_data.max()],
            linestyle="--",
            color="gray",
        )
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("True vs Predicted")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return str(out_path)

    def _feature_importance(
        self, out_file: str | Path = "plots/feature_importance.png"
    ) -> str:
        """
        Save a horizontal bar chart of feature importances.

        Args:
            out_file: Destination PNG path. Parent directories are created if
                necessary.

        Returns:
            Absolute path to the saved plot.
        """
        out_path = Path(out_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._pipeline.named_steps["model"]
        features = PipelineInput.get_features()
        importances = model.feature_importances_

        plt.figure(figsize=(8, 4))
        plt.barh(features, importances)
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return str(out_path)
