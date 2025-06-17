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

import json
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
        artifacts_dir: Directory where evaluation artifacts will be saved.

    Attributes:
        _pipeline: The fitted pipeline passed at construction.
        _data: Original, unvalidated training frame.
        _feature_data: Features validated by :class:`DataValidator`.
        _target_data: Target column validated by :class:`DataValidator`.
        _predictions: Pipeline predictions on *_feature_data*.
        artifacts_dir: Directory for storing evaluation artifacts.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        data: pd.DataFrame,
        metrics_path: str = "metrics.json",
        artifacts_dir: str = "artifacts",
    ):
        self._pipeline: Pipeline = pipeline
        self._data: pd.DataFrame = data
        self._metrics_path: Path = Path(metrics_path)
        self._artifacts_dir: Path = Path(artifacts_dir)
        self._feature_data, self._target_data = DataValidator.validate_training_data(
            self._data
        )
        self._predictions = self._pipeline.predict(self._feature_data)

    def evaluate(self) -> dict[str, float]:
        """
        Execute all evaluation steps.

        Returns:
            * **metrics** - aggregated cross-validation scores.
        """
        metrics = self._cross_validate()
        self._metrics_plot()
        self._prediction_plot()
        self._feature_importance_plot()

        return metrics

    def _cross_validate(self, cv: int = 5) -> dict[str, float]:
        """
        Run *k*-fold CV and aggregate RMSE and R².

        Args:
            cv: Number of folds for cross-validation.

        Returns:
            Dictionary with mean and standard deviation of train/test RMSE and
            R² across folds.

        Note:
            The metrics are saved to the path specified in :attr:`_metrics_path`.
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

        self._metrics = {
            "train_rmse_mean": train_rmse_mean,
            "train_rmse_std": train_rmse_std,
            "train_r2_mean": train_r2_mean,
            "train_r2_std": train_r2_std,
            "test_rmse_mean": test_rmse_mean,
            "test_rmse_std": test_rmse_std,
            "test_r2_mean": test_r2_mean,
            "test_r2_std": test_r2_std,
        }

        with open(self._metrics_path, "w") as file:
            json.dump(self._metrics, file)

        return self._metrics

    def _metrics_plot(self, out_file: str | Path = "metrics.png") -> str:
        """Create a bar plot comparing train and test metrics.

        Args:
            out_file: Destination PNG path. Parent directories are created if
                necessary.

        Returns:
            Absolute path to the saved plot.
        """

        out_path = self._artifacts_dir / out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(2)
        width = 0.35

        # Plot RMSE on primary y-axis
        ax1.bar(
            x - width / 2,
            [self._metrics["train_rmse_mean"], self._metrics["test_rmse_mean"]],
            width,
            label="RMSE",
            yerr=[self._metrics["train_rmse_std"], self._metrics["test_rmse_std"]],
            capsize=5,
            color="tab:blue",
        )
        ax1.set_ylabel("RMSE Score", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        # Create secondary y-axis for R²
        ax2 = ax1.twinx()
        ax2.bar(
            x + width / 2,
            [self._metrics["train_r2_mean"], self._metrics["test_r2_mean"]],
            width,
            label="R²",
            yerr=[self._metrics["train_r2_std"], self._metrics["test_r2_std"]],
            capsize=5,
            color="tab:orange",
        )
        ax2.set_ylabel("R² Score", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.xlabel("Metric Type")
        plt.title("Cross-validation Metrics")
        plt.xticks(x, ["Train", "Test"])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return str(out_path)

    def _prediction_plot(self, out_file: str | Path = "true-vs-predicted.png") -> str:
        """
        Save a scatter plot of true vs. predicted values.

        Args:
            out_file: Destination PNG path. Parent directories are created if
                necessary.

        Returns:
            Absolute path to the saved plot.
        """
        out_path = self._artifacts_dir / out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.scatter(self._predictions, self._target_data, alpha=0.5)
        plt.plot(
            [self._predictions.min(), self._predictions.max()],
            [self._target_data.min(), self._target_data.max()],
            linestyle="--",
            color="gray",
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("True vs Predicted")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        return str(out_path)

    def _feature_importance_plot(
        self, out_file: str | Path = "feature-importance.png"
    ) -> str:
        """
        Save a horizontal bar chart of feature importances.

        Args:
            out_file: Destination PNG path. Parent directories are created if
                necessary.

        Returns:
            Absolute path to the saved plot.
        """
        out_path = self._artifacts_dir / out_file
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
