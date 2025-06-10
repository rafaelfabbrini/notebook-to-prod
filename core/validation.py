"""
Validation helpers for both model pipelines and raw training data.

The module provides two complementary classes:

* :class:`PipelineValidator` — Ensures that a deserialised scikit-learn
  :class:`~sklearn.pipeline.Pipeline` is structurally identical to a reference
  pipeline constructed from source code.
* :class:`DataValidator` — Performs row-wise schema validation of tabular
  training data against :class:`core.schemas.PipelineInput` (features) and
  :class:`core.schemas.PipelineOutput` (target), returning clean
  :class:`pandas.DataFrame` / :class:`pandas.Series` objects suitable for
  fitting the pipeline.
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from core.schemas import PipelineInput, PipelineOutput


class PipelineValidator:
    """
    Validate that a loaded scikit-learn pipeline matches an expected structure.

    The validator checks three conditions in sequence:

    1. The loaded object is an instance of :class:`~sklearn.pipeline.Pipeline`.
    2. All step names present in the reference pipeline are also present in the
       loaded pipeline.
    3. Each corresponding step is of the same **type** (class), ignoring fitted
       state and hyper-parameters.
    """

    def __init__(self, pipeline: Pipeline, reference_pipeline: Pipeline):
        """
        Args:
            pipeline: Pipeline deserialised from storage (e.g. MLflow).
            reference_pipeline: Pipeline freshly instantiated from code.
        """
        self._pipeline: Pipeline = pipeline
        self._reference_pipeline: Pipeline = reference_pipeline

    @property
    def _pipeline_steps(self) -> dict:
        """Return a mapping of step names to objects for the loaded pipeline."""
        return dict(self._pipeline.named_steps)

    @property
    def _reference_pipeline_steps(self) -> dict:
        """Return a mapping of step names to objects for the reference pipeline."""
        return dict(self._reference_pipeline.named_steps)

    def validate(self) -> None:
        """Run all validation checks, raising on the first failure."""
        self._validate_pipeline_type()
        self._validate_pipeline_step_names()
        self._validate_pipeline_step_types()

    def _validate_pipeline_type(self) -> None:
        """Ensure *pipeline* is an instance of :class:`~sklearn.pipeline.Pipeline`."""
        if not isinstance(self._pipeline, Pipeline):
            raise TypeError("Loaded object is not a scikit-learn Pipeline.")

    def _validate_pipeline_step_names(self) -> None:
        """Verify that no steps are missing when compared with the reference."""
        missing_steps = set(self._reference_pipeline_steps) - set(self._pipeline_steps)
        if missing_steps:
            raise ValueError(f"Loaded pipeline is missing steps: {missing_steps}")

    def _validate_pipeline_step_types(self) -> None:
        """Confirm that each step is of the same class as its reference."""
        for step_name, expected in self._reference_pipeline_steps.items():
            actual = self._pipeline_steps[step_name]
            if not isinstance(actual, type(expected)):
                raise TypeError(
                    f"Step '{step_name}' is expected to be of type "
                    f"{type(expected).__name__}, but got {type(actual).__name__}"
                )


class DataValidator:
    """
    Validate tabular training data against input and output schemas.

    The validator operates row-wise:

    * Feature columns are validated with :class:`core.schemas.PipelineInput`;
      invalid rows are rejected and numerical values are coerced to the correct
      types.
    * The target column is validated with :class:`core.schemas.PipelineOutput`.
    """

    @classmethod
    def validate_training_data(
        cls, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Validate and split a training frame into features and target.

        Args:
            data: Raw training dataset containing *all* feature columns and the
                target column.

        Returns:
            Tuple ``(validated_features, validated_target)`` where

            * ``validated_features`` is a :class:`pandas.DataFrame` matching the
              schema defined by :class:`core.schemas.PipelineInput`, and
            * ``validated_target`` is a :class:`pandas.Series` holding the
              target values validated by :class:`core.schemas.PipelineOutput`.
        """
        features = PipelineInput.get_features()
        target = PipelineOutput.get_target()

        validated_feature_data = cls.validate_feature_data(data[features])
        validated_target_data = cls.validate_target_data(data[target])

        return validated_feature_data, validated_target_data

    @staticmethod
    def validate_feature_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate feature columns row-by-row using :class:`PipelineInput`.

        Args:
            data: DataFrame containing only feature columns.

        Returns:
            Clean DataFrame with values coerced to types defined by
            :class:`PipelineInput`.
        """
        records = data.to_dict(orient="records")
        validated = PipelineInput.validate_many(records)
        return pd.DataFrame([v.model_dump() for v in validated])

    @staticmethod
    def validate_target_data(data: pd.Series) -> pd.Series:
        """
        Validate the target column row-by-row using :class:`PipelineOutput`.

        Args:
            data: Series with raw target values.

        Returns:
            Series of validated targets, retaining the original name.
        """
        target = PipelineOutput.get_target()
        validated = [
            getattr(PipelineOutput(**{target: value}), target) for value in data
        ]
        return pd.Series(validated, name=target)
