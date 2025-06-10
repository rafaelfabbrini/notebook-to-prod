"""
End-to-end scikit-learn training and inference pipeline.

The module defines :class:`ModelPipeline`, a convenience wrapper that combines
data validation, preprocessing, model fitting, MLflow-backed persistence and
online prediction in a single class suitable for both batch and real-time use.
"""

import pandas as pd
from category_encoders import TargetEncoder
from model import ModelStore
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from core.schemas import PipelineInput, PipelineOutput
from core.validation import DataValidator, PipelineValidator

HYPERPARAMETERS = {
    "learning_rate": 0.01,
    "n_estimators": 300,
    "max_depth": 5,
    "loss": "absolute_error",
}


class ModelPipeline:
    """
    Thin orchestration layer around preprocessing, model and registry."""

    def __init__(self):
        """Initialise the pipeline and associated :class:`ModelStore`."""
        self.model_store = ModelStore()

    @property
    def pipeline(self) -> Pipeline:
        """
        Construct the preprocessing-plus-model pipeline.

        Returns:
            A scikit-learn :class:`~sklearn.pipeline.Pipeline` consisting of a
            categorical :class:`~category_encoders.target_encoder.TargetEncoder`
            followed by a :class:`~sklearn.ensemble.GradientBoostingRegressor`.
        """
        preprocessor = ColumnTransformer(
            [
                (
                    "categorical",
                    TargetEncoder(),
                    PipelineInput.get_categorical_fields(),
                )
            ]
        )
        model = GradientBoostingRegressor(**HYPERPARAMETERS)
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def train(self, data: pd.DataFrame) -> None:
        """
        Validate and fit the model.

        Args:
            data: Training dataset containing both features and target column.
        """
        feature_data, target_data = DataValidator.validate_training_data(data)
        self.pipeline.fit(feature_data, target_data)

    def predict(self, input_data: PipelineInput) -> PipelineOutput:
        """
        Generate a single prediction.

        Args:
            input_data: Typed feature payload.

        Returns:
            A :class:`core.schemas.PipelineOutput` with the predicted target
            value.
        """
        pipeline = self.model_store.load()
        PipelineValidator(pipeline, self.pipeline).validate()
        data = self._prepare_input(input_data)
        prediction = pipeline.predict(data)[0]
        return PipelineOutput.from_prediction(prediction)

    @staticmethod
    def _prepare_input(input_data: PipelineInput) -> list[list]:
        """
        Transform structured input data into model-ready format.

        Args:
            input_data: Typed feature payload.

        Returns:
            A two-dimensional list matching the pipeline’s expected order of
            features.
        """
        features = PipelineInput.get_features()
        input_dict = input_data.model_dump()
        return [[input_dict[feature] for feature in features]]
