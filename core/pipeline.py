"""
End-to-end scikit-learn training and inference pipeline.

The module defines :class:`ModelPipeline`, a convenience wrapper that combines
data validation, preprocessing, model fitting, MLflow-backed persistence and
online prediction in a single class suitable for both batch and real-time use.
"""

import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from core.model import ModelStore
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
        self.pipeline = self._create_pipeline()
        self.store = ModelStore()
        self._input_example = None

    def _create_pipeline(self) -> Pipeline:
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
            ],
            remainder="passthrough",
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
        This method validates the training data, fits the pipeline, and stores a
        representative input example from the validated feature data for MLflow
        model signature inference.

        Args:
            data: Training dataset containing both features and target column.
        """
        feature_data, target_data = DataValidator.validate_training_data(data)
        self.pipeline.fit(feature_data, target_data)
        self._input_example = feature_data.iloc[0].to_dict()

    def save(self) -> None:
        """
        Save the trained pipeline to the model store.

        This method persists the trained pipeline to MLflow, registering it under the
        name specified in the model store configuration. Any metrics and artifacts
        generated during training are automatically logged alongside the model.

        Raises:
            TypeError: If the pipeline is not a scikit-learn estimator.
        """
        self.store.save(self.pipeline, input_example=self._input_example)

    def predict(self, input_data: PipelineInput) -> PipelineOutput:
        """
        Generate a single prediction.

        Args:
            input_data: Typed feature payload.

        Returns:
            A :class:`core.schemas.PipelineOutput` with the predicted target
            value.
        """
        pipeline = self.store.load()
        PipelineValidator(pipeline, self.pipeline).validate()
        data = self._prepare_input(input_data)
        prediction = pipeline.predict(data)[0]
        return PipelineOutput.from_prediction(prediction)

    @staticmethod
    def _prepare_input(input_data: PipelineInput) -> pd.DataFrame:
        """
        Transform structured input data into model-ready format.

        Args:
            input_data: Typed feature payload.

        Returns:
            A pandas DataFrame containing the input features in the order expected
            by the pipeline.
        """
        features = PipelineInput.get_features()
        input_dict = input_data.model_dump()
        return pd.DataFrame(
            {feature: input_dict[feature] for feature in features}, index=[0]
        )
