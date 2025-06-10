"""
CLI entry-point for training, evaluating and registering the model.

Run the script from the project root:

    python train.py --data-path data/training.csv

When *--data-path* is omitted the loader falls back to the SQL connection
defined in :pymod:`core.config`.
"""

from argparse import ArgumentParser

from data import DataLoader
from evaluate import ModelEvaluator

from core.pipeline import ModelPipeline


def main(data_path: str | None = None) -> None:
    """
    Train the pipeline, evaluate it and save metrics/artefacts.

    Args:
        data_path: Optional path to a CSV file containing the training dataset.
            When omitted, :class:`data.DataLoader` falls back to the SQL
            connection specified in the project settings.
    """
    data = DataLoader(data_path).load()
    model_pipeline = ModelPipeline()
    model_pipeline.train(data)
    metrics, artifact_files = ModelEvaluator(model_pipeline.pipeline, data).evaluate()
    model_pipeline.model_store.save(metrics=metrics, artifact_files=artifact_files)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data-path", default=None, help="Path to a training CSV file."
    )
    args = parser.parse_args()
    main(data_path=args.data)
