import mlflow


def init_mlflow():
    """Initialize MLflow with proper directory structure."""
    # Set tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")

    # Create a default experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name("Default")
        if experiment is None:
            mlflow.create_experiment("Default")
    except Exception as e:
        print(f"Error initializing MLflow: {e}")
        raise


if __name__ == "__main__":
    init_mlflow()
