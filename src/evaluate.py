"""Evaluates the trained Neural Bandit model."""

import argparse
from datetime import datetime
from pathlib import Path
from src.common.logging import logger
from src.common.config import ConfigLoader, get_config_path
from src.utilities.formats.load import load_model, setup_preprocessing
from src.utilities.formats.export import validate_create_dir
from src.utilities.datasets.loader import create_train_eval_datasets


def save_evaluation_results(eval_results: list, eval_dir: Path) -> None:
    """Saves model evaluation results to a file."""
    eval_file = eval_dir / "evaluation_results.txt"
    with eval_file.open("w") as f:
        f.write(str(eval_results))

    logger.info(f"Evaluation results saved at: {eval_file}")


def evaluate(config_loader: ConfigLoader, args: argparse.Namespace) -> None:
    """Evaluates the trained Neural Bandit model."""
    logger.info("--- Loading Configurations ---")
    features_config = config_loader.get_config("features")
    columns_config = config_loader.get_config("columns")
    export_config = config_loader.get_config("model_export")

    logger.info("--- Loading and Splitting Dataset ---")
    train_dataset, eval_dataset, action_space_weighted = create_train_eval_datasets()

    model = load_model(args)
    model.preproc_model = setup_preprocessing(
        features_config, columns_config, train_dataset, action_space_weighted
    )

    logger.info("--- Evaluating Model ---")
    eval_results = model.evaluate(eval_dataset)
    logger.info(f"Model evaluation results: {eval_results}")

    if args.model_type in ["keras", "h5"]:
        eval_dir = validate_create_dir(
            export_config["model_directory"], args.model_type
        )
        save_evaluation_results(eval_results, eval_dir)
    else:
        raise ValueError(
            f"Invalid model type '{args.model_type}'. Supported types: keras, h5."
        )

    logger.info("||| Evaluation completed successfully |||")


if __name__ == "__main__":
    DATE = datetime.now().strftime("%Y%m%d")

    parser = argparse.ArgumentParser(description="Evaluate a trained TensorFlow model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=f"data/artifacts/models/{DATE}/keras/NeuralBanditModel.keras",
        help="Path to the model file.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["keras", "h5"],
        default="keras",
        help="Type of model to evaluate.",
    )
    args = parser.parse_args()

    config_loader = ConfigLoader(get_config_path())
    config_loader.validate_dtypes()

    evaluate(config_loader, args)
