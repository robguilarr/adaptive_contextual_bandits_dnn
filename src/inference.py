"""Inference script for running predictions on a trained Neural Bandit model"""

import argparse
from datetime import datetime
import tensorflow as tf
from pathlib import Path
import pandas as pd
from src.common.logging import logger
from src.models.neural_bandit import NeuralBanditModel
from src.utilities.formats.load import (
    load_model,
    load_inference_dataset,
    setup_preprocessing,
)
from src.common.config import ConfigLoader, get_config_path
from src.utilities.formats.export import validate_create_dir


def split_dataset_into_lists(dataset: tf.data.Dataset) -> tuple[list, list, list]:
    """
    Splits a tf.data.Dataset into separate lists for features, labels, and sample weights
    Args:
        dataset (tf.data.Dataset): The dataset to split.
    Returns:
        tuple: (features_list, labels_list, sample_weights_list)
    """
    features_list, labels_list, sample_weights_list = [], [], []

    for features, labels, sample_weights in dataset.unbatch().as_numpy_iterator():
        features_list.append(features)
        labels_list.append(labels)
        sample_weights_list.append(sample_weights)

    return features_list, labels_list, sample_weights_list


def load_action_mapping(model: NeuralBanditModel) -> dict:
    """Loads action ID to label mapping from the model"""
    if hasattr(model, "preproc_model") and hasattr(
        model.preproc_model, "action_mapping"
    ):
        action_mapping = model.preproc_model.action_mapping
        logger.info(f"Action mapping loaded: {action_mapping}")
        return action_mapping
    else:
        logger.warning("Action mapping not found in the model.")
        return {}


def map_predictions(q_values: list, action_ids: list, action_mapping: dict) -> list:
    """Maps action IDs to their corresponding labels"""
    return [
        (
            action_mapping.get(aid, b"UNKNOWN").decode("utf-8")
            if isinstance(action_mapping.get(aid), bytes)
            else str(action_mapping.get(aid, "UNKNOWN"))
        )
        for aid in action_ids
    ]


def save_inference_results(
    q_values_np, action_id_np, predicted_actions, output_file_path: Path
) -> None:
    """Saves inference results to a CSV file"""
    logger.info("--- Saving Inference Results ---")

    data_for_csv = [
        {
            "sample_index": i,
            "chosen_action_id": aid,
            "chosen_action_label": lbl,
            **{f"qv_{j}": float(val) for j, val in enumerate(qv)},
        }
        for i, (qv, aid, lbl) in enumerate(
            zip(q_values_np, action_id_np, predicted_actions)
        )
    ]

    df = pd.DataFrame(data_for_csv)
    df.to_csv(output_file_path, index=False)

    logger.info(f"||| Inference results saved to: {output_file_path} |||")


def run_inference(config_loader: ConfigLoader, args: argparse.Namespace) -> None:
    """Runs inference on a trained Neural Bandit model"""
    logger.info("--- Loading Dataset and Model ---")

    dataset, action_space_weighted = load_inference_dataset(config_loader, args)
    features_list, labels_list, sample_weights_list = split_dataset_into_lists(dataset)

    features_config = config_loader.get_config("features")
    columns_config = config_loader.get_config("columns")

    bandit_model = load_model(args)
    bandit_model.preproc_model = setup_preprocessing(
        features_config, columns_config, dataset, action_space_weighted
    )

    features_dict = {
        key: tf.convert_to_tensor([row[key] for row in features_list])
        for key in features_list[0]
    }

    logger.info("--- Running Inference ---")
    q_values, action_ids = bandit_model(features_dict)

    q_values_np = q_values.numpy()
    action_id_np = action_ids.numpy().flatten()

    logger.info("--- Loading Action Mapping ---")
    action_mapping = load_action_mapping(bandit_model)

    logger.info("--- Mapping Inference Results ---")
    predicted_actions = map_predictions(q_values_np, action_id_np, action_mapping)

    today = datetime.now().strftime("%Y%m%d")
    output_dir = validate_create_dir(Path(args.output_dir))
    output_file_path = output_dir / f"{today}_inference.csv"

    save_inference_results(
        q_values_np, action_id_np, predicted_actions, output_file_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference.")

    parser.add_argument(
        "--model_path",
        default="models/NeuralBanditModel.keras",
        help="Path to the saved model.",
    )
    parser.add_argument(
        "--input_data",
        default=None,
        help="Path to input data file or any feature source",
    )
    parser.add_argument(
        "--output_dir",
        default="data/processed/inference",
        help="Directory to save inference results.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for inference"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs for inference"
    )
    parser.add_argument(
        "--field_delim", type=str, default=",", help="Delimiter for input data file"
    )
    parser.add_argument(
        "--compression_type",
        type=str,
        default=None,
        help="Compression type for input data file",
    )
    parser.add_argument(
        "--shuffle_buffer",
        type=int,
        default=10000,
        help="Buffer size for shuffling input data",
    )

    args = parser.parse_args()

    config_loader = ConfigLoader(get_config_path())
    config_loader.validate_dtypes()

    run_inference(config_loader, args)
