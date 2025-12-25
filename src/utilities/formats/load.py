"""Model loading and preprocessing setup utilities for inference and evaluation."""

import argparse
import tensorflow as tf
from src.common.logging import logger
from src.common.config import ConfigLoader
from src.models.neural_bandit import NeuralBanditModel
from src.utilities.datasets.weights import prep_actions_weights
from src.models.preprocessing import create_preprocessing_submodel
from src.utilities.datasets.loader import features_and_labels


def setup_preprocessing(
    features_config: dict,
    columns_config: dict,
    train_dataset: tf.data.Dataset,
    action_space_weighted: dict,
) -> tf.keras.Model:
    """Creates and reattaches preprocessing submodel to the model"""
    logger.info("--- Reattaching Preprocessing Submodel ---")

    input_cols = {
        colname: colinfo
        for colname, colinfo in columns_config.items()
        if colname
        not in {features_config["label_column"], *features_config["unwanted_cols"]}
    }

    return create_preprocessing_submodel(
        input_cols=input_cols,
        action_col=features_config["action_weight_column"],
        dataset=train_dataset,
        action_space_weighted=action_space_weighted,
    )


def load_model(args: argparse.Namespace) -> tf.keras.Model:
    """Loads the model from the specified directory"""
    logger.info(f"--- Loading Model from {args.model_path} ---")
    try:
        return tf.keras.models.load_model(
            args.model_path,
            custom_objects={"NeuralBanditModel": NeuralBanditModel},
        )
    except Exception as e:
        raise ValueError(f"Error loading model from {args.model_path}: {e}")


def load_inference_dataset(
    config_loader: ConfigLoader, args: argparse.Namespace
) -> tuple[tf.data.Dataset, dict]:
    """
    Load and preprocess the dataset from plain files while matching the training schema.
    Args:
        config_loader (ConfigLoader): Configuration loader object.
        args (argparse.Namespace): Command-line arguments.
    Returns:
        tf.data.Dataset: Loaded dataset.

    """
    columns_config = config_loader.get_config("columns")
    csv_columns = list(columns_config.keys())
    csv_defaults = [columns_config[col]["default"] for col in csv_columns]

    try:
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=args.input_data,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            column_names=csv_columns,
            column_defaults=csv_defaults,
            shuffle_buffer_size=args.shuffle_buffer,
            field_delim=args.field_delim,
            compression_type=args.compression_type,
        )
    except Exception as e:
        logger.error(f"Error loading dataset from {args.input_data}: {e}")
        raise

    action_space_weighted = prep_actions_weights(dataset)
    dataset = dataset.map(
        lambda row_data: features_and_labels(
            row_data, action_space_weighted["weights"]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.prefetch(tf.data.AUTOTUNE), action_space_weighted
