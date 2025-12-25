"""Numerical feature normalization layers for preprocessing."""

import tensorflow as tf
from src.common.logging import logger
from src.common.config import ConfigLoader, get_config_path

config_loader = ConfigLoader(get_config_path())
config_loader.validate_dtypes()
train_config = config_loader.get_config("training")


def create_normalization_layer(
    name: str, dataset: tf.data.Dataset
) -> tf.keras.layers.Layer:
    """
    Create a normalization layer for a numerical feature.
    Args:
        name (str): The feature name in the `features` dict.
        dataset (tf.data.Dataset): The dataset yielding (features, label,
        sample_weight). Each `features` is a dict that must contain `name`.
    Returns:
        tf.keras.layers.Layer: A `Normalization` layer.
    """
    normalizer = tf.keras.layers.Normalization(
        axis=None,  # or axis=-1 if each feature is a 1D vector to normalize per component
        name=name + "_normalizer",
    )
    feature_ds = (
        dataset.map(
            lambda features, label, sample_weight: features[name],
            name=f"{name}_normalizer_map",
        )
        .unbatch()
        .prefetch(tf.data.AUTOTUNE)
    )

    feature_ds_subset = feature_ds.take(train_config["adaption_batch_size"])

    logger.warning(
        f"Adapting stats for feature: {name} to statistics from the feature. This is "
        f"a one-time & offline step that builds the vocabulary or stats."
    )
    normalizer.adapt(feature_ds_subset)

    return normalizer

