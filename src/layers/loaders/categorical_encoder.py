import tensorflow as tf
from src.common.logging import logger
from src.layers.subclass.nodes import DynamicCategoryEncoding
from src.common.config import ConfigLoader, get_config_path

config_loader = ConfigLoader(get_config_path())
config_loader.validate_dtypes()
train_config = config_loader.get_config("training")


class SqueezeLayer(tf.keras.layers.Layer):
    """
    A custom layer that squeezes a dimension from the input tensor.
    This replaces Lambda layers for better serialization compatibility with Keras 3.
    """

    def __init__(self, axis=-1, **kwargs):
        """
        Initialize the SqueezeLayer.

        Args:
            axis (int): The axis to squeeze. Defaults to -1.
            **kwargs: Additional keyword arguments passed to the parent Layer class.
        """
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        """Squeeze the specified axis from the input tensor."""
        return tf.squeeze(inputs, axis=self.axis)

    def get_config(self):
        """Get the configuration dictionary for serialization."""
        config = super().get_config()
        config.update({"axis": self.axis})
        return config


def create_one_hot_encoding_layer(
    name: str, dataset: tf.data.Dataset, dtype: str = "int", max_tokens: int = None
) -> tf.keras.layers.Layer:
    """
    Create a one-hot encoding layer for a categorical feature.
    Args:
        name (str): The feature name to be encoded (key in the features dict).
        dataset (tf.data.Dataset): The dataset yielding triplets (features, label, sample_weight).
        dtype (str): Either "string" or "int"/"integer" indicating the feature data type.
        max_tokens (int): The maximum number of unique values for the lookup.
    Returns:
        tf.keras.layers.Layer: Return a small callable (Lmabda) that takes a tensor
        of feature values, passes them through 'index' -> 'encoder', returning one-hot vectors.
    Note:
        This approach can be more efficient than just using Lookup, in scenarios
        where you need to handle multiple encoding schemes or large datasets.
    """
    # maps string/integer to an integer ID
    if dtype.lower() == "string":
        index = tf.keras.layers.StringLookup(
            max_tokens=max_tokens,
            output_mode="int",
            name=name + "_string_lookup",
        )
    elif dtype.lower() == "int":
        index = tf.keras.layers.IntegerLookup(
            max_tokens=max_tokens,
            output_mode="int",
            name=name + "_integer_lookup",
        )
    else:
        raise ValueError(f"Unsupported data type: {dtype}. Must be 'string' or 'int'.")

    feature_ds = (
        dataset.map(
            lambda features, label, sample_weight: features[name],
            name=f"{name}_encoder_map",
        )
        .unbatch()
        .prefetch(tf.data.AUTOTUNE)
    )

    feature_ds_subset = feature_ds.take(train_config["adaption_batch_size"])

    logger.warning(
        f"Adapting index for feature: {name} to learn unique tokens from the feature. "
        f"This is a one-time & offline step that builds the vocabulary."
    )
    index.adapt(feature_ds_subset)

    encoder = tf.keras.layers.CategoryEncoding(
        num_tokens=index.vocabulary_size(),
        output_mode="one_hot",
        name=name + "_category_encoder",
    )

    # Use custom SqueezeLayer instead of Lambda for better serialization
    squeeze_layer = SqueezeLayer(axis=-1, name=name + "_squeeze")

    def encode_fn(feature):
        feature = squeeze_layer(feature)
        return encoder(index(feature))

    return encode_fn


def create_dynamic_category_encoding_layer(
    action_space_weighted: dict,
    layer_name: str = "dynamic_category_encoding",
    oov_value: int = None,
    **kwargs,
) -> DynamicCategoryEncoding:
    """
    Factory function to create a DynamicCategoryEncoding layer.
    Args:
        action_space_weighted (dict): A dictionary mapping action space to weights.
        layer_name (str): Name to give the DynamicCategoryEncoding layer.
        oov_value (int): Value to assign for out-of-vocabulary items in the final encoding.
    Returns:
        DynamicCategoryEncoding: A fully instantiated encoding layer ready for usage.
    """
    return DynamicCategoryEncoding(
        action_space_map=action_space_weighted,
        name=layer_name,
        oov_value=oov_value,
        **kwargs,
    )
