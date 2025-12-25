"""Custom TensorFlow layers for dynamic category encoding and missing value handling."""

import tensorflow as tf
from src.common.constants import FILLNA_VALUES


class DynamicCategoryEncoding(tf.keras.layers.Layer):
    """
    Custom layer to encode string categories to integer codes using a precomputed vocabulary.
    """

    def __init__(
        self,
        action_space_map: dict,
        name: str = "dynamic_category_encoding",
        oov_value: int = None,
        **kwargs,
    ):
        """
        Initialize a DynamicCategoryEncoding layer with a dictionary from `calc_action_space`.
        Args:
            action_space_map (dict): Must contain "unique_actions" and
            "action_counts". Typically, this is the output of calc_action_space(dataset).
            name (str): Name for the layer.
            oov_value (int): The integer to assign to out-of-vocabulary items.
        """
        super(DynamicCategoryEncoding, self).__init__(name=name, **kwargs)
        if (
            not action_space_map
            or "unique_actions" not in action_space_map
            or "action_counts" not in action_space_map
        ):
            raise ValueError(
                "Invalid or empty action_space_map provided. "
                "Verify 'unique_actions' and 'action_counts' keys exist."
            )

        self.action_data = action_space_map
        self.unique_actions = self.action_data["unique_actions"]
        self.action_counts = self.action_data["action_counts"]
        self.oov_value = oov_value or (len(self.unique_actions) + 2)

        self.action_mapping = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(self.unique_actions, dtype=tf.string),
                values=tf.range(len(self.unique_actions), dtype=tf.int64),
            ),
            num_oov_buckets=self.oov_value,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Encodes string inputs to integer codes.
        Args:
            inputs (tf.Tensor): A 1D or 2D string tensor (batch dimension optional).
        Returns:
            tf.Tensor: Encoded integer tensor. OOV items are forced to `self.oov_value`.
        Note:
            If the bucket is out-of-vocab, it will produce an ID >= len(self.unique_actions).
        """
        encoded = self.action_mapping.lookup(inputs)
        return tf.where(
            encoded < len(self.unique_actions),
            encoded,
            tf.constant(self.oov_value, dtype=encoded.dtype),
        )

    def get_category_mapping(self) -> dict:
        """
        Retrieve the category-to-integer mapping.
        """
        return {action: idx for idx, action in enumerate(self.unique_actions)}

    def get_integer_mapping(self) -> dict:
        """
        Retrieve the integer-to-category mapping.
        """
        return {idx: action for idx, action in enumerate(self.unique_actions)}


class FillNA(tf.keras.layers.Layer):
    def __init__(self, fill_values=None, name=None, **kwargs):
        """
        Initialize the FillNA layer.
        Args:
            fill_values (dict): A dictionary mapping data types to fill values.
            Example: {tf.float32: 0.0, tf.int32: -1, tf.string: ""}
        """
        super(FillNA, self).__init__(name=name, **kwargs)
        self.fill_values = fill_values or FILLNA_VALUES

    def call(self, inputs):
        """
        Perform the forward pass to fill missing values.
        Args:
            inputs (tf.Tensor): Input tensor with potential missing values.
        Returns:
            tf.Tensor: Tensor with missing values filled.
        """
        dtype = inputs.dtype
        fill_value = self.fill_values.get(dtype, 0)

        if dtype == tf.float32:
            return tf.where(tf.math.is_nan(inputs), fill_value, inputs)
        elif dtype.is_integer:
            return tf.where(tf.equal(inputs, 0), fill_value, inputs)
        elif dtype == tf.string:
            return tf.where(tf.equal(inputs, ""), fill_value, inputs)
        else:
            raise ValueError(f"Unsupported data type: {dtype}")

    def get_config(self):
        """
        Return the configuration of the layer for serialization.
        """
        config = super(FillNA, self).get_config()
        config.update({"fill_values": self.fill_values})
        return config

