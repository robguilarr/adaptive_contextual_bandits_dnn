"""Preprocessing model creation for feature encoding and normalization."""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Concatenate
from src.layers.dynamic_category_encoding import FillNA
from src.layers.numerical_encoder import create_normalization_layer
from src.layers.categorical_encoder import (
    create_one_hot_encoding_layer,
    create_dynamic_category_encoding_layer,
)
from src.common.logging import logger
from src.common.config import ConfigLoader, get_config_path

config_loader = ConfigLoader(get_config_path())
config_loader.validate_dtypes()


def create_preprocessing_submodel(
    input_cols: dict,
    action_col: str,
    dataset: tf.data.Dataset,
    action_space_weighted: dict,
) -> tf.keras.Model:
    """
    Build a functional sub-model by creating and adapting the preprocessing layers.
    Args:
        input_cols (dict): A dictionary mapping feature names to data types.
        action_col (str): The column name for the action.
        dataset (tf.data.Dataset): The dataset to adapt the layers.
    Returns:
        tf.keras.Model: A functional sub-model for preprocessing.
    """
    logger.info("1.1 - Taking raw columns as separate Input(...) layers")
    inputs_dict = {}
    for colname, colinfo in input_cols.items():
        inputs_dict[colname] = Input(name=colname, shape=(1,), dtype=colinfo["dtype"])

    logger.info("1.2 - Normalization or One-hot")
    # Get feature types from config if available
    features_config = config_loader.get_config("features")
    feature_types = features_config.get("feature_types", {})
    numerical_features = set(feature_types.get("numerical", []))
    categorical_features = set(feature_types.get("categorical", []))

    numeric_layers = {}
    string_layers = {}
    for colname, colinfo in input_cols.items():
        if colname == action_col:
            continue

        # EXPLICIT mapping: Check feature type from config first
        if colname in numerical_features:
            numeric_layers[colname] = create_normalization_layer(colname, dataset)
        elif colname in categorical_features:
            if colinfo["dtype"] == tf.string:
                string_layers[colname] = create_one_hot_encoding_layer(
                    name=colname,
                    dataset=dataset,
                    dtype="string",
                )
            elif colinfo["dtype"] in [tf.int32, tf.int64]:
                string_layers[colname] = create_one_hot_encoding_layer(
                    name=colname,
                    dataset=dataset,
                    dtype="int",
                )
        
        # Fallback mapping: use dtype if not in feature_types config
        elif colinfo["dtype"] in [tf.float32, tf.float64]:
            numeric_layers[colname] = create_normalization_layer(colname, dataset)
        elif colinfo["dtype"] == tf.string:
            string_layers[colname] = create_one_hot_encoding_layer(
                name=colname,
                dataset=dataset,
                dtype="string",
            )
        elif colinfo["dtype"] in [tf.int32, tf.int64]:
            string_layers[colname] = create_one_hot_encoding_layer(
                name=colname,
                dataset=dataset,
                dtype="int",
            )

    logger.info("1.3 - Encoding Actions into Integer IDs")
    action_encoder = create_dynamic_category_encoding_layer(
        action_space_weighted=action_space_weighted,
        layer_name=f"{action_col}_encoder",
    )

    logger.info("2.1 - Building & adapting functional Graph for preprocessing")
    transformed_tensors = []
    for colname, inp in inputs_dict.items():
        if colname == action_col:
            continue
        fillna_layer = FillNA(name=f"fill_na_{colname}")
        filled = fillna_layer(inp)

        colinfo = input_cols[colname]

        # Use the same logic as above to determine if it's numeric or categorical
        if colname in numerical_features:
            norm = numeric_layers[colname](filled)  # shape: (batch, ) if axis=None
            reshape = Reshape((1,))  # reshape so we can concat with one-hot
            reshape.name = f"{colname}_reshape"
            norm = reshape(norm)
            transformed_tensors.append(norm)
        elif colname in categorical_features:
            oh = string_layers[colname](filled)  # shape: (batch, X)
            transformed_tensors.append(oh)
        
        elif colinfo["dtype"] in [tf.float32, tf.float64]:
            norm = numeric_layers[colname](filled)  # shape: (batch, ) if axis=None
            reshape = Reshape((1,))  # reshape so we can concat with one-hot
            reshape.name = f"{colname}_reshape"
            norm = reshape(norm)
            transformed_tensors.append(norm)
        elif colinfo["dtype"] in [tf.string, tf.int32, tf.int64]:
            oh = string_layers[colname](filled)  # shape: (batch, X)
            transformed_tensors.append(oh)

    logger.info("2.2 - Concatenating all non-action features")
    if len(transformed_tensors) > 1:
        concat_features = Concatenate(axis=-1, name="concat_all")(transformed_tensors)
    else:
        concat_features = transformed_tensors[0]

    logger.info("2.3 - Encoding the actions feature")
    action_inp = inputs_dict[action_col]
    fillna_layer = FillNA(name=f"fill_na_{action_col}")
    action_filled = fillna_layer(action_inp)
    action_id = action_encoder(action_filled)  # shape: (batch, ) or (batch, 1)

    logger.info("3.1 - Building the sub-model")
    submodel = Model(
        inputs=list(inputs_dict.values()),
        outputs=[concat_features, action_id],
        name="preprocessing_submodel",
    )

    action_mapping = action_encoder.get_integer_mapping()
    logger.info(f"3.2 - Saving Action Mapping: {action_mapping}")
    submodel.action_mapping = action_mapping

    return submodel
