import numpy as np
import tensorflow as tf

from src.layers.subclass.nodes import FillNA
from src.common.config import ConfigLoader, get_config_path
from src.common.logging import logger

config_loader = ConfigLoader(get_config_path())
config_loader.validate_dtypes()
features_config = config_loader.get_config("features")
columns_config = config_loader.get_config("columns")


class ActionSpaceCache:
    """
    A simple cache for action space calculations to avoid recomputing
    the same action space multiple times.
    """

    _cache = {}

    @classmethod
    def get_or_compute(cls, dataset: tf.data.Dataset, cache_key: str = None) -> dict:
        """
        Get cached action space or compute and cache it.

        Args:
            dataset (tf.data.Dataset): The dataset containing the actions.
            cache_key (str, optional): A unique key for caching. If None, uses
                a hash of the dataset. Defaults to None.

        Returns:
            dict: The action space dictionary.
        """
        if cache_key is None:
            cache_key = "default" # default key since datasets are not easily hashable

        if cache_key not in cls._cache:
            logger.info(f"Computing action space (cache key: {cache_key})...")
            cls._cache[cache_key] = calc_action_space(dataset)
        else:
            logger.info(f"Using cached action space (cache key: {cache_key})...")

        return cls._cache[cache_key]

    @classmethod
    def clear(cls):
        """Clear the action space cache."""
        cls._cache.clear()
        logger.info("Action space cache cleared.")


def calc_action_space(dataset: tf.data.Dataset) -> dict:
    """
    Calculate the action space, actions, and actions mapping
    Args:
        dataset (tf.data.Dataset): The dataset containing the actions
    Example:
        Action Mapping
        >> {0: b"coin_magnet", 1: b"coin_multiplier", ...}
    """
    logger.info("Calculating action space...")
    
    # Extract action and label
    mapped_ds = dataset.map(
        lambda row_data: (
            row_data[features_config["action_weight_column"]],
            tf.cast(row_data[features_config["label_column"]], tf.int32)
        ),
        name="actions_labels_map",
    )
    
    actions_list = []
    labels_list = []
    
    for action_batch, label_batch in mapped_ds.as_numpy_iterator():
        actions_list.append(action_batch)
        labels_list.append(label_batch)
        
    actions_np = np.concatenate(actions_list)
    labels_np = np.concatenate(labels_list)

    unique_actions, action_counts = np.unique(actions_np, return_counts=True)
    actions_mapping = {i: action for i, action in enumerate(unique_actions)}
    action_space_size = len(unique_actions)
    
    total_count = len(labels_np)
    pos_count = np.sum(labels_np == 1)
    
    logger.info(f"Total samples: {total_count}, Positive samples: {pos_count}")

    return {
        "actions": dataset.map(lambda x: x[features_config["action_weight_column"]]), # Keep compatible with original return type if needed, or just remove if unused
        "actions_np": actions_np,
        "unique_actions": unique_actions,
        "action_counts": action_counts,
        "actions_mapping": actions_mapping,
        "action_space": unique_actions,
        "action_space_size": action_space_size,
        "total_count": total_count,
        "pos_count": pos_count,
    }


def prep_actions_weights(
    dataset: tf.data.Dataset, cache_key: str = "default"
) -> dict:
    """
    Prepare the weights for the actions, this calculates the sample weight based on
    frequency of reward showing up. Uses cached action space if available.
    Args:
        dataset (tf.data.Dataset): The dataset containing the actions.
        cache_key (str): Cache key for action space. Defaults to "default".
    Returns:
        dict: An actions space map dictionary with a mapping action values to weights.
    Note:
        Sample with reward of 1 is 2x the weight of those with reward of 0. This is
        decided based on the distribution of rewards. Then the samples with reward of
         1 is scaled based on their distribution.
    Example:
        Normalized Weights
        >> {b'coin_magnet': 0.12456487023654975, b'coin_multiplier': 0.12500909512156827,
            b'extra_life': 0.12516036135396683, b'head_start': 0.124164684887546,
            b'nuclear_missle': 0.12570606864806283, b'parachute': 0.12506462323219558,
            b'sparky_armor': 0.1244136040041512, b'time_machine': 0.12591669251595955}

        Actions Weights
        >> {b'coin_magnet': 4.00698030554317, b'coin_multiplier': 3.999854485995665,
            b'extra_life': 3.997436684413578, b'head_start': 4.0134324011409745,
            b'nuclear_missle': 3.9887505366216107, b'parachute': 3.9989664290237927,
            b'sparky_armor': 4.009415476099381, b'time_machine': 3.9854131010989815}
    """
    logger.info("Preparing actions weights...")
    action_space_map = ActionSpaceCache.get_or_compute(dataset, cache_key)

    normalized_counts = (
        action_space_map["action_counts"] / action_space_map["action_counts"].sum()
    )
    value_counts_normalized = dict(
        zip(action_space_map["unique_actions"], normalized_counts)
    )
    logger.info(f"Normalized Weights: {value_counts_normalized}")

    action_space_map["weights"] = {
        key: (2 / value) ** 0.5 for key, value in value_counts_normalized.items()
    }

    # Calculate global class imbalance multiplier
    total_count = action_space_map.get("total_count", 0)
    pos_count = action_space_map.get("pos_count", 0)
    
    if total_count > 0 and pos_count > 0:
        # Calculate ratio of negatives to positives
        neg_count = total_count - pos_count

        # Dynamic Weighting Fix: Global Class Imbalance Multiplier
        imbalance_multiplier = neg_count / pos_count
        logger.info(f"Class imbalance multiplier (Neg/Pos): {imbalance_multiplier:.2f}")
        
        for key in action_space_map["weights"]:
             action_space_map["weights"][key] *= imbalance_multiplier
    
    # Set the weight of the null action to 0.0
    null_action = columns_config["presented_powerup"]["default"].encode("utf-8")
    if null_action in action_space_map["weights"]:
        logger.info(
            f"Null Actions found in weights: {null_action}, setting weight to 0.0"
        )
        action_space_map["weights"][null_action] = 0.0

    logger.info(f"Actions Weights: {action_space_map['weights']}")

    return action_space_map


def compute_sample_weight(
    presented_powerup: tf.Tensor, label: tf.Tensor, actions_weight: dict
) -> tf.Tensor:
    """
    Create a sample weight tensor based on the presented powerup (action) based on a
    weight.
    E.g.: Sample weight = 1.0 if label==0, else actions_weight looked up by action key.
    Args:
        presented_powerup (tf.Tensor): Tensor of presented_powerup values.
        label (tf.Tensor): Tensor of reward values.
        actions_weight (dict): Mapping of actions to weights.
    Returns:
        tf.Tensor: Computed sample weights.
    """
    logger.info(f"Computing Actions Weight: {actions_weight}")

    keys = tf.constant(list(actions_weight.keys()), dtype=tf.string)
    values = tf.constant(list(actions_weight.values()), dtype=tf.float32)
    weight_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys, values),
        default_value=1.0,
    )

    fill_na_layer = FillNA(name="fill_na")
    presented_powerup = fill_na_layer(tf.cast(presented_powerup, tf.string))
    action_weight = weight_table.lookup(presented_powerup)

    # if label == 0 => weight=1.0, else weight=action_weight
    label_int = tf.cast(label, tf.int32)
    weight = tf.where(
        tf.equal(label_int, 0),
        tf.constant(1.0, dtype=tf.float32),
        action_weight,
    )
    return weight
