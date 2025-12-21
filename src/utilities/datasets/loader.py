import tensorflow as tf
from src.utilities.datasets.weights import prep_actions_weights, compute_sample_weight
from src.common.config import ConfigLoader, get_config_path
from src.common.logging import logger

config_loader = ConfigLoader(get_config_path())
config_loader.validate_dtypes()
data_config = config_loader.get_config("data")


def features_and_labels(
    row_data: dict, actions_weight: dict
) -> tuple[dict, tf.Tensor, tf.Tensor]:
    """
    Extract features, label, and sample_weight from row_data.
    Args:
        row_data (dict): Row-level data from the dataset.
        actions_weight (dict): Mapping of the weight for each action type.
    Returns:
        tuple: A tuple of features, label, and sample_weight as tensors.
    """
    features_config = config_loader.get_config("features")

    # Remove unwanted columns
    for unwanted_col in features_config["unwanted_cols"]:
        row_data.pop(unwanted_col, None)

    label = tf.cast(row_data.pop(features_config["label_column"], 0.0), tf.float32)

    logger.debug(
        f"Extracted label: {label.numpy() if hasattr(label, 'numpy') else label}"
    )

    presented_powerup = row_data.get(features_config["action_weight_column"], "")

    return (
        row_data,
        label,
        compute_sample_weight(presented_powerup, label, actions_weight),
    )


def load_dataset(
    columns_config: ConfigLoader, file_pattern: str = None
) -> tf.data.Dataset:
    """
    Load and preprocess the dataset from a CSV file.
    Args:
        columns_config (ConfigLoader): The columns' configuration.
        file_pattern (str, optional): File pattern or path to load. Defaults to
        data_config["data_file_path"].
    Returns:
        tf.data.Dataset: Loaded dataset.
    """
    csv_columns = list(columns_config.keys())
    csv_defaults = [columns_config[col]["default"] for col in csv_columns]

    if file_pattern is None:
        file_pattern = data_config["data_file_path"]

    try:
        dataset = tf.data.experimental.make_csv_dataset(
            file_pattern=file_pattern,
            batch_size=data_config["ingest_batch_size"],  # read one row at a time
            num_epochs=data_config[
                "ingest_epochs"
            ],  # single pass (or adjust as needed)
            shuffle=data_config["ingest_shuffle"],
            shuffle_buffer_size=data_config["ingest_buffer_size"],
            column_names=csv_columns,
            column_defaults=csv_defaults,
            field_delim=data_config["field_delim"],
            compression_type=data_config["compression_type"],
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset from {file_pattern}: {e}")
        raise


def create_train_eval_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Create separate training and evaluation datasets from distinct CSV files.
    The evaluation dataset is balanced by ensuring equal representation of positive
    reward samples across actions.
    Returns:
        tuple: (train_dataset, balanced_eval_dataset, action_space_weighted)
    """
    columns_config = config_loader.get_config("columns")

    logger.info("Loading training dataset...")
    train_dataset = load_dataset(
        columns_config, file_pattern=data_config["train_data_file_path"]
    )
    action_space_weighted = prep_actions_weights(train_dataset)

    logger.info("Loading evaluation dataset...")
    eval_dataset = load_dataset(
        columns_config, file_pattern=data_config["eval_data_file_path"]
    )

    logger.info("Mapping features and labels for training dataset...")
    train_dataset = train_dataset.map(
        lambda row_data: features_and_labels(
            row_data, action_space_weighted["weights"]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_dataset = (
        train_dataset.shuffle(buffer_size=data_config["shuffle_buffer_size"])
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    logger.info("Mapping features and labels for evaluation dataset...")
    eval_dataset = eval_dataset.map(
        lambda row_data: features_and_labels(
            row_data, action_space_weighted["weights"]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    logger.info("Creating balanced evaluation dataset...")
    eval_dataset = balance_eval_dataset(
        eval_dataset,
        action_space_weighted["weights"],
        batch_size=data_config["batch_size"],
    )

    return train_dataset, eval_dataset, action_space_weighted


def balance_eval_dataset(
    dataset: tf.data.Dataset, action_weights: dict, batch_size: int = 1024
) -> tf.data.Dataset:
    """
    Filter for positive reward samples, partition by action, and balance the dataset
    by taking an equal number of samples from each action.
    Args:
        dataset (tf.data.Dataset): The evaluation dataset.
        action_weights (dict): Dictionary with action IDs as keys.
        batch_size (int): The batch size to use.
    Returns:
        tf.data.Dataset: A balanced evaluation dataset.
    """
    features_config = config_loader.get_config("features")
    columns_config = config_loader.get_config("columns")
    action_col = features_config["action_weight_column"]

    dataset = dataset.unbatch().cache()
    pos_dataset = dataset.filter(
        lambda features, label, sw: tf.equal(label, 1.0)
    ).cache()

    logger.info("Partitioning positive reward samples by action...")
    actions = list(action_weights.keys())
    ds_per_action = {}
    for action in actions:
        if columns_config[action_col]["default"] == action.decode("utf-8"):
            continue  # Skip null actions
        ds_action = pos_dataset.filter(
            lambda features, label, sw: tf.equal(features[action_col], action)
        ).cache()
        ds_per_action[action] = ds_action

    counts = {}
    for action, ds in ds_per_action.items():
        count = ds.reduce(0, lambda acc, _: acc + 1)
        counts[action] = int(count.numpy())
        logger.info(f"Action {action} has {counts[action]} positive samples")

    if not counts:
        raise ValueError(
            "No positive reward samples found for any action (after excluding 'N/A')"
        )

    min_count = min(counts.values())
    logger.info(f">> Minimum count per action (excluding 'N/A'): {min_count}")

    if min_count < 2:
        raise ValueError(
            "Not enough samples for at least one action to create a balanced dataset"
        )

    balanced_datasets = [ds.take(min_count) for ds in ds_per_action.values()]
    balanced_dataset = balanced_datasets[0]
    for ds in balanced_datasets[1:]:
        balanced_dataset = balanced_dataset.concatenate(ds)

    balanced_dataset = (
        balanced_dataset.shuffle(buffer_size=data_config["balanced_shuffle_buffer_size"])
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return balanced_dataset
