import json
from datetime import datetime
import tensorflow as tf
from pathlib import Path
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb
from src.models.neural_bandit import NeuralBanditModel
from src.common.config import ConfigLoader, get_config_path
from src.common.logging import logger

config_loader = ConfigLoader(get_config_path())
config_loader.validate_dtypes()
features_config = config_loader.get_config("features")


def validate_create_dir(directory: Path | str, sub_dir: str = "") -> Path:
    """
    Validate and create a directory if it doesn't exist.
    Args:
        directory (Path | str): The base directory.
        sub_dir (str, optional): A subdirectory to be created inside the base
        directory.
    Returns:
        Path: The validated directory path.
    """
    directory = Path(directory) / datetime.now().strftime("%Y%m%d") / sub_dir
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def export_tflite_model(
    model: NeuralBanditModel,
    input_cols: dict,
    train_dataset: tf.data.Dataset,
    model_directory: Path | str,
    tflite_model_version: str,
) -> None:
    """
    Convert a trained TensorFlow model to TFLite, generate preprocessing metadata
    (using stats computed from train_dataset and the submodel.action_mapping), and
    attach the metadata to the TFLite model.
    Args:
        model: The trained model (NeuralBanditModel only).
        input_cols: A dictionary of input column definitions.
        train_dataset: A tf.data.Dataset used for training.
        model_directory: Path to save the converted TFLite model and metadata.
        tflite_model_version: Version of the TFLite model.
    """
    model_directory = validate_create_dir(model_directory, "tflite")

    logger.info("1.1 - Extracting sample features for conversion")
    sample_batch = next(iter(train_dataset))
    sample_features = sample_batch[0]  # first batch is used as a representative sample
    sample_input = {
        key: value[:1] for key, value in sample_features.items()
    }  # 1 dim reduction

    logger.info("1.2 - Converting model to TFLite")
    concrete_func = tf.function(model).get_concrete_function(sample_input)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [concrete_func], trackable_obj=model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]  # !!! Enabling TensorFlow Select ops for any ops not supported natively
    tflite_model = converter.convert()

    output_tflite_path = model_directory / "NeuralBanditModel.tflite"
    logger.info(
        f"1.3 - Converting TF model to TFLite and saved at: {output_tflite_path}"
    )
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)

    logger.info("2.1 - Creating preprocessing metadata JSON")
    num_columns = [
        col
        for col, info in input_cols.items()
        if info["dtype"] in [tf.float32, tf.int32, tf.int64]
        and col != features_config["action_weight_column"]
    ]
    cat_columns = [
        col
        for col, info in input_cols.items()
        if info["dtype"] == tf.string and col != features_config["action_weight_column"]
    ]

    logger.info(
        "2.2 - Extracting numerical feature stats from the preprocessing submodel"
    )
    numerical_stats = {}
    for col in num_columns:
        # Try getting Normalization layer from the preprocessing submodel (PRE-COMPUTED, its mean and variance after adapt)
        try:
            norm_layer = model.preproc_model.get_layer(f"{col}_normalizer")
            mean = (
                norm_layer.mean.numpy().tolist() if hasattr(norm_layer, "mean") else 0.0
            )
            variance = (
                norm_layer.variance.numpy().tolist()
                if hasattr(norm_layer, "variance")
                else 0.0
            )
            std = (variance**0.5) if variance else 0.0
            numerical_stats[col] = {"type": "numerical", "mean": mean, "std": std}
        except ValueError:
            # If the layer is not found, fall back to dummy values
            numerical_stats[col] = {"type": "numerical", "mean": 0.0, "std": 0.0}

    logger.info(
        "2.3 - Extracting categorical feature stats from the preprocessing submodel"
    )
    categorical_info = {}
    for col in cat_columns:
        # Try getting vocabulary from lookup layers
        try:
            lookup_layer = model.preproc_model.get_layer(f"{col}_string_lookup")
            vocab = (
                lookup_layer.get_vocabulary()
                if hasattr(lookup_layer, "get_vocabulary")
                else []
            )
            # Ensure all vocabulary entries are strings in UTF-8 format
            vocab = [v.decode("utf-8") if isinstance(v, bytes) else v for v in vocab]
        except ValueError:
            vocab = ["dummy"]
        categorical_info[col] = {"type": "categorical", "all_values": vocab}

    logger.info("2.4 - Extracting action mapping from the preprocessing submodel")
    actions_mapping = model.preproc_model.action_mapping
    output_mapping = [
        val.decode("utf-8") if isinstance(val, bytes) else val
        for val in actions_mapping.values()
    ]

    preprocess_info = {}
    preprocess_info.update(numerical_stats)
    preprocess_info.update(categorical_info)
    preprocess_info["output_mapping"] = output_mapping

    preprocess_json_path = model_directory / "preprocess.json"
    logger.info(
        f"2.5 - Saving preprocessing metadata to JSON at: {preprocess_json_path}"
    )
    with open(preprocess_json_path, "w") as f:
        json.dump(preprocess_info, f, indent=4)

    logger.info("3.1 - Attaching metadata to the TFLite model")
    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = "NeuralBanditModel"
    model_meta.description = (
        "This model outputs Q-values for each powerup action given a user's state. "
        "It uses a preprocessing submodel with normalization for numerical features and "
        "lookup-based encoding for categorical features. The attached JSON file contains "
        "the dynamically computed preprocessing parameters."
    )
    model_meta.version = tflite_model_version

    logger.info("3.2 - Serializing metadata to a buffer")
    builder = flatbuffers.Builder(0)
    builder.Finish(
        model_meta.Pack(builder), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER
    )
    metadata_buf = builder.Output()

    metadata_path = model_directory / "model_metadata.tflite"
    logger.info(f"3.3 - Saving metadata to: {metadata_path}")
    with open(metadata_path, "wb") as f:
        f.write(metadata_buf)

    logger.info("3.4 - Attaching metadata to the TFLite model")
    populator = _metadata.MetadataPopulator.with_model_file(output_tflite_path)
    populator.load_associated_files([preprocess_json_path])
    populator.populate()

    logger.info("--- Metadata successfully attached to the TFLite model ---")


def export_base_model(
    model: tf.keras.Model,
    save_format: str,
    model_directory: Path | str,
    model_name: str = "NeuralBanditModel"
) -> None:
    """
    Convert a trained TensorFlow model
    Args:
        model: The trained model (NeuralBanditModel only).
        save_format: Format to save the model in.
        model_directory: Path to save the converted model.
    """
    try:
        if save_format in ["h5", "keras", "tf"]:
            if save_format == "keras":
                model_directory = validate_create_dir(model_directory, "keras")
                output_model_path = model_directory / f"{model_name}.{save_format}"
                model.save(output_model_path)
            elif save_format == "h5":
                model_directory = validate_create_dir(model_directory, save_format)
                output_model_path = model_directory / f"{model_name}.{save_format}"
                model.save(output_model_path, save_format="h5")
            elif save_format == "tf":
                model_directory = validate_create_dir(model_directory, "tf")
                output_model_path = model_directory / f"{model_name}_tf"
                tf.saved_model.save(model, str(output_model_path))
            logger.info(
                f"TensorFlow model saved in {save_format} format at: {output_model_path}"
            )
        else:
            raise ValueError(
                f"Invalid save format: {save_format}. Use 'keras', 'h5', or 'tf'.")
    except Exception as e:
        logger.error(f"Error saving model in {save_format} format: {e}")
        raise e
