"""Train the Neural Bandit Model and save it to storage."""

import tensorflow as tf
from src.common.config import ConfigLoader, get_config_path
from src.common.logging import logger
from src.utilities.formats.export import (
    export_tflite_model,
    export_base_model,
    validate_create_dir,
)
from src.models.neural_bandit import NeuralBanditModel
from src.models.callback import ValidationCallback
from src.models.preprocessing import create_preprocessing_submodel
from src.utilities.datasets.loader import create_train_eval_datasets


def setup_model_directory(export_config: dict) -> dict:
    """Set up necessary directories for saving model artifacts"""
    base_dir = export_config["model_directory"]
    directories = {
        "plots": validate_create_dir(base_dir, "plots"),
        "checkpoints": validate_create_dir(base_dir, "checkpoints"),
        "summary": validate_create_dir(base_dir, "summary"),
    }
    return directories


def setup_callbacks(
    export_config: dict,
    eval_dataset: tf.data.Dataset,
    eval_config: dict = None,
) -> list:
    """Configures model callbacks including validation and checkpointing"""
    callbacks = []

    # Setup validation callback if enabled
    if eval_config is None:
        eval_config = {}
    validation_config = eval_config.get("validation_callback", {})
    if validation_config.get("enabled", True):
        callbacks.append(
            ValidationCallback(
                eval_dataset,
                max_batches=validation_config.get("max_batches"),
                filter_positive_rewards=validation_config.get(
                    "filter_positive_rewards", True
                ),
                downsample_actions=validation_config.get("downsample_actions", True),
                downsample_seed=validation_config.get("downsample_seed", 42),
            )
        )

    if export_config["checkpoints"]["enabled"]:
        checkpoint_dir = validate_create_dir(
            export_config["model_directory"], "checkpoints"
        )
        checkpoint_args = export_config["checkpoints"]
        state_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir / "NeuralBanditModel_state_checkpoint.keras",
            monitor=checkpoint_args["monitor"],
            save_best_only=checkpoint_args["save_best_only"],
            mode=checkpoint_args["mode"],
        )
        callbacks.append(state_checkpoint)

    return callbacks


def setup_preprocessing_model(
    features_config: dict,
    columns_config: dict,
    train_dataset: tf.data.Dataset,
    action_space: dict,
) -> tuple:
    """Creates and adapts preprocessing layers"""
    input_cols = {
        colname: colinfo
        for colname, colinfo in columns_config.items()
        if colname
        not in {features_config["label_column"], *features_config["unwanted_cols"]}
    }

    preproc_model = create_preprocessing_submodel(
        input_cols=input_cols,
        action_col=features_config["action_weight_column"],
        dataset=train_dataset,
        action_space_weighted=action_space,
    )

    return preproc_model, input_cols, action_space["action_space_size"]


def create_bandit_model(
    preproc_model: tf.keras.Model, output_dim: int, train_config: dict
) -> NeuralBanditModel:
    """Initializes and compiles the Neural Bandit model"""
    model = NeuralBanditModel(
        preprocessing_submodel=preproc_model, output_dim=output_dim
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        train_config["initial_learning_rate"],
        decay_steps=train_config["lr_decay_steps"],
        decay_rate=train_config["lr_decay_rate"],
        staircase=train_config["lr_staircase"],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[
            tf.keras.metrics.MeanSquaredError(name="MSE"),
            tf.keras.metrics.MeanAbsoluteError(name="MAE"),
            tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
        ],
        run_eagerly=False,
    )

    return model


def train_and_save(config_loader: ConfigLoader):
    """Train the Neural Bandit Model and save it to disk"""
    logger.info("--- Loading Configurations ---")
    features_config = config_loader.get_config("features")
    columns_config = config_loader.get_config("columns")
    export_config = config_loader.get_config("model_export")
    train_config = config_loader.get_config("training")
    try:
        eval_config = config_loader.get_config("evaluation")
    except ValueError:
        eval_config = {}

    logger.info("--- Loading Datasets ---")
    train_dataset, eval_dataset, action_space_weighted = create_train_eval_datasets()

    logger.info("--- Setting Up Preprocessing Model ---")
    preproc_model, input_cols, output_dim = setup_preprocessing_model(
        features_config, columns_config, train_dataset, action_space_weighted
    )

    logger.info("--- Setting Up Model Directories ---")
    directories = setup_model_directory(export_config)

    if export_config["plotting"]["enabled"]:
        logger.info("--- Plotting Preprocessing Model ---")
        tf.keras.utils.plot_model(
            preproc_model,
            show_layer_names=export_config["plotting"]["show_layer_names"],
            expand_nested=export_config["plotting"]["expand_nested"],
            show_trainable=export_config["plotting"]["show_trainable"],
            rankdir=export_config["plotting"]["rankdir"],
            show_layer_activations=export_config["plotting"]["show_layer_activations"],
            show_shapes=export_config["plotting"]["show_shapes"],
            to_file=directories["plots"] / "preprocessing_model.png",
        )

    logger.info("--- Initializing Neural Bandit Model ---")
    bandit_model = create_bandit_model(preproc_model, output_dim, train_config)

    logger.info("--- Configuring Callbacks ---")
    callbacks = setup_callbacks(export_config, eval_dataset, eval_config)

    logger.info("--- Training Neural Bandit Model ---")
    logger.info(
        f"steps_per_epoch: {train_config['steps_per_epoch']}, epochs: {train_config['epochs']}"
    )
    bandit_model.fit(
        train_dataset.repeat(),
        epochs=train_config["epochs"],
        steps_per_epoch=train_config[
            "steps_per_epoch"
        ],  # should be = train rows / batch_size
        validation_data=eval_dataset,
        callbacks=callbacks,
    )

    if export_config["model_summary"]["enabled"]:
        logger.info("--- Saving Model Summary ---")
        summary_file = directories["summary"] / "model_summary.txt"
        bandit_model.summary(
            line_length=export_config["model_summary"]["line_length"],
            positions=export_config["model_summary"]["positions"],
            expand_nested=export_config["model_summary"]["expand_nested"],
            print_fn=lambda x: print(x, file=open(summary_file, "a")),
        )

    if export_config["plotting"]["enabled"]:
        logger.info("--- Plotting Neural Bandit Model ---")
        tf.keras.utils.plot_model(
            bandit_model.qnet,
            show_layer_names=export_config["plotting"]["show_layer_names"],
            expand_nested=export_config["plotting"]["expand_nested"],
            show_trainable=export_config["plotting"]["show_trainable"],
            rankdir=export_config["plotting"]["rankdir"],
            show_layer_activations=export_config["plotting"]["show_layer_activations"],
            show_shapes=export_config["plotting"]["show_shapes"],
            to_file=directories["plots"] / "bandit_model.png",
        )

    logger.info("--- Exporting Models ---")
    export_base_model(
        model=bandit_model,
        save_format=export_config["save_format"],
        model_directory=export_config["model_directory"],
        model_name="NeuralBanditModel",
    )
    export_base_model(
        model=preproc_model,
        save_format=export_config["save_format"],
        model_directory=export_config["model_directory"],
        model_name="PreprocessingModel",
    )

    if export_config["tflite_enabled"]:
        logger.info("--- Exporting TFLite Model ---")
        export_tflite_model(
            model=bandit_model,
            input_cols=input_cols,
            train_dataset=train_dataset,
            model_directory=export_config["model_directory"],
            tflite_model_version=export_config["tflite_model_version"],
        )

    logger.info("||| Training and saving completed successfully |||")


if __name__ == "__main__":
    config_loader = ConfigLoader(get_config_path())
    config_loader.validate_dtypes()
    train_and_save(config_loader)
