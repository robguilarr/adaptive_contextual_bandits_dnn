"""Train the Neural Bandit Model and save it to storage."""

import tensorflow as tf
import datetime
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
        "tensorboard": validate_create_dir(base_dir, "tensorboard"),
    }
    return directories


def setup_callbacks(
    export_config: dict,
    eval_dataset: tf.data.Dataset,
    eval_config: dict = None,
    log_dir: str = None,
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
                log_dir=log_dir,  # Pass log_dir to ValidationCallback for logging
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

    if log_dir:
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq="epoch",
                profile_batch=0,
            )
        )

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


class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Binary Focal Loss for imbalanced classification/regression tasks.
    Adapted for 0/1 labels.
    """

    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        return tf.keras.losses.binary_focal_crossentropy(
            y_true,
            y_pred,
            gamma=self.gamma,
            alpha=self.alpha,
            from_logits=self.from_logits,
        )


def create_bandit_model(
    preproc_model: tf.keras.Model, output_dim: int, train_config: dict
) -> NeuralBanditModel:
    """Initializes and compiles the Neural Bandit model"""

    # Check for optional loss configuration
    loss_function_name = train_config.get("loss_function", "mse")
    output_activation = "relu"  # Default for MSE

    if loss_function_name == "focal":
        # Focal loss requires logits (linear output) or probabilities (sigmoid)
        output_activation = "linear"
        loss_fn = BinaryFocalLoss(
            gamma=train_config.get("focal_loss_gamma", 2.0),
            alpha=train_config.get("focal_loss_alpha", 0.25),
            from_logits=True,
        )
        logger.info(
            f"Using Focal Loss (gamma={loss_fn.gamma}, alpha={loss_fn.alpha}) with Linear Output"
        )
    else:
        loss_fn = tf.keras.losses.MeanSquaredError()
        logger.info("Using Mean Squared Error Loss with ReLU Output")

    model = NeuralBanditModel(
        preprocessing_submodel=preproc_model,
        output_dim=output_dim,
        output_activation=output_activation,
    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        train_config["initial_learning_rate"],
        decay_steps=train_config["lr_decay_steps"],
        decay_rate=train_config["lr_decay_rate"],
        staircase=train_config["lr_staircase"],
    )

    from_logits_metric = loss_function_name == "focal"

    metrics = [
        tf.keras.metrics.MeanSquaredError(name="MSE"),
        tf.keras.metrics.MeanAbsoluteError(name="MAE"),
        tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
        tf.keras.metrics.AUC(name="auc", from_logits=from_logits_metric),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss_fn,
        metrics=metrics,
        run_eagerly=False,
    )

    return model


def trace_mlp_graph(
    train_dataset: tf.data.Dataset,
    preproc_model: tf.keras.Model,
    bandit_model: NeuralBanditModel,
    log_dir,
):
    """
    Traces and exports the MLP (Q-network) graph to TensorBoard.

    Args:
        train_dataset: Training dataset to get sample batch
        preproc_model: Preprocessing model to transform features
        bandit_model: Neural bandit model containing the Q-network
        log_dir: Directory path for TensorBoard logs
    """
    try:
        sample_batch = next(iter(train_dataset))
        features, _, _ = sample_batch

        # Run preproc to get MLP inputs
        concat_features, _ = preproc_model(features, training=False)

        # Define a function to trace JUST the MLP
        @tf.function
        def trace_mlp(inputs):
            return bandit_model.qnet(inputs, training=False)

        # Trace and export
        mlp_log_dir = log_dir / "mlp_graph"
        writer = tf.summary.create_file_writer(str(mlp_log_dir))
        with writer.as_default():
            tf.summary.trace_on(graph=True, profiler=False)
            trace_mlp(concat_features)
            tf.summary.trace_export(
                name="MLP_QNetwork", step=0, profiler_outdir=str(mlp_log_dir)
            )
        logger.info(f"MLP Graph trace exported to {mlp_log_dir}")

    except Exception as e:
        logger.warning(f"Failed to explicitly trace MLP graph: {e}")


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

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = directories["tensorboard"] / timestamp

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
    # Log the output shape of preprocessing model for debugging
    if hasattr(preproc_model, "output_shape"):
        logger.info(f"Preprocessing model output shape: {preproc_model.output_shape}")

    bandit_model = create_bandit_model(preproc_model, output_dim, train_config)

    logger.info("--- Configuring Callbacks ---")
    callbacks = setup_callbacks(
        export_config, eval_dataset, eval_config, log_dir=str(log_dir)
    )

    # Manual trace for MLP only (tensorboard use)
    trace_mlp_graph(train_dataset, preproc_model, bandit_model, log_dir)

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
