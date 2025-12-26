"""Custom Keras callbacks for model validation and monitoring."""

import tensorflow as tf
from src.common.config import ConfigLoader, get_config_path

config_loader = ConfigLoader(get_config_path())
config_loader.validate_dtypes()


class ValidationCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback for fast evaluation of a model's performance on a
    preprocessed evaluation dataset.
    """

    def __init__(
        self,
        eval_dataset: tf.data.Dataset,
        max_batches: int = None,
        filter_positive_rewards: bool = True,
        downsample_actions: bool = True,
        downsample_seed: int = 42,
        log_dir: str = None,
        **kwargs,
    ):
        """
        Initializes the ValidationCallback.
        """
        super().__init__(**kwargs)
        self.eval_dataset = eval_dataset
        self.max_batches = max_batches
        self.filter_positive_rewards = filter_positive_rewards
        self.downsample_actions = downsample_actions
        self.downsample_seed = downsample_seed
        self.writer = (
            tf.summary.create_file_writer(f"{log_dir}/validation") if log_dir else None
        )

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to compute and log the balanced accuracy.
        """
        logs = logs or {}

        all_pred = []
        all_true = []

        # Limit evaluation batches
        eval_subset = (
            self.eval_dataset.take(self.max_batches)
            if self.max_batches is not None
            else self.eval_dataset
        )

        for batch in eval_subset:
            features, label, sample_weight = batch
            label = tf.reshape(label, [-1])

            if self.filter_positive_rewards:
                pos_mask = tf.equal(label, 1.0)
                if not tf.reduce_any(pos_mask):
                    continue

                pos_indices = tf.where(pos_mask)[:, 0]
                pos_features = {
                    key: tf.gather(val, pos_indices) for key, val in features.items()
                }
            else:
                pos_features = features

            q_values, true_action = self.model(pos_features, training=False)
            pred_action = tf.argmax(q_values, axis=-1, output_type=tf.int32)
            true_action = tf.reshape(true_action, [-1])
            true_action = tf.cast(true_action, tf.int32)

            all_pred.append(pred_action)
            all_true.append(true_action)

        if not all_pred:
            balanced_accuracy = 0.0
        else:
            all_pred = tf.concat(all_pred, axis=0)
            all_true = tf.concat(all_true, axis=0)

            if self.downsample_actions:
                unique_actions, _, counts = tf.unique_with_counts(all_true)
                min_count = tf.reduce_min(counts)

                if min_count == 0:
                    balanced_accuracy = 0.0
                else:
                    tf.random.set_seed(self.downsample_seed)
                    balanced_indices = []
                    for action in unique_actions:
                        action_indices = tf.where(tf.equal(all_true, action))[:, 0]
                        shuffled = tf.random.shuffle(action_indices)
                        selected = shuffled[:min_count]
                        balanced_indices.append(selected)

                    balanced_indices = tf.concat(balanced_indices, axis=0)
                    balanced_pred = tf.gather(all_pred, balanced_indices)
                    balanced_true = tf.gather(all_true, balanced_indices)
            else:
                balanced_pred = all_pred
                balanced_true = all_true

            correct = tf.cast(tf.equal(balanced_pred, balanced_true), tf.float32)
            balanced_accuracy = tf.reduce_mean(correct).numpy()

        print(
            f"Epoch {epoch + 1}: Balanced Accuracy on Preprocessed Eval Set: {balanced_accuracy:.4f}"
        )
        logs["action_accuracy"] = balanced_accuracy

        # Explicitly write to TensorBoard
        if self.writer:
            with self.writer.as_default():
                tf.summary.scalar("action_accuracy", balanced_accuracy, step=epoch)

                # Also log other metrics from logs dict if needed, specifically AUC
                if "auc" in logs:
                    tf.summary.scalar("epoch_auc", logs["auc"], step=epoch)
