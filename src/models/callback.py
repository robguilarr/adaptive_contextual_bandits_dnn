import tensorflow as tf
from src.common.config import ConfigLoader, get_config_path

config_loader = ConfigLoader(get_config_path())
config_loader.validate_dtypes()


class ValidationCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback for fast evaluation of a model's performance on a
    preprocessed evaluation dataset.

    This callback uses the balanced accuracy from the evaluation dataset at the end of
    each epoch. It filters for positive rewards (is_powerup_clicked == 1) and
    downsamples actions to ensure balanced evaluation. It assumes that the evaluation
    dataset yields tuples (features, label, sample_weight) and that the model returns
    a tuple (q_values, true_action) when called. The balanced accuracy is printed and
    stored in the logs under the key "action_accuracy".
    """

    def __init__(
        self,
        eval_dataset: tf.data.Dataset,
        max_batches: int = None,
        filter_positive_rewards: bool = True,
        downsample_actions: bool = True,
        downsample_seed: int = 42,
        **kwargs,
    ):
        """
        Initializes the ValidationCallback.

        Args:
            eval_dataset (tf.data.Dataset): A TensorFlow Dataset yielding batches of
            evaluation data.
            max_batches (int, optional): Maximum number of batches to evaluate.
                If None, evaluates all batches. Defaults to None.
            filter_positive_rewards (bool): If True, only evaluate on samples where
                label == 1.0 (positive rewards). Defaults to True.
            downsample_actions (bool): If True, downsample actions to balance the
                evaluation. Defaults to True.
            downsample_seed (int): Random seed for downsampling. Defaults to 42.
            **kwargs: Additional keyword arguments passed to the parent Callback class.
        """
        super().__init__(**kwargs)
        self.eval_dataset = eval_dataset
        self.max_batches = max_batches
        self.filter_positive_rewards = filter_positive_rewards
        self.downsample_actions = downsample_actions
        self.downsample_seed = downsample_seed

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to compute and log the balanced accuracy on
        the evaluation dataset.

        The method performs the following steps:
            1. Limits evaluation to max_batches if specified.
            2. Filters for positive rewards (label == 1.0) if enabled.
            3. Performs inference and gathers predicted and true actions.
            4. Downsamples actions to balance the evaluation if enabled.
            5. Computes the balanced accuracy as the mean of correct predictions.
            6. Prints the balanced accuracy and adds it to the logs dictionary.

        Args:
            epoch: The current epoch number.
            logs: A dictionary for logging metrics. If None, a new dictionary is created.
        """
        logs = logs or {}
        all_pred = []
        all_true = []

        # Limit evaluation batches to prevent OOM / Long wait
        eval_subset = (
            self.eval_dataset.take(self.max_batches)
            if self.max_batches is not None
            else self.eval_dataset
        )

        for batch in eval_subset:
            features, label, sample_weight = batch
            label = tf.reshape(label, [-1])

            # 1. Filter positive rewards (ground truth relevance)
            if self.filter_positive_rewards:
                pos_mask = tf.equal(label, 1.0)
                if not tf.reduce_any(pos_mask):
                    continue

                pos_indices = tf.where(pos_mask)[:, 0]
                pos_features = {
                    key: tf.gather(val, pos_indices) for key, val in features.items()
                }
                pos_label = tf.gather(label, pos_indices)
            else:
                pos_features = features
                pos_label = label

            # 2. Predict only on relevant samples
            q_values, true_action = self.model(pos_features, training=False)
            pred_action = tf.argmax(q_values, axis=-1, output_type=tf.int32)
            true_action = tf.reshape(true_action, [-1])
            true_action = tf.cast(true_action, tf.int32)

            all_pred.append(pred_action)
            all_true.append(true_action)

        if not all_pred:
            print(f"Epoch {epoch + 1}: No positive rewards found in eval subset.")
            logs["action_accuracy"] = 0.0
            return

        # Concatenate all batches
        all_pred = tf.concat(all_pred, axis=0)
        all_true = tf.concat(all_true, axis=0)

        # 3. Downsample to balance actions (Global Balancing)
        # This prevents the metric from being dominated by frequent actions
        if self.downsample_actions:
            # Get unique actions and their counts
            unique_actions, _, counts = tf.unique_with_counts(all_true)
            min_count = tf.reduce_min(counts)

            if min_count == 0:
                print(f"Epoch {epoch + 1}: No samples found after filtering.")
                logs["action_accuracy"] = 0.0
                return

            # Set random seed for reproducibility
            tf.random.set_seed(self.downsample_seed)

            balanced_indices = []
            for i, action in enumerate(unique_actions):
                # Get indices for this action
                action_indices = tf.where(tf.equal(all_true, action))[:, 0]
                # Shuffle and take min_count
                shuffled = tf.random.shuffle(action_indices)
                selected = shuffled[:min_count]
                balanced_indices.append(selected)

            balanced_indices = tf.concat(balanced_indices, axis=0)

            # Select predictions
            balanced_pred = tf.gather(all_pred, balanced_indices)
            balanced_true = tf.gather(all_true, balanced_indices)
        else:
            balanced_pred = all_pred
            balanced_true = all_true

        # Compute accuracy
        correct = tf.cast(tf.equal(balanced_pred, balanced_true), tf.float32)
        balanced_accuracy = tf.reduce_mean(correct)

        print(
            f"Epoch {epoch + 1}: Balanced Accuracy on Preprocessed Eval Set: {balanced_accuracy:.4f}"
        )
        logs["action_accuracy"] = balanced_accuracy.numpy()
