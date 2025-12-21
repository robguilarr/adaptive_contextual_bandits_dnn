import tensorflow as tf


class ValidationCallback(tf.keras.callbacks.Callback):
    """
    A custom Keras callback for fast evaluation of a model's performance on a
    preprocessed evaluation dataset.

    This callback uses the balanced accuracy from the evaluation dataset at the end of
    each epoch. It assumes that the evaluation dataset yields tuples (features,
    label, sample_weight) and that the model returns a tuple (q_values, true_action)
    when called. The balanced accuracy is printed and stored in the logs under the
    key "action_accuracy".
    """

    def __init__(self, eval_dataset: tf.data.Dataset, **kwargs):
        """
        Initializes the ValidationCallback.

        Args:
            eval_dataset (tf.data.Dataset): A TensorFlow Dataset yielding batches of
            evaluation data.
            **kwargs: Additional keyword arguments passed to the parent Callback class.
        """
        super().__init__(**kwargs)
        self.eval_dataset = eval_dataset

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to compute and log the balanced accuracy on
        the evaluation dataset.

        The method performs the following steps:
            1. Iterates over the evaluation dataset to perform inference and gather
            predicted and true actions.
            2. Concatenates the predictions and ground truth across batches.
            3. Computes the balanced accuracy as the mean of correct predictions.
            4. Prints the balanced accuracy and adds it to the logs dictionary.

        Args:
            epoch: The current epoch number.
            logs: A dictionary for logging metrics. If None, a new dictionary is created.
        """
        logs = logs or {}
        all_pred = []
        all_true = []

        # take the first batch of the evaluation dataset
        for batch in self.eval_dataset:
            features, label, sample_weight = batch

            q_values, true_action = self.model(features, training=False)
            pred_action = tf.argmax(q_values, axis=-1, output_type=tf.int32)
            all_pred.append(pred_action)

            true_action = tf.cast(tf.reshape(true_action, [-1]), tf.int32)
            all_true.append(true_action)

        all_pred = tf.concat(all_pred, axis=0)
        all_true = tf.concat(all_true, axis=0)

        correct = tf.cast(tf.equal(all_pred, all_true), tf.float32)
        # fraction of samples where the predicted action matches the ground truth
        balanced_accuracy = tf.reduce_mean(correct)

        print(
            f"Epoch {epoch + 1}: Balanced Accuracy on Preprocessed Eval Set: {balanced_accuracy:.4f}"
        )
        logs["action_accuracy"] = balanced_accuracy.numpy()
