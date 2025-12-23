import tensorflow as tf


class NeuralBanditModel(tf.keras.Model):
    def __init__(
        self, preprocessing_submodel: tf.keras.Model, output_dim: int, **kwargs
    ):
        """
        Initialize and build a "Multi-Layer Perceptron" using Preprocessing designed
        as a Functional API, which is adapted to the dataset before building the
        NeuralBanditModel model.
        The network is designed to learn Q-values for a bandit problem, where
        `output_dim` corresponds to the number of distinct actions (e.g., powerups).
        Args:
          preprocessing_submodel: The functional model that outputs (concat_features,
           action_id).
          output_dim: Number of possible actions => dimension of Q-values.
        """
        super().__init__(**kwargs)
        self.preproc_model = preprocessing_submodel
        self.output_dim = output_dim

        self.qnet = tf.keras.Sequential(  # Build a Dense Neural Q-network
            [
                tf.keras.layers.Dense(256, activation="relu", name="hidden_dense_1"),
                tf.keras.layers.Dense(512, activation="relu", name="hidden_dense_2"),
                tf.keras.layers.Dense(512, activation="relu", name="hidden_dense_3"),
                tf.keras.layers.Dense(256, activation="relu", name="hidden_dense_4"),
                tf.keras.layers.Dropout(0.2, name="dropout_1"),
                tf.keras.layers.Dense(128, activation="relu", name="hidden_dense_5"),
                tf.keras.layers.Dense(64, activation="relu", name="hidden_dense_6"),
                tf.keras.layers.Dropout(0.2, name="dropout_2"),
                tf.keras.layers.Dense(32, activation="relu", name="hidden_dense_7"),
                tf.keras.layers.Dense(
                    output_dim, activation="relu", name="output_dense"
                ),  # Q-values output: one per action ("powerup")
            ],
            name="neural_bandit_q_network",
        )

    def call(self, inputs: dict, training: bool = False):
        """
        Forward pass:
         1) Run raw inputs through the preprocessing sub-model, pre-adapted to the
         dataset.
         2) Q-network forward pass on `concat_features`.
         3) Return (q_values, action_id) so we can build bandit logic in prediction
         within the `train_step`.
         Args:
            inputs: A dictionary of raw features
            training (bool): If True, layers like Dropout & Batch Normalization will
            work in training mode; otherwise, they're in inference mode.
        Note:
            Batch Normalization is a pre-processing layers, so we don't need to
            consider the training mode here (pre-adapted).
        """
        concat_features, action_id = self.preproc_model(inputs, training=training)
        q_values = self.qnet(concat_features, training=training)

        return q_values, action_id

    def train_step(self, data: tuple):
        """
        Custom training logic for the bandit approach. Keras calls this method once
        per batch during model.fit().
        Note: the one-hot mask is applied to the Q-values to select the predicted,
        this to compute the loss only for the chosen action.
        Args:
            data (tuple):
                (features, label, sample_weight).
                - features is a dict with multiple columns (distance_avg, coins_spent,
                  geo_country, etc.).
                - label (rewards) is a float tensor (is_powerup_clicked)
                representing the reward for the chosen action.
                - sample_weight (action weights e.g.: 1.0, 4.12, 1.0, 4.34, ...) is
                an optional tensor or None for weighting the loss function.
        """
        features, label, sample_weight = data

        with tf.GradientTape() as tape:
            # Forward pass: get Q-values and integer action_id
            q_values, action_id = self(features, training=True)  # True: training mode

            # Shape assertions
            batch_size = tf.shape(q_values)[0]
            tf.debugging.assert_equal(
                tf.shape(action_id)[0],
                batch_size,
                message="Action ID shape mismatch: action_id batch size must match q_values batch size",
            )
            tf.debugging.assert_equal(
                tf.shape(q_values)[1],
                self.output_dim,
                message=f"Q-values output dim mismatch: expected {self.output_dim}, got {tf.shape(q_values)[1]}",
            )
            tf.debugging.assert_equal(
                tf.shape(label)[0],
                batch_size,
                message="Label shape mismatch: label batch size must match q_values batch size",
            )

            action_id = tf.reshape(action_id, [-1])  # shape=(batch,)
            action_mask = tf.one_hot(
                tf.cast(action_id, tf.int32), depth=self.output_dim
            )  # Bandit logic: build one-hot mask

            # Overwrite predicted Q-values with the actual rewards
            label = tf.reshape(label, [-1, 1])  # shape=(batch,1)
            target = q_values * (1.0 - action_mask) + label * action_mask

            loss = self.compute_loss(  # Compute sample-weighted loss
                features,  # x
                target,  # y
                q_values,  # y_pred
                sample_weight=sample_weight,
                training=True,
            )

        grads = tape.gradient(
            loss, self.trainable_variables
        )  # Apply gradients (backprop)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics - sample_weight is used to compute the weighted loss
        self.compute_metrics(features, target, q_values, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data: tuple):
        """
        Custom evaluation logic, similar to train_step but no gradient updates.
        """
        features, label, sample_weight = data
        q_values, action_id = self(features, training=False)  # False: inference mode

        # Shape assertions
        batch_size = tf.shape(q_values)[0]
        tf.debugging.assert_equal(
            tf.shape(action_id)[0],
            batch_size,
            message="Action ID shape mismatch: action_id batch size must match q_values batch size",
        )
        tf.debugging.assert_equal(
            tf.shape(q_values)[1],
            self.output_dim,
            message=f"Q-values output dim mismatch: expected {self.output_dim}, got {tf.shape(q_values)[1]}",
        )
        tf.debugging.assert_equal(
            tf.shape(label)[0],
            batch_size,
            message="Label shape mismatch: label batch size must match q_values batch size",
        )

        action_id = tf.reshape(action_id, [-1])
        action_mask = tf.one_hot(tf.cast(action_id, tf.int32), depth=self.output_dim)

        label = tf.reshape(label, [-1, 1])
        target = q_values * (1.0 - action_mask) + label * action_mask

        loss = self.compute_loss(
            features, target, q_values, sample_weight=sample_weight, training=False
        )
        self.compute_metrics(features, target, q_values, sample_weight=sample_weight)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        """
        Placing serializable items, excluding preproc_model because it's too complex
        and non-serializable
        """
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        """
        Create an instance with a placeholder for the preprocessing submodel, must
        reattach the actual preprocessing submodel after loading.
        """
        output_dim = config.pop("output_dim")
        return cls(preprocessing_submodel=None, output_dim=output_dim, **config)
