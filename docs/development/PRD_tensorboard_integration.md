# Product Requirements Document: TensorBoard Integration

**Date:** 2025-12-24
**Status:** Draft
**Target Component:** `src/models/` (NeuralBandit, Preprocessing, Callback) & `src/train.py`

## 1. Executive Summary
The objective is to integrate **TensorBoard** into the `adaptive_contextual_bandits_dnn` training pipeline. This will enable real-time visualization of training metrics, model architectures (specifically the Preprocessing and MLP sub-components), and weight distributions. This observability is critical for diagnosing model behavior, optimizing hyperparameters, and ensuring the correct connectivity of the custom `NeuralBanditModel`.

## 2. System Analysis & Dependencies

### 2.1 Current Stack
- **TensorFlow/Keras Version:** `2.17.1` (Confirmed via `pyproject.toml`)
  - *Compatibility:* Full support for `tf.keras.callbacks.TensorBoard`.
- **Architecture:**
  - `NeuralBanditModel` (`src/models/neural_bandit.py`): A subclassed `tf.keras.Model`.
    - Contains `preproc_model` (Functional API).
    - Contains `qnet` (Sequential MLP).
  - **Training Loop:** Standard `model.fit()` in `src/train.py`.
  - **Custom Logic:** `train_step` and `test_step` are overridden.

### 2.2 Feasibility Check
- **Graph Visualization:** Subclassed models (like `NeuralBanditModel`) are often harder to visualize in TensorBoard's "Graphs" plugin than Functional/Sequential models.
  - *Mitigation:* We will rely on `trace_on` / `trace_export` or ensure the `TensorBoard` callback receives a concrete input signature to trace the graph effectively.
- **Metrics Tracking:** The `ValidationCallback` (`src/models/callback.py`) currently computes `action_accuracy`. This needs to be seamlessly logged alongside standard metrics (MSE, AUC).

## 3. Functional Requirements

### 3.1 Metrics Visualization (Scalars)
The system must log the following scalars to `logs/fit/<timestamp>`:
- **Standard Metrics:** Loss (Training/Validation), MSE, MAE, RMSE, AUC.
- **Custom Metrics:** `action_accuracy` from `ValidationCallback`.
- **Hyperparameters:** Learning rate (if using a schedule).

### 3.2 Model Architecture Visualization (Graphs)
The "Graphs" dashboard must display:
- The high-level `NeuralBanditModel` flow.
- The internal structure of the `preprocessing_submodel` (inputs -> transformations -> concatenation).
- The internal structure of the `qnet` (MLP layers).

### 3.3 Weights & Histograms
The system must log histograms for:
- **Trainable Variables:** Weights and biases of the `qnet` layers (`hidden_dense_*`, `output_dense`).
- **Frequency:** Configurable (e.g., every epoch).

## 4. Technical Implementation Plan

### 4.1 Update `src/train.py`
**Goal:** Inject the `TensorBoard` callback into the `setup_callbacks` function.

- **Action:** Add `tf.keras.callbacks.TensorBoard` to the callbacks list.
- **Configuration:**
  ```python
  tf.keras.callbacks.TensorBoard(
      log_dir=log_dir,
      histogram_freq=1,        # Track weight distributions every epoch
      write_graph=True,        # Visualise the graph
      write_images=False,
      update_freq='epoch',
      profile_batch=0          # Disable profiler by default to save overhead
  )
  ```
- **Log Directory Structure:** `data/artifacts/tensorboard/<YYYYMMDD-HHMMSS>` to match existing artifact patterns.

### 4.2 Update `src/models/callback.py`
**Goal:** Ensure `ValidationCallback` plays nicely with TensorBoard.
- **Action:** No major code changes required if `ValidationCallback` writes to the `logs` dictionary in `on_epoch_end`. The `TensorBoard` callback automatically picks up keys added to `logs` by other callbacks.
- **Verification:** Ensure `logs["action_accuracy"]` is being set correctly (already present in current code).

### 4.3 Visualizing Sub-Models (Special Handling)
Since `NeuralBanditModel` is a subclass, the graph might appear as a single "Op" or be incomplete.
- **Strategy:**
  1. Rely on the `TensorBoard` callback's auto-tracing during `model.fit`.
  2. If the sub-models (`preproc` and `qnet`) are not expandable in the viewer, explicitly log their graphs using a file writer summary in `src/train.py` before training starts:
     ```python
     # Example logic to force graph tracing
     writer = tf.summary.create_file_writer(log_dir)
     with writer.as_default():
         # Trace the preprocessing model explicitly
         tf.summary.graph(preproc_model.input, keras_model=preproc_model)
     ```

## 5. Acceptance Criteria
1. **Directory Created:** A `tensorboard/` folder appears in `data/artifacts/` after training.
2. **Scalars Visible:** Opening TensorBoard (`tensorboard --logdir data/artifacts/tensorboard`) shows curves for Loss, MSE, and Action Accuracy.
3. **Graph Visible:** The "Graphs" tab allows drilling down into `preprocessing_submodel` and `qnet`.
4. **Distributions:** The "Distributions" or "Histograms" tab shows weight updates over time for the dense layers.

## 6. Security & Performance
- **Overhead:** TensorBoard logging adds I/O overhead. `histogram_freq` should be kept to `1` (every epoch) or higher (less frequent) to minimize impact.
- **Storage:** Log files can grow large. Implement a clean-up policy or documentation on how to archive old runs.

## 7. Migration Steps
1. Create `docs/development/PRD_tensorboard_integration.md`.
2. Implement changes in `src/train.py`.
3. Run a test training cycle with `make train` (or equivalent).
4. Verify logs with TensorBoard.

