# Neural Contextual Bandit: Improvement Plan & Experiment Summary

**Date:** December 22, 2025  
**Context:** resolving `action_accuracy` stagnation at 0.1250.

## 1. Score Justifications (Current Status)

### The "Random Guessing" Baseline (0.1250)
The initial training runs consistently showed a **Balanced Accuracy on Preprocessed Eval Set** of `0.1250`.
*   **Cause:** There are 8 distinct actions (powerups). Random guessing yields an accuracy of $1/8 = 0.125$.
*   **Diagnosis:** The model was failing to learn any meaningful relationship between user context features and rewards. Instead, it collapsed to a trivial solution.

### The "Zero Loss" Anomaly
Training logs reported `loss: 0.0000e+00` despite the model clearly not performing well.
*   **Cause:** **Severe Class Imbalance**. The original synthetic data had a Click-Through Rate (CTR) of only **~3.25%**.
*   **Mechanism:** In a dataset where 96.75% of labels are `0` (No Click), the Mean Squared Error (MSE) is minimized by simply predicting `0.0` for every input.
*   **Result:** The loss contribution from the rare positive examples was mathematically negligible compared to the mass of negative examples, even with the previous weighting scheme (which only applied a ~4x boost).

---

## 2. Experiments & Fixes Executed

### Experiment 1: Class Imbalance Diagnosis
*   **Action:** Analyzed the raw `training.csv` data distribution.
*   **Finding:** 677,224 negative samples vs. 22,777 positive samples (~30:1 ratio).
*   **Conclusion:** The model had almost no incentive to predict a click.

### Experiment 2: Dynamic Weighting Fix
*   **Action:** Modified `src/utilities/datasets/weights.py`.
*   **Change:** Implemented a **Global Class Imbalance Multiplier**.
    *   Old Logic: Balanced weights based on *action frequency* (how often a powerup was shown).
    *   New Logic: Calculates `Multiplier = Count(Negatives) / Count(Positives)` and applies this factor to all positive sample weights.
*   **Result:** Positive samples now carry ~30x (or ~5x with new data) more weight, forcing the loss function to penalize "missing a click" as heavily as "falsely predicting a click".

### Experiment 3: Data Regeneration
*   **Action:** Modified `src/utilities/datasets/data_generator.py`.
*   **Change:** Increased `base_prob` from `0.02` to `0.10`.
*   **Result:** Regenerated `training.csv`, `validation.csv`, and `test.csv`.
    *   New CTR is approximately **16%**.
    *   Class Imbalance Multiplier dropped from **~30.0** to **~5.20**.
    *   The dataset is now much "healthier" for training a neural network.

---

## 3. Next Improvement Steps

Now that the foundational data and weighting issues are resolved, we can focus on model performance tuning.

### A. Parameter Tuning
1.  **Learning Rate:** The current `0.002` might be too aggressive or too conservative given the new weighting scale.
    *   *Suggestion:* Run a sweep with `0.001`, `0.0005`, and `0.0001`.
2.  **Epochs:** The model was trained for only 5 epochs.
    *   *Suggestion:* Increase to **10-20 epochs** to allow convergence, monitoring `val_loss`.
3.  **Batch Size:** Currently `512`.
    *   *Suggestion:* Experiment with smaller batches (`128` or `256`) to get more frequent gradient updates, which can help escape local minima in bandit problems.

### B. Data & Context
1.  **Feature Engineering:**
    *   Ensure features like `game_day`, `distance_avg`, and `last_run_end_reason` are being normalized/encoded correctly (verified in `preprocessing.py`, but worth double-checking statistics).
2.  **Policy Simulation:**
    *   The current data is generated with `epsilon=1.0` (Pure Random).
    *   *Suggestion:* Generate a dataset with `epsilon=0.3` (Production Logs). This creates "off-policy" data where the logging policy is biased. This is a harder but more realistic scenario for Contextual Bandits.

### C. Code & Metrics
1.  **Evaluation Metric:**
    *   `action_accuracy` (checking if `argmax(Q)` == `Action`) is a very harsh metric, especially if multiple actions have similar probabilities.
    *   *Suggestion:* Implement **AUC-ROC** or **Expected Reward** on the validation set. We want to know if the model *ranks* the good action highly, even if it's not the absolute top #1 every time.
2.  **Loss Function:**
    *   Consider implementing **Focal Loss** for the classification/regression task. It is designed specifically to handle class imbalance by down-weighting easy negatives and focusing on hard positives.
3.  **Model Architecture:**
    *   The current architecture is a simple MLP.
    *   *Suggestion:* If underfitting persists, widen the layers (e.g., `512 -> 256 -> 128`). If overfitting, add more `Dropout` or regularization.

