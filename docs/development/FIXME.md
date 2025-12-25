# Neural Bandit DNN Model

---

## Roberto - Where you  should to continue?

In the `legacy_code/training.ipynb` notebook, we have a validation callback that both 
balances and evaluates the model's performance. I attempted to shift the balancing 
logic into the preprocessing step. However, this change is problematic because the 
model isn't aware that the data is being balanced at preprocessing—it ends up seeing 
an artificially even distribution of actions (each action is limited to a minimum 
count). As a result, the model consistently predicts each action with roughly `1/8 (0.125)` accuracy.

The solution is to remove the balancing from the preprocessing step and instead 
incorporate it into the training process (via the callback or another 
high-performance mechanism) so that the model can learn from the naturally 
imbalanced data while still being fairly evaluated.

List of next steps:

1. **Remove Balancing from Preprocessing:**  
   Stop enforcing equal sample counts for each action in the preprocessing step, so
   that the model sees the natural data distribution (`src/utilities/datasets/loader.py`).

2. **Reintegrate Balancing During Training/Evaluation:**  
   Rework your validation callback (or design a new high-performance approach) to balance the evaluation data on the fly. This ensures that during training the model learns from the true imbalanced distribution while evaluation remains fair.

3. **Rethink Evaluation Metrics:**  
   Since your model is learning Q-values rather than classifying actions, consider 
   monitoring alternative metrics (e.g., average reward, regret, or other off-policy 
   evaluation metrics) instead of relying solely on action accuracy (`src/models/callback.py`).

4. **Experiment and Iterate:**  
   After completing your course, use your improved understanding to refine the 
   balancing strategy during training and adjust the model architecture or loss 
   function if necessary.

---

Install project and all dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

Evaluate the model:

```bash
python -m src.evaluate --model_path data/artifacts/models/20250215/keras/NeuralBanditModel.keras --model_type keras
```

Train the model configuration at `config/default_config.yaml`:

```bash
python -m src.train
```

Run Inference:

```bash
python -m src.inference \
    --model_path data/artifacts/models/20250215/keras/NeuralBanditModel.keras \
    --input_data data/raw/iap_purchases_test.csv.gz \
    --output_dir data/processed/inference \
    --batch_size 512 \
    --num_epochs 1 \
    --field_delim "|" \
    --compression_type GZIP
```

Visualize the model:

```bash
$ pip install ai-edge-model-explorer
$ model-explorer
```

