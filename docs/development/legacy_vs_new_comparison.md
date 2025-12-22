# Legacy vs New Approach: Comprehensive Comparison

## Executive Summary

This document compares the legacy NumPy-based approach (`legacy_code/training.ipynb`) with the new Keras preprocessing layers approach (`notebooks/02_model_draft.ipynb`). The analysis covers business logic, technical implementation, mathematical correctness, scaling risks, and provides an improvement plan.

---

## 1. Business Logic Correctness

### ✅ **Correctly Preserved**

1. **Bandit Loss Computation**: Both approaches correctly implement the contextual bandit loss where only the chosen action's Q-value is updated with the observed reward. The mask-based approach in the new implementation is mathematically equivalent.

2. **Sample Weighting**: The action-based weighting formula `(2 / normalized_frequency) ** 0.5` is preserved, ensuring positive rewards are weighted more heavily.

3. **Action Space Discovery**: Both approaches dynamically discover the action space from data, maintaining flexibility for new powerups.

4. **Evaluation Strategy**: The balanced accuracy evaluation on positive rewards is conceptually preserved, though implementation differs.

### ⚠️ **Potential Issues**

1. **ValidationCallback Logic Change**: 
   - **Legacy**: Filters for positive rewards (`is_powerup_clicked == 1`), downsamples to balance action distribution, then evaluates.
   - **New**: Evaluates on ALL samples, then filters. This changes the evaluation metric meaning.
   - **Impact**: The new callback may not accurately measure "can the model predict which action generated a positive reward?"

2. **Action Encoding Consistency**:
   - **Legacy**: Uses pandas categorical encoding which may have different ordering.
   - **New**: Uses TensorFlow lookup tables with explicit vocabulary ordering.
   - **Impact**: Action IDs may differ between approaches, requiring careful mapping.

---

## 2. Technical Implementation Comparison

### Legacy Approach (NumPy/Pandas)

**Strengths:**
- Simple, explicit data flow
- Easy to debug with pandas DataFrames
- Direct control over preprocessing steps

**Weaknesses:**
- Not scalable (loads entire dataset into memory)
- Preprocessing not part of model graph (can't export to TFLite)
- Manual one-hot encoding with pandas
- Hard to parallelize
- No GPU acceleration for preprocessing

### New Approach (Keras Preprocessing Layers)

**Strengths:**
- Scalable: Uses `tf.data.Dataset` for streaming
- Preprocessing integrated into model graph
- Can export to TFLite with preprocessing
- GPU-accelerated preprocessing
- Better for production deployment

**Weaknesses:**
- More complex architecture
- Harder to debug (tensor operations)
- Requires careful handling of dataset adaptation

---

## 3. Mathematical Correctness

### ✅ **Correct Implementations**

1. **Q-Value Update**: Both use the same mask-based update:
   ```python
   target = q_values * (1.0 - action_mask) + label * action_mask
   ```
   This correctly preserves Q-values for non-chosen actions.

2. **Loss Function**: MSE loss with sample weights is correctly applied.

3. **Normalization**: Both use StandardScaler (legacy) / Normalization layer (new), which are mathematically equivalent when adapted on the same data.

4. **One-Hot Encoding**: Both create one-hot vectors, though implementation differs.

### ⚠️ **Potential Mathematical Issues**

1. **Adaptation Data Leakage Risk**:
   - **Issue**: In `create_preprocessing_submodel`, layers are adapted on `train_dataset`, but if the dataset is not properly split before adaptation, validation data could leak into preprocessing statistics.
   - **Current Status**: ✅ Correctly adapted on train_dataset only (line 1036 in notebook)

2. **Sample Weight Computation**:
   - **Legacy**: Computes weights per batch using pandas operations.
   - **New**: Uses TensorFlow lookup tables, which is more efficient but should produce identical results.
   - **Verification Needed**: Ensure the lookup table handles edge cases (OOV actions) correctly.

3. **Integer vs Float Normalization**:
   - **Legacy**: Only normalizes float columns.
   - **New**: Normalizes float columns, but also one-hot encodes integer columns (game_day, distance_avg).
   - **Impact**: This changes the feature representation. Integer columns like `game_day` and `distance_avg` should likely be normalized, not one-hot encoded.

---

## 4. Scaling Risks & Performance Analysis

### Critical Risks

#### 🔴 **HIGH RISK: ValidationCallback Performance**

**Issue**: The new `ValidationCallback` iterates through the ENTIRE evaluation dataset at the end of each epoch.

```python
for batch in self.eval_dataset:  # Iterates ALL batches
    features, label, sample_weight = batch
    q_values, true_action = self.model(features, training=False)
    # ... accumulates predictions
```

**Problems:**
1. **Memory**: Accumulates all predictions in memory (`all_pred`, `all_true` lists)
2. **Time**: For large eval datasets, this can take significant time per epoch
3. **No Early Stopping**: Even if model is clearly overfitting, still evaluates full dataset

**Recommendation:**
- Limit evaluation to a subset (e.g., first N batches or sample)
- Use `tf.data.Dataset.take()` to limit evaluation size
- Consider evaluating only every N epochs

#### 🟡 **MEDIUM RISK: Dataset Adaptation Memory**

**Issue**: In `create_normalization_layer` and `create_one_hot_encoding_layer`, the adaptation process may load significant data into memory.

```python
feature_ds_subset = feature_ds.take(train_config["adaption_batch_size"])
index.adapt(feature_ds_subset)
```

**Current Mitigation**: ✅ Uses `adaption_batch_size` config (5000 samples), which is good.

**Recommendation:**
- Monitor memory usage during adaptation
- Consider streaming adaptation for very large vocabularies

#### 🟡 **MEDIUM RISK: Action Space Calculation**

**Issue**: `calc_action_space` loads all actions into memory:

```python
actions_np = np.concatenate(list(actions.as_numpy_iterator()))
```

**Problems:**
- For very large datasets, this could be memory-intensive
- Called multiple times (once for weights, once for preprocessing)

**Recommendation:**
- Cache the action space calculation
- Use streaming approach if dataset is extremely large

#### 🟢 **LOW RISK: Preprocessing Model Serialization**

**Issue**: The preprocessing submodel is not easily serializable with the main model.

**Current Status**: ✅ Handled by saving separately and reattaching (line 1690 in notebook)

**Recommendation:**
- Document the loading procedure clearly
- Consider saving preprocessing config separately

### Performance Improvements Over Legacy

1. **Streaming Data**: Can handle datasets larger than memory
2. **GPU Acceleration**: Preprocessing runs on GPU when available
3. **Batch Processing**: More efficient than row-by-row pandas operations
4. **TFLite Export**: Can deploy preprocessing with model

---

## 5. Detailed Component Analysis

### 5.1 NeuralBanditModel

**Legacy Implementation:**
```python
class NeuralBanditModel(keras.Model):
    def train_step(self, data):
        x, y, sample_weight = data
        states = x['states']
        actions = x['actions']
        
        # Convert to numpy, update target, convert back
        target = y_pred.numpy()
        target[np.arange(states.shape[0]), actions] = y
        target_tensor = tf.convert_to_tensor(target)
```

**New Implementation:**
```python
def train_step(self, data: tuple):
    features, label, sample_weight = data
    q_values, action_id = self(features, training=True)
    
    # Pure TensorFlow operations
    action_mask = tf.one_hot(tf.cast(action_id, tf.int32), depth=self.output_dim)
    target = q_values * (1.0 - action_mask) + label * action_mask
```

**Analysis:**
- ✅ **Better**: New approach uses pure TensorFlow, no numpy conversion
- ✅ **Better**: More efficient (no CPU-GPU transfers)
- ✅ **Correct**: Mathematically equivalent
- ⚠️ **Risk**: Action ID shape handling - ensure `action_id` is correctly reshaped

**Recommendation**: Add shape assertions to catch dimension mismatches early.

### 5.2 create_preprocessing_submodel

**Key Differences:**

1. **Integer Column Handling**:
   - **Legacy**: Treats `game_day` and `distance_avg` as numerical (normalized)
   - **New**: One-hot encodes integer columns
   - **Issue**: This is a **BREAKING CHANGE** in feature representation

2. **Action Encoding**:
   - **Legacy**: Uses pandas categorical codes
   - **New**: Uses `DynamicCategoryEncoding` with lookup table
   - **Status**: ✅ More robust, handles OOV better

3. **Layer Adaptation**:
   - **Legacy**: Adapts on full dataset (memory-intensive)
   - **New**: Adapts on subset (`adaption_batch_size`)
   - **Status**: ✅ More scalable

**Critical Issue Found:**
```python
# Line 54-61 in preprocessing.py
elif colinfo["dtype"] in [tf.int32, tf.int64]:  # int => one-hot
    string_layers[colname] = create_one_hot_encoding_layer(
        name=colname,
        dataset=dataset,
        dtype="int",
    )
```

**Problem**: `game_day` and `distance_avg` are integers but should be normalized, not one-hot encoded. One-hot encoding would create thousands of dimensions for these features.

**Recommendation**: 
- Add a feature type configuration (numerical vs categorical)
- Or normalize integer columns that represent continuous values

### 5.3 ValidationCallback

**Legacy Implementation:**
```python
def on_epoch_end(self, epoch, logs=None):
    test_data = read_files_into_df(validation_files)  # Load full dataset
    test_states, test_actions, test_rewards = pre_process(test_data)
    # Filter positive rewards
    positive_test_data = test_data[test_data['is_powerup_clicked']==1]
    # Downsample to balance actions
    down_sampled_positive_test_data = positive_test_data.groupby('action').apply(...)
    # Evaluate
```

**New Implementation:**
```python
def on_epoch_end(self, epoch, logs=None):
    all_pred = []
    all_true = []
    for batch in self.eval_dataset:  # Iterate ALL batches
        features, label, sample_weight = batch
        q_values, true_action = self.model(features, training=False)
        # No filtering for positive rewards!
        pred_action = tf.argmax(q_values, axis=-1)
        all_pred.append(pred_action)
        all_true.append(true_action)
    # No downsampling!
    balanced_accuracy = tf.reduce_mean(tf.cast(tf.equal(all_pred, all_true), tf.float32))
```

**Critical Issues:**

1. **Missing Positive Reward Filter**: The new callback evaluates on ALL samples, not just positive rewards. This changes the metric meaning.

2. **Missing Downsampling**: No action balancing, so the metric is biased toward frequent actions.

3. **Memory Accumulation**: Accumulates all predictions in memory.

**Recommendation**: Restore the positive reward filtering and downsampling logic from the legacy implementation.

---

## 6. Improvement Plan

### Priority 1: Critical Fixes

1. **Fix ValidationCallback** (🔴 HIGH)
   - Add positive reward filtering
   - Add action downsampling for balanced evaluation
   - Limit evaluation to subset of batches
   - File: `src/models/callback.py`

2. **Fix Integer Column Handling** (🔴 HIGH)
   - Add feature type configuration (numerical vs categorical)
   - Normalize integer columns that represent continuous values
   - File: `src/models/preprocessing.py`, `config/default_config.yaml`

3. **Add Shape Assertions** (🟡 MEDIUM)
   - Add assertions in `NeuralBanditModel.train_step` and `test_step`
   - Ensure action_id shapes are correct
   - File: `src/models/neural_bandit.py`

### Priority 2: Performance Optimizations

4. **Optimize ValidationCallback** (🟡 MEDIUM)
   - Limit evaluation to first N batches or sample
   - Add config parameter for evaluation subset size
   - File: `src/models/callback.py`, `config/default_config.yaml`

5. **Cache Action Space Calculation** (🟡 MEDIUM)
   - Cache `calc_action_space` results
   - Avoid recomputing for weights and preprocessing
   - File: `src/utilities/datasets/weights.py`

6. **Add Memory Monitoring** (🟢 LOW)
   - Log memory usage during adaptation
   - Add warnings if adaptation batch size is too large
   - File: `src/layers/loaders/*.py`

### Priority 3: Code Quality

7. **Add Unit Tests** (🟡 MEDIUM)
   - Test bandit loss computation
   - Test sample weight calculation
   - Test preprocessing layers
   - File: `tests/test_models.py`, `tests/test_layers.py`

8. **Add Integration Tests** (🟡 MEDIUM)
   - Test end-to-end training pipeline
   - Compare outputs with legacy implementation
   - File: `tests/test_integration.py`

9. **Documentation** (🟢 LOW)
   - Document preprocessing model loading procedure
   - Document action space mapping
   - File: `docs/`

---

## 7. Implementation PRD: Structured Repository Migration

### 7.1 Objectives

Move the working notebook code (`notebooks/02_model_draft.ipynb`) into a production-ready, structured repository with:
- Modular, testable code
- Configuration-driven training
- Proper error handling
- Documentation
- CI/CD readiness

### 7.2 Current State Analysis

**Existing Structure:**
```
src/
├── models/
│   ├── neural_bandit.py          ✅ Exists
│   ├── preprocessing.py          ✅ Exists
│   └── callback.py               ✅ Exists (needs fixes)
├── layers/
│   ├── subclass/nodes.py        ✅ Exists
│   └── loaders/                  ✅ Exists
├── utilities/
│   ├── datasets/                 ✅ Exists
│   └── formats/                  ✅ Exists
├── common/
│   ├── config.py                 ✅ Exists
│   └── logging.py                ✅ Exists
├── train.py                      ✅ Exists
├── evaluate.py                   ✅ Exists
└── inference.py                  ✅ Exists

config/
└── default_config.yaml           ✅ Exists

data/
└── raw/                          ✅ Exists
```

**Status**: ~80% complete. Main gaps are in testing, documentation, and the fixes identified above.

### 7.3 Implementation Plan

#### Phase 1: Critical Fixes (Week 1)

**Tasks:**
1. Fix `ValidationCallback` to match legacy behavior
2. Fix integer column handling in preprocessing
3. Add shape assertions to `NeuralBanditModel`

**Deliverables:**
- Updated `src/models/callback.py`
- Updated `src/models/preprocessing.py`
- Updated `config/default_config.yaml` (add feature types)
- Unit tests for fixed components

#### Phase 2: Testing & Validation (Week 2)

**Tasks:**
1. Create comprehensive unit tests
2. Create integration test comparing with legacy output
3. Add regression tests for critical paths

**Deliverables:**
- `tests/test_models.py` (comprehensive)
- `tests/test_layers.py`
- `tests/test_integration.py`
- Test coverage > 80%

#### Phase 3: Documentation & Polish (Week 3)

**Tasks:**
1. Write API documentation
2. Create user guide
3. Document preprocessing model loading
4. Add code comments

**Deliverables:**
- `docs/api/` directory with API docs
- `docs/user_guide.md`
- `docs/preprocessing_loading.md`
- Inline code documentation

#### Phase 4: Performance Optimization (Week 4)

**Tasks:**
1. Optimize ValidationCallback evaluation
2. Cache action space calculation
3. Add memory monitoring
4. Profile and optimize bottlenecks

**Deliverables:**
- Optimized callback
- Performance benchmarks
- Memory usage reports

### 7.4 File Structure (Final)

```
adaptive_contextual_bandits_dnn/
├── config/
│   └── default_config.yaml          # Main configuration
├── data/
│   ├── raw/                          # Raw CSV files
│   ├── processed/                    # Processed data
│   └── artifacts/                    # Model artifacts
├── docs/
│   ├── api/                          # API documentation
│   ├── user_guide.md
│   ├── preprocessing_loading.md
│   └── analysis/                     # This document
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neural_bandit.py
│   │   ├── preprocessing.py
│   │   └── callback.py
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── subclass/
│   │   └── loaders/
│   ├── utilities/
│   │   ├── datasets/
│   │   └── formats/
│   ├── common/
│   │   ├── config.py
│   │   ├── constants.py
│   │   └── logging.py
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   └── inference.py                  # Inference script
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_layers.py
│   ├── test_integration.py
│   └── fixtures/                     # Test data
├── notebooks/
│   └── 02_model_draft.ipynb          # Development notebook
├── legacy_code/                      # Legacy reference
├── README.md
├── requirements.txt
├── pyproject.toml
└── Makefile
```

### 7.5 Configuration Enhancements

**Add to `config/default_config.yaml`:**

```yaml
features:
  # ... existing ...
  feature_types:  # NEW
    numerical:
      - distance_avg
      - coins_spent
      - game_day
    categorical:
      - geo_country
      - device_os
      - last_run_end_reason

evaluation:
  validation_callback:
    enabled: true
    max_batches: 100  # Limit evaluation batches
    filter_positive_rewards: true  # Only evaluate on positive rewards
    downsample_actions: true  # Balance action distribution
    downsample_seed: 42
```

### 7.6 Testing Strategy

**Unit Tests:**
- Test each component in isolation
- Mock dependencies
- Test edge cases

**Integration Tests:**
- Test full training pipeline
- Compare outputs with legacy implementation
- Test model loading/saving

**Regression Tests:**
- Test that fixes don't break existing functionality
- Test on small dataset for quick feedback

**Performance Tests:**
- Benchmark training time
- Benchmark memory usage
- Compare with legacy approach

### 7.7 Migration Checklist

- [ ] Fix ValidationCallback
- [ ] Fix integer column handling
- [ ] Add shape assertions
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] Update configuration
- [ ] Write documentation
- [ ] Performance optimization
- [ ] Code review
- [ ] Final validation against legacy

### 7.8 Success Criteria

1. **Functional**: Model produces equivalent results to legacy approach
2. **Performance**: Training time within 20% of legacy (accounting for scalability improvements)
3. **Quality**: Test coverage > 80%
4. **Documentation**: All public APIs documented
5. **Maintainability**: Code follows best practices, passes linting

---

## 8. Recommendations Summary

### Immediate Actions (This Week)

1. **Fix ValidationCallback** - Restore positive reward filtering and downsampling
2. **Fix Integer Column Handling** - Normalize continuous integer features
3. **Add Configuration for Feature Types** - Explicitly define numerical vs categorical

### Short-term (This Month)

4. Optimize ValidationCallback performance
5. Add comprehensive tests
6. Document preprocessing model loading

### Long-term (Next Quarter)

7. Performance benchmarking
8. Advanced optimizations
9. Production deployment guide

---

## 9. Conclusion

The new approach is **directionally correct** and represents a significant improvement in scalability and production-readiness. However, there are **critical issues** that must be addressed:

1. **ValidationCallback** does not match legacy behavior
2. **Integer column handling** incorrectly one-hot encodes continuous features
3. **Performance risks** in evaluation callback

Once these are fixed, the new approach will be superior to the legacy implementation in all aspects: scalability, maintainability, and deployment readiness.

**Overall Assessment**: ✅ **Approved with Critical Fixes Required**

---

*Document Version: 1.0*  
*Last Updated: 2025-02-15*  
*Author: AI Assistant (Neural Contextual Bandits Expert)*

