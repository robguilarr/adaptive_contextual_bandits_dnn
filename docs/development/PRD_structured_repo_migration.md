# Product Requirements Document: Structured Repository Migration

## 1. Overview

### 1.1 Purpose
Migrate the working notebook implementation (`notebooks/02_model_draft.ipynb`) into a production-ready, structured Python package with proper testing, documentation, and configuration management.

### 1.2 Scope
- Refactor notebook code into modular Python modules
- Fix critical bugs identified in comparison analysis
- Add comprehensive testing
- Create documentation
- Set up CI/CD pipeline
- Performance optimization

### 1.3 Success Criteria
- ✅ All critical bugs fixed
- ✅ Test coverage > 80%
- ✅ Documentation complete
- ✅ Model outputs match legacy implementation
- ✅ Training pipeline runs end-to-end
- ✅ Code passes linting and type checking
- ✅ Compatible with TensorFlow 2.17.1 (Keras 3)

---

## 2. Current State

### 2.1 Existing Code Structure

```
src/
├── models/
│   ├── neural_bandit.py          ✅ Complete
│   ├── preprocessing.py          ⚠️ Needs integer column fix & serialization fix
│   └── callback.py               🔴 Needs critical fixes
├── layers/
│   ├── subclass/nodes.py         ✅ Complete
│   └── loaders/                  ✅ Complete
├── utilities/
│   ├── datasets/                 ✅ Complete
│   └── formats/                  ✅ Complete
├── common/
│   ├── config.py                 ✅ Complete
│   └── logging.py                ✅ Complete
├── train.py                      ✅ Complete
├── evaluate.py                   ✅ Exists
└── inference.py                  ✅ Exists
```

### 2.2 Known Issues

**Critical (P0):**
1. ValidationCallback doesn't filter positive rewards
2. ValidationCallback doesn't downsample actions
3. Integer columns incorrectly one-hot encoded

**High (P1):**
4. ValidationCallback performance (evaluates full dataset)
5. Missing shape assertions in NeuralBanditModel
6. Preprocessing serialization issues (Lambda layers)

**Medium (P2):**
7. Action space calculation not cached
8. Missing unit tests
9. Missing integration tests

**Low (P3):**
10. Documentation incomplete
11. Memory monitoring missing

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR1: Fix ValidationCallback
- **Description**: Restore legacy behavior of filtering positive rewards and downsampling actions
- **Acceptance Criteria**:
  - Only evaluates on samples where `is_powerup_clicked == 1`
  - Downsamples each action to minimum count for balanced evaluation
  - Limits evaluation to configurable number of batches
  - Prints balanced accuracy metric
- **Files**: `src/models/callback.py`
- **Priority**: P0

#### FR2: Fix Integer Column Handling
- **Description**: Normalize continuous integer features instead of one-hot encoding
- **Acceptance Criteria**:
  - `game_day` and `distance_avg` are normalized, not one-hot encoded
  - Feature types configurable via YAML
  - Backward compatible with existing configs
- **Files**: `src/models/preprocessing.py`, `config/default_config.yaml`
- **Priority**: P0

#### FR3: Add Shape Assertions
- **Description**: Add assertions to catch dimension mismatches early
- **Acceptance Criteria**:
  - Assertions in `train_step` and `test_step`
  - Clear error messages on shape mismatches
  - Tests verify assertions work
- **Files**: `src/models/neural_bandit.py`
- **Priority**: P1

#### FR4: Optimize ValidationCallback Performance
- **Description**: Limit evaluation to subset of batches
- **Acceptance Criteria**:
  - Configurable `max_batches` parameter
  - Default limits to first 100 batches
  - Memory usage stays bounded
- **Files**: `src/models/callback.py`, `config/default_config.yaml`
- **Priority**: P1

#### FR5: Cache Action Space Calculation
- **Description**: Avoid recomputing action space multiple times
- **Acceptance Criteria**:
  - Action space computed once and cached
  - Used by both weights and preprocessing
  - Cache invalidated on dataset change
- **Files**: `src/utilities/datasets/weights.py`, `src/utilities/datasets/loader.py`
- **Priority**: P2

#### FR6: Robust Preprocessing Serialization
- **Description**: Ensure preprocessing model can be saved/loaded in Keras 3 format
- **Acceptance Criteria**:
  - Replace `Lambda` layers with `Reshape` or custom layers
  - Verify `.keras` export and import works without custom scope issues
- **Files**: `src/layers/loaders/categorical_encoder.py`, `src/models/preprocessing.py`
- **Priority**: P1

### 3.2 Non-Functional Requirements

#### NFR1: Testing
- **Unit Tests**: > 80% code coverage
- **Integration Tests**: Full pipeline test
- **Regression Tests**: Compare with legacy output
- **Performance Tests**: Benchmark training time

#### NFR2: Documentation
- **API Docs**: All public functions documented
- **User Guide**: How to train/evaluate/infer
- **Preprocessing Guide**: How to load preprocessing model
- **Architecture Docs**: System design overview

#### NFR3: Code Quality
- **Linting**: Passes `ruff` or `black` + `flake8`
- **Type Checking**: Passes `mypy` (where applicable)
- **Code Review**: All code reviewed before merge

#### NFR4: Performance
- **Training Time**: Within 20% of legacy (accounting for scalability)
- **Memory Usage**: Bounded, no memory leaks
- **Evaluation Time**: < 5 minutes per epoch for large datasets

---

## 4. Technical Design

### 4.1 ValidationCallback Fix

**Current Implementation:**
```python
def on_epoch_end(self, epoch, logs=None):
    for batch in self.eval_dataset:
        features, label, sample_weight = batch
        q_values, true_action = self.model(features, training=False)
        # No filtering!
        all_pred.append(pred_action)
        all_true.append(true_action)
```

**Fixed Implementation:**
```python
def on_epoch_end(self, epoch, logs=None):
    all_pred = []
    all_true = []
    
    # Limit evaluation batches to prevent OOM / Long wait
    eval_subset = self.eval_dataset.take(self.max_batches)
    
    for batch in eval_subset:
        features, label, sample_weight = batch
        label = tf.reshape(label, [-1])
        
        # 1. Filter positive rewards (ground truth relevance)
        pos_mask = tf.equal(label, 1.0)
        if not tf.reduce_any(pos_mask):
            continue
            
        pos_indices = tf.where(pos_mask)[:, 0]
        pos_features = {
            key: tf.gather(val, pos_indices) 
            for key, val in features.items()
        }
        
        # 2. Predict only on relevant samples
        q_values, true_action = self.model(pos_features, training=False)
        pred_action = tf.argmax(q_values, axis=-1, output_type=tf.int32)
        true_action = tf.reshape(true_action, [-1])
        
        all_pred.append(pred_action)
        all_true.append(true_action)
    
    if not all_pred:
        print(f"Epoch {epoch}: No positive rewards found in eval subset.")
        return

    # Concatenate all batches
    all_pred = tf.concat(all_pred, axis=0)
    all_true = tf.concat(all_true, axis=0)
    
    # 3. Downsample to balance actions (Global Balancing)
    # This prevents the metric from being dominated by frequent actions
    unique_actions, _, count = tf.unique_with_counts(all_true)
    min_count = tf.reduce_min(count)
    
    balanced_indices = []
    for action in unique_actions:
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
    
    # Compute accuracy
    correct = tf.cast(tf.equal(balanced_pred, balanced_true), tf.float32)
    balanced_accuracy = tf.reduce_mean(correct)
    
    print(f"Epoch {epoch}: Balanced Accuracy: {balanced_accuracy:.4f}")
    logs["action_accuracy"] = balanced_accuracy.numpy()
```

### 4.2 Integer Column Handling Fix

**Configuration Addition:**
```yaml
features:
  feature_types:
    numerical:
      - distance_avg
      - coins_spent
      - game_day
    categorical:
      - geo_country
      - device_os
      - last_run_end_reason
```

**Preprocessing Logic:**
```python
def create_preprocessing_submodel(...):
    for colname, colinfo in input_cols.items():
        if colname == action_col:
            continue
        
        # Check feature type from config
        # Use set lookup for O(1)
        if colname in feature_types["numerical"]:
            numeric_layers[colname] = create_normalization_layer(colname, dataset)
        elif colname in feature_types["categorical"]:
            string_layers[colname] = create_one_hot_encoding_layer(...)
        else:
            # Fallback: use dtype
            if colinfo["dtype"] in [tf.float32, tf.float64, tf.int32, tf.int64]:
                numeric_layers[colname] = create_normalization_layer(colname, dataset)
            else:
                string_layers[colname] = create_one_hot_encoding_layer(...)
```

### 4.3 Shape Assertions

```python
def train_step(self, data: tuple):
    features, label, sample_weight = data
    
    with tf.GradientTape() as tape:
        q_values, action_id = self(features, training=True)
        
        # Assertions
        batch_size = tf.shape(q_values)[0]
        # Check explicit shapes if available, otherwise tensor shapes
        tf.debugging.assert_equal(
            tf.shape(action_id)[0], 
            batch_size, 
            message="Action ID shape mismatch"
        )
        tf.debugging.assert_equal(
            tf.shape(q_values)[1], 
            self.output_dim, 
            message="Q-values output dim mismatch"
        )
        
        action_id = tf.reshape(action_id, [-1])
        # ... rest of implementation
```

### 4.4 Action Space Caching

```python
class ActionSpaceCache:
    _cache = {}
    
    @classmethod
    def get_or_compute(cls, dataset, key):
        if key not in cls._cache:
            cls._cache[key] = calc_action_space(dataset)
        return cls._cache[key]
    
    @classmethod
    def clear(cls):
        cls._cache.clear()
```

### 4.5 Serialization & Keras 3 Compatibility (New)

**Issue**: The current implementation uses `Lambda` layers for reshaping/squeezing in `create_one_hot_encoding_layer`. `Lambda` layers are difficult to serialize safely and are discouraged in production Keras pipelines.

**Fix**: Replace `Lambda` with `Reshape` layers or explicit operations within a custom `Layer` subclass.

```python
# Instead of Lambda(lambda x: tf.squeeze(x, axis=-1))
# Use:
reshaper = tf.keras.layers.Reshape((-1,)) # If flattening last dim
# Or better for (Batch, 1) -> (Batch,):
# Using a custom SqueezeLayer is safest for saving
class SqueezeLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.squeeze(x, axis=-1)
    def get_config(self):
        return super().get_config()
```

---

## 5. Implementation Plan

### Phase 1: Critical Fixes (Week 1)

**Day 1-2: Fix ValidationCallback**
- [ ] Implement positive reward filtering
- [ ] Implement action downsampling (Global)
- [ ] Add max_batches configuration
- [ ] Write unit tests
- [ ] Integration test with small dataset

**Day 3-4: Fix Integer Column Handling**
- [ ] Add feature_types to config
- [ ] Update preprocessing logic
- [ ] Update loader to use feature_types
- [ ] Write unit tests
- [ ] Verify with legacy output

**Day 5: Add Shape Assertions & Serialization Fix**
- [ ] Add assertions to train_step
- [ ] Replace Lambda layers in preprocessing
- [ ] Verify `.keras` saving/loading
- [ ] Update documentation

### Phase 2: Testing (Week 2)

**Day 1-2: Unit Tests**
- [ ] Test NeuralBanditModel
- [ ] Test preprocessing layers
- [ ] Test ValidationCallback
- [ ] Test sample weight computation
- [ ] Achieve > 80% coverage

**Day 3-4: Integration Tests**
- [ ] End-to-end training test
- [ ] Compare outputs with legacy
- [ ] Test model loading/saving
- [ ] Test TFLite export (if enabled)

**Day 5: Performance Tests**
- [ ] Benchmark training time
- [ ] Benchmark memory usage
- [ ] Compare with legacy
- [ ] Document results

### Phase 3: Documentation (Week 3)

**Day 1-2: API Documentation**
- [ ] Document all public functions
- [ ] Add docstrings
- [ ] Generate API docs (Sphinx/MkDocs)

**Day 3: User Guide**
- [ ] Training guide
- [ ] Evaluation guide
- [ ] Inference guide
- [ ] Configuration guide

**Day 4: Technical Documentation**
- [ ] Architecture overview
- [ ] Preprocessing model loading
- [ ] Troubleshooting guide

**Day 5: Code Review & Polish**
- [ ] Code review
- [ ] Fix linting issues
- [ ] Update README

### Phase 4: Optimization (Week 4)

**Day 1-2: Performance Optimization**
- [ ] Optimize ValidationCallback
- [ ] Cache action space
- [ ] Profile bottlenecks
- [ ] Optimize hot paths

**Day 3: Memory Monitoring**
- [ ] Add memory logging
- [ ] Add warnings for large batches
- [ ] Document memory requirements

**Day 4-5: Final Validation**
- [ ] Run full test suite
- [ ] Validate against legacy
- [ ] Performance benchmarks
- [ ] Final documentation review

---

## 6. Testing Strategy

### 6.1 Unit Tests

**Test Files:**
- `tests/test_models.py` - Model tests
- `tests/test_layers.py` - Layer tests
- `tests/test_utilities.py` - Utility tests
- `tests/test_common.py` - Common module tests

**Key Test Cases:**
1. NeuralBanditModel.train_step correctness
2. NeuralBanditModel.test_step correctness
3. ValidationCallback positive reward filtering
4. ValidationCallback action downsampling (verify balance)
5. Preprocessing layer adaptation
6. Sample weight computation
7. Action space calculation
8. Shape assertion triggers
9. Preprocessing serialization (save/load)

### 6.2 Integration Tests

**Test File:** `tests/test_integration.py`

**Key Test Cases:**
1. End-to-end training pipeline
2. Model saving and loading (verify weights restored)
3. Preprocessing model loading
4. Evaluation pipeline
5. Inference pipeline
6. TFLite export (if enabled)

### 6.3 Regression Tests

**Test File:** `tests/test_regression.py`

**Key Test Cases:**
1. Compare Q-values with legacy (small dataset)
2. Compare sample weights with legacy
3. Compare action predictions with legacy
4. Verify feature representations match

### 6.4 Performance Tests

**Test File:** `tests/test_performance.py`

**Key Benchmarks:**
1. Training time per epoch
2. Evaluation time
3. Memory usage
4. Throughput (samples/second)

---

## 7. Configuration Changes

### 7.1 New Configuration Sections

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
    max_batches: 100  # NEW: Limit evaluation batches
    filter_positive_rewards: true  # NEW: Filter positive rewards
    downsample_actions: true  # NEW: Balance action distribution
    downsample_seed: 42  # NEW: Random seed for downsampling
```

### 7.2 Backward Compatibility

- Default `feature_types` inferred from dtype if not specified
- Default `max_batches` = None (evaluate all) for backward compatibility
- Default `filter_positive_rewards` = True (matches legacy)
- Default `downsample_actions` = True (matches legacy)

---

## 8. Risk Mitigation

### 8.1 Technical Risks

**Risk**: Breaking changes in feature representation
- **Mitigation**: Comprehensive regression tests
- **Mitigation**: Feature flag for legacy mode

**Risk**: Performance degradation
- **Mitigation**: Performance benchmarks
- **Mitigation**: Profiling and optimization

**Risk**: Memory issues with large datasets
- **Mitigation**: Memory monitoring
- **Mitigation**: Configurable batch sizes

**Risk**: Keras 3 Serialization Issues
- **Mitigation**: Replace Lambda layers
- **Mitigation**: Verify export formats explicitly

### 8.2 Process Risks

**Risk**: Timeline slippage
- **Mitigation**: Prioritize critical fixes first
- **Mitigation**: Weekly progress reviews

**Risk**: Incomplete testing
- **Mitigation**: Test coverage requirements
- **Mitigation**: Code review process

---

## 9. Success Metrics

### 9.1 Functional Metrics
- ✅ All P0 bugs fixed
- ✅ Test coverage > 80%
- ✅ Model outputs match legacy (within tolerance)
- ✅ All integration tests pass
- ✅ Keras 3 model saving works

### 9.2 Quality Metrics
- ✅ Code passes linting
- ✅ All public APIs documented
- ✅ User guide complete
- ✅ Architecture docs complete

### 9.3 Performance Metrics
- ✅ Training time within 20% of legacy
- ✅ Evaluation time < 5 min per epoch
- ✅ Memory usage bounded
- ✅ No memory leaks

---

## 10. Deliverables

### 10.1 Code
- [ ] Fixed ValidationCallback
- [ ] Fixed preprocessing (Integer columns + Serialization)
- [ ] Shape assertions
- [ ] Performance optimizations
- [ ] Unit tests (> 80% coverage)
- [ ] Integration tests
- [ ] Regression tests

### 10.2 Documentation
- [ ] API documentation
- [ ] User guide
- [ ] Preprocessing loading guide
- [ ] Architecture overview
- [ ] Updated README

### 10.3 Configuration
- [ ] Updated default_config.yaml
- [ ] Configuration migration guide
- [ ] Example configurations

### 10.4 Testing
- [ ] Test suite
- [ ] Test fixtures
- [ ] Performance benchmarks
- [ ] Test coverage report

---

## 11. Timeline

| Phase | Duration | Start | End |
|-------|----------|-------|-----|
| Phase 1: Critical Fixes | 1 week | Week 1 | Week 1 |
| Phase 2: Testing | 1 week | Week 2 | Week 2 |
| Phase 3: Documentation | 1 week | Week 3 | Week 3 |
| Phase 4: Optimization | 1 week | Week 4 | Week 4 |
| **Total** | **4 weeks** | | |

---

## 12. Dependencies

### 12.1 External Dependencies
- TensorFlow 2.17.1+
- NumPy
- Pandas
- PyYAML
- pytest
- pytest-cov

### 12.2 Internal Dependencies
- Config loader
- Logging setup
- Dataset loaders
- Format exporters

---

## 13. Open Questions

1. **Q**: Should we maintain backward compatibility with legacy config format?
   - **A**: Yes, for at least one release cycle

2. **Q**: Should TFLite export be enabled by default?
   - **A**: No, keep it opt-in via config

3. **Q**: What tolerance for regression test comparisons?
   - **A**: Within 1% for Q-values, exact match for predictions

4. **Q**: Should we support multiple evaluation metrics?
   - **A**: Yes, but start with balanced accuracy, add more later

---

## 14. Appendix

### 14.1 File Change Summary

**Modified Files:**
- `src/models/callback.py` - Fix ValidationCallback
- `src/models/preprocessing.py` - Fix integer column handling & serialization
- `src/models/neural_bandit.py` - Add shape assertions
- `config/default_config.yaml` - Add new config options
- `src/layers/loaders/categorical_encoder.py` - Remove Lambda layers

**New Files:**
- `tests/test_models.py` - Model unit tests
- `tests/test_layers.py` - Layer unit tests
- `tests/test_integration.py` - Integration tests
- `tests/test_regression.py` - Regression tests
- `docs/api/` - API documentation
- `docs/user_guide.md` - User guide
- **Full Analysis (OLD vs NEW)**: `legacy_vs_new_comparison.md`

**No Deletions** (backward compatible)

### 14.2 Migration Checklist

- [ ] Review and approve PRD
- [ ] Set up development branch
- [ ] Implement Phase 1 fixes
- [ ] Write tests
- [ ] Code review
- [ ] Merge to main
- [ ] Update documentation
- [ ] Release notes
- [ ] User communication

---

*PRD Version: 1.1*  
*Last Updated: 2025-02-15*  
*Status: Draft - Pending Approval*
