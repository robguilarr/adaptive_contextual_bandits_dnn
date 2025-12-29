# Performance & Scaling Issues in Legacy Contextual Bandits DNN Implementation

**Document Purpose:** Identify architectural, performance, and scalability issues in the legacy `training.ipynb` notebook approach.

---

## Table of Contents

1. [Eager Execution & NumPy Integration](#1-eager-execution--numpy-integration)
2. [Data Pipeline Inefficiencies](#2-data-pipeline-inefficiencies)
3. [Preprocessing Architecture](#3-preprocessing-architecture)
4. [TensorFlow Lite Conversion](#4-tensorflow-lite-conversion)
5. [Code Architecture & Maintainability](#5-code-architecture--maintainability)
6. [Scalability Constraints](#6-scalability-constraints)
7. [Summary](#7-summary)

---

## 1. Eager Execution & NumPy Integration

### 1.1 Forced Eager Mode (`run_eagerly=True`)

**Location:** Cell with `model.compile()`

```python
model.compile(optimizer=Adam(learning_rate=lr_schedule), 
              loss="mse", 
              run_eagerly=True)  # ⚠️ Critical performance issue
```

**Problems:**
- **No Graph Compilation:** Disables TensorFlow's graph optimization, XLA compilation, and operation fusion.
- **~10-50x Slower Training:** Each forward/backward pass incurs Python interpreter overhead.
- **No GPU Kernel Fusion:** Operations execute individually rather than as fused kernels.
- **Memory Fragmentation:** Intermediate tensors cannot be optimized away.

**Why It Was Necessary:**
The legacy code uses `.numpy()` inside `train_step()`, which forces eager mode:

```python
def train_step(self, data):
    # ...
    target = y_pred.numpy()  # ⚠️ Breaks graph compilation
    target[np.arange(states.shape[0]), actions] = y  # ⚠️ NumPy array indexing
    target_tensor = tf.convert_to_tensor(target)  # ⚠️ Back to tensor
```

### 1.2 NumPy ↔ TensorFlow Conversion Overhead

**Problem:** Repeated conversion between NumPy arrays and TensorFlow tensors causes significant overhead.

| Operation | Overhead |
|:----------|:---------|
| `tensor.numpy()` | GPU→CPU memory copy, Python GIL acquisition |
| `tf.convert_to_tensor(np_array)` | CPU→GPU memory copy, tensor allocation |
| NumPy indexing on tensors | Implicit `.numpy()` call |

**Occurrences in Legacy Code:**

```python
# In train_step
target = y_pred.numpy()                           # GPU → CPU
target[np.arange(states.shape[0]), actions] = y   # NumPy ops
target_tensor = tf.convert_to_tensor(target)      # CPU → GPU

# In data generator
tf.convert_to_tensor(s.values)   # DataFrame → Tensor
tf.convert_to_tensor(a.values)   # DataFrame → Tensor
tf.convert_to_tensor(r.values)   # DataFrame → Tensor
```

**Impact:** For batch size 2048 with 8 actions:
- ~32KB copied per batch (predictions)
- ~3 full GPU↔CPU round-trips per training step
- Cannot overlap computation with data transfer

---

## 2. Data Pipeline Inefficiencies

### 2.1 Pandas-Based Data Loading

**Location:** `read_files_into_df()` function

```python
def read_files_into_df(file_list):
    li = []
    for filename in file_list:
        df = pd.read_csv(filename, index_col=None, header=0)
        df.sample(frac=1)  # ⚠️ Result not captured!
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)
```

**Problems:**

| Issue | Impact |
|:------|:-------|
| Sequential file reading | No parallelism, I/O bound |
| Full dataset in memory | Cannot scale beyond RAM |
| `df.sample(frac=1)` result discarded | Bug: no actual shuffling occurs |
| `pd.concat()` copies all data | 2x memory usage during concat |

### 2.2 Python Generator Bottleneck

**Location:** `train_generator()` function

```python
def train_generator():
    for filename in training_files * 100:  # ⚠️ Repeat files 100x
        df = pd.read_csv(filename)          # ⚠️ Blocking I/O
        df.sample(frac=1)                   # ⚠️ Result discarded (bug)
        states, actions, rewards = pre_process(df)
        scale_transform(states)             # ⚠️ In-place mutation
        i = 0
        while i * batch_size < states.shape[0]:
            # Yield batches...
```

**Problems:**

1. **No Prefetching:** GPU idles waiting for CPU to prepare next batch.
2. **Sequential I/O:** Cannot read file N+1 while processing file N.
3. **No Caching:** Same preprocessing repeated 100x per file.
4. **GIL Contention:** Python generator holds GIL during yields.

### 2.3 One-Hot Encoding Explosion

**Location:** `one_hot()` function

```python
def one_hot(df, cols):
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        dummies = dummies.T.reindex(category_space_mapping[col]).T.fillna(0)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(col, axis=1)
    return df
```

**Memory Impact:**
For a categorical column with 100 categories and 1M rows:
- Original: `1M × 1 × 4 bytes (int32) = 4 MB`
- One-Hot: `1M × 100 × 4 bytes (float32) = 400 MB`

**Scaling:**
| Categories | Rows | One-Hot Memory |
|:-----------|:-----|:---------------|
| 100 | 1M | 400 MB |
| 100 | 10M | 4 GB |
| 1,000 | 10M | 40 GB |

---

## 3. Preprocessing Architecture

### 3.1 External Preprocessing Pipeline

**Problem:** Preprocessing is done outside the model graph.

```python
# External scaler (scikit-learn)
scaler = StandardScaler()
scaler.fit(sample_states[NUM_COLUMNS])

# Must be called before every inference
def scale_transform(df):
    df[NUM_COLUMNS] = scaler.transform(df[NUM_COLUMNS])
```

**Consequences:**

| Issue | Impact |
|:------|:-------|
| **Deployment Complexity** | Must ship scaler + model + preprocessing code |
| **Training/Serving Skew** | Risk of different preprocessing in production |
| **No GPU Acceleration** | Preprocessing runs on CPU |
| **Serialization Fragile** | sklearn pickle vs TF SavedModel versioning |

### 3.2 Hard-Coded Category Mappings

```python
category_space_mapping = {}
for c in CAT_COLUMNS:
    category_space_mapping[c] = list(
        map(lambda v: c+'_'+str(v), 
            list(sample_data[c].astype('category').cat.categories))
    )
```

**Problems:**
- Category vocabulary determined at training time from sample.
- New categories at inference time will cause crashes.
- No OOV (Out-of-Vocabulary) handling mechanisms.

---

## 4. TensorFlow Lite Conversion

### 4.1 Naive Conversion

**Location:** TFLite conversion cell

```python
converter = tflite.TFLiteConverter.from_keras_model(model.nn)
tflite_model = converter.convert()
```

**Missing Optimizations:**

| Optimization | Benefit | Status in Legacy |
|:-------------|:--------|:-----------------|
| **Post-training Quantization** | 4x model size reduction | ❌ Not applied |
| **Float16 Quantization** | 2x size, faster on mobile GPUs | ❌ Not applied |
| **Operator Fusion** | Fewer ops, less overhead | ⚠️ Limited |
| **Representative Dataset** | Full integer quantization | ❌ Not used |

### 4.2 Preprocessing Not Bundled

The converted TFLite model (`model.nn`) does NOT include:
- Numerical normalization (requires external scaler).
- Categorical encoding (requires external mapping).
- NaN handling (requires external code).

**Result:** Client applications must replicate all preprocessing logic perfectly, increasing integration cost and error risk.

### 4.3 Metadata Usage

```python
# Only basic metadata added
model_meta.name = "IAP optimizer"
model_meta.description = "Determines the expected reward..."
model_meta.version = "v1"
```

**Missing:**
- Input tensor descriptions.
- Output tensor semantics.
- Feature preprocessing specifications.
- Quantization parameters.

---

## 5. Code Architecture & Maintainability

### 5.1 Monolithic Notebook Design

**Problems:**

| Aspect | Issue |
|:-------|:------|
| **Testing** | Cannot unit test notebook cells. |
| **Versioning** | JSON diffs are unreadable. |
| **Reusability** | Code must be copied to be reused. |
| **CI/CD** | No standard test runners or linting. |
| **Collaboration** | High risk of merge conflicts. |

### 5.2 Global State & Magic Numbers

```python
# Global variables scattered throughout
CAT_COLUMNS = ['geo_country', 'device_os', 'last_run_end_reason']
TRAIN_BATCH_SIZE = 2048
EPOCHS = 10
STEPS_PER_EPOCH = 256
initial_learning_rate = 0.0002

# Magic numbers embedded in model
Dense(256, activation='relu'),
Dense(512, activation='relu'),
# ...
```

**Problems:**
- Experiment configuration mixed with code.
- No reproducibility guarantees.
- Difficult hyperparameter tuning.
- No config file for deployment.

### 5.3 No Separation of Concerns

Single notebook contains:
- Data loading
- Data preprocessing  
- Model definition
- Training loop
- Validation callback
- Testing
- TFLite conversion
- Metadata handling

---

## 6. Scalability Constraints

### 6.1 Memory-Bound Training

**Constraint:** Entire dataset must fit in RAM.

```python
sample_data = read_files_into_df(training_files)  # Load ALL training data
```

**Scaling Limits:**

| Dataset Size | RAM Required | Feasibility |
|:-------------|:-------------|:------------|
| 1M rows | ~2 GB | ✅ |
| 10M rows | ~20 GB | ⚠️ High-memory machine |
| 100M rows | ~200 GB | ❌ Impractical |
| 1B rows | ~2 TB | ❌ Impossible |

### 6.2 Single-File I/O

```python
for filename in training_files * 100:
    df = pd.read_csv(filename)  # Blocking read
```

**Problems:**
- Cannot parallelize across files.
- I/O latency directly impacts training time.
- No overlap between I/O and GPU computation.

### 6.3 No Distributed Training Support

**Missing:**
- `tf.distribute.Strategy` integration.
- Gradient accumulation for effective large batches.
- Multi-worker data sharding.

### 6.4 Validation Memory Duplication

```python
class ValidationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        test_data = read_files_into_df(validation_files)  # ⚠️ Full reload each epoch
        test_states, test_actions, test_rewards = pre_process(test_data)  # ⚠️ Full reprocess
```

**Impact:** Validation overhead grows linearly with validation set size and epochs.

---

## 7. Summary

### Issue Severity Matrix

| Issue | Severity | Performance Impact | Scalability Impact |
|:------|:---------|:-------------------|:-------------------|
| `run_eagerly=True` | 🔴 Critical | 10-50x slower | Blocks optimization |
| NumPy in train_step | 🔴 Critical | 3x overhead/batch | Breaks graph mode |
| Pandas data loading | 🟠 High | I/O bound | Memory limited |
| Python generators | 🟠 High | GPU idle time | No parallelism |
| One-hot encoding | 🟠 High | Memory explosion | Category scaling |
| External preprocessing | 🟡 Medium | Deployment risk | Serving skew |
| No quantization | 🟡 Medium | 4x larger models | Mobile deployment |
| Monolithic notebook | 🟡 Medium | Dev velocity | Team scaling |
| Global state | 🟢 Low | Reproducibility | Experiment tracking |
