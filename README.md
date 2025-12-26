# Adaptive Contextual Bandits DNN

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17.1-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0-red.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for training neural contextual bandits, evaluating their performance, and deploying models for personalized recommendation systems. This implementation uses deep neural networks to learn Q-values for contextual multi-armed bandit problems, with a focus on in-app purchase (IAP) optimization.

## Overview

<div align="center" style="display: flex; align-items: center; justify-content: center; gap: 20px;">
  <img src="docs/static/images/iap_powerups.png" alt="IAP Powerups" width="350"/>
  <img src="docs/static/images/tensorflow_logo.png" alt="TensorFlow Logo" width="350"/>
</div>

This package implements a **Neural Contextual Bandit** model that learns to recommend actions (e.g., power-ups in a mobile game) based on user context. The model uses a deep Q-network (DQN) architecture to estimate Q-values for each action given contextual features, enabling personalized recommendations that maximize expected reward.

### Core Features

- **Deep Q-Network Architecture**: Multi-layer perceptron (MLP) that learns Q-values for contextual bandit problems
- **Preprocessing Submodel**: builds a functional preprocessing graph with conditional layer chaining based on data types, and includes persisted (**cached in-memory**) action mapping for action-to-integer encoding
- **Flexible Preprocessing**: Automatic feature encoding with normalization for numerical features and one-hot encoding for categorical features
- **ValidationCallback**: Evaluation callback that measures balanced accuracy by filtering and downsampling on positive rewards, comparing Q-values against predicted rewards during training
- **Class Imbalance Reward-Handling**: Dynamic sample weighting to address imbalanced reward distributions
- **Multiple Loss Functions**: Support for MSE and Binary Focall Loss cross-entropy for handling class imbalance
- **TensorBoard Integration**: Comprehensive logging and visualization of training metrics
- **Synthetic Data Generation**: Built-in data generator for creating training datasets with configurable exploration *Epsilon-Greedy* policies
- **Model Export**: Export models in multiple formats (Keras, H5, TensorFlow SavedModel, TFLite)

## Acknowledgments

This project is inspired by Google's Firebase team work (*E. Sun, I. Ulukaya, et al.*) on Realtime on-device In-app-purchase optimization. This implementation is an **enhanced and higher-performance version** that includes:

- Class imbalance handler with dynamic weighting and Focal Loss support.
- **TensorFlow Keras Functional API preprocessing**: The original implementation used NumPy for encoding preprocessing layers, while this version uses TensorFlow Keras 3 functional programming for end-to-end preprocessing pipelines.
- Enhanced preprocessing pipeline with flexible feature encoding and automatic layer adaptation.
- TensorBoard integration for training monitoring.
- Synthetic data generation module for testing and development.
- Multiple model export formats including TFLite with metadata (Not 100% ready on version 0.1.).

**Original Inspiration:**

- [Firebase IAP Optimization Codelab](https://firebase.google.com/codelabs/iap-optimization?hl=en#6)
- [Original Repository](https://github.com/googlecodelabs/firebase-iap-optimization/tree/main)

## Installation

### Prerequisites

- Python >= 3.10
- Make (for installation automation)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/robguilarr/adaptive_contextual_bandits_dnn.git
cd adaptive_contextual_bandits_dnn

# Create virtual environment and install dependencies
make install

# Or install with optional dependencies manually
make venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"  # For development dependencies
pip install -e ".[docs]"  # For documentation dependencies
```

The `make install` command will:
- Create a Python 3.10 virtual environment in `venv/`
- Upgrade pip, setuptools, and wheel
- Install the package in development mode with all dependencies

To clean up the virtual environment:
```bash
make clean
```

### Dependencies

See `pyproject.toml` for the complete list of dependencies.

## Quick Start

### 1. Generate Synthetic Data (Optional)

If you don't have training data, you can generate synthetic data:

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # On Windows: venv\Scripts\activate

python -m src.utilities.datasets.data_generator \
    --train_size 700000 \
    --val_size 200000 \
    --test_size 100000 \
    --epsilon 1.0  # 1.0 = pure random (cold start, 100% exploration), 0.3 = production logs (exploit)
```

This will create CSV files in `data/raw/`:
- `training.csv` - Training dataset
- `validation.csv` - Validation dataset
- `test.csv` - Test dataset

### 2. Configure the Model

Edit `config/default_config.yaml` to configure:
- Data paths and preprocessing settings
- Model architecture and training hyperparameters
- Feature types (numerical vs categorical)
- Export settings

Key configuration sections:
- `data`: Dataset paths and batch sizes
- `features`: Feature column names and types
- `training`: Epochs, learning rate, loss function
- `model_export`: Save formats and directories

### 3. Train the Model

```bash
# Activate virtual environment (if using make install)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Set PYTHONPATH if needed
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run training
python src/train.py
# Or use venv's python directly: venv/bin/python src/train.py
```

### 4. Monitor Training with TensorBoard

```bash
tensorboard --logdir data/artifacts/models
```

Open `http://localhost:6006` in your browser to view

### 5. Evaluate the Model

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # On Windows: venv\Scripts\activate

python src/evaluate.py \
    --model_path data/artifacts/models/YYYYMMDD/keras/NeuralBanditModel.keras \
    --model_type keras
```

### 6. Run Inference

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # On Windows: venv\Scripts\activate

python src/inference.py \
    --model_path data/artifacts/models/YYYYMMDD/keras/NeuralBanditModel.keras \
    --input_data data/raw/test.csv \
    --output_dir data/processed/inference \
    --batch_size 32
```

## Project Structure

```
adaptive_contextual_bandits_dnn/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   ├── inference.py                # Inference script
│   ├── common/                     # Common utilities
│   │   ├── config.py              # Configuration loader
│   │   ├── constants.py           # TensorFlow constants
│   │   └── logging.py             # Logging setup
│   ├── models/                     # Model definitions
│   │   ├── neural_bandit.py       # Neural Bandit model
│   │   ├── preprocessing.py       # Preprocessing model
│   │   └── callback.py            # Training callbacks
│   ├── layers/                     # Custom TensorFlow layers
│   │   ├── categorical_encoder.py # Categorical encoding
│   │   ├── numerical_encoder.py   # Numerical normalization
│   │   └── dynamic_category_encoding.py  # Dynamic encoding
│   └── utilities/                  # Utility modules
│       ├── datasets/               # Dataset utilities
│       │   ├── loader.py          # Dataset loading
│       │   ├── weights.py         # Sample weighting
│       │   └── data_generator.py  # Synthetic data generation
│       └── formats/                # Format utilities
│           ├── export.py          # Model export
│           └── load.py            # Model loading
├── config/
│   └── default_config.yaml        # Default configuration
├── data/                          # Data directory
│   ├── raw/                       # Raw datasets
│   ├── processed/                 # Processed data
│   └── artifacts/                 # Model artifacts
│       └── models/                # Saved models
├── docs/                          # Documentation/Debuggin & Materials
├── tests/                         # Unit tests
├── notebooks/                     # Jupyter notebooks
├── pyproject.toml                 # Package configuration
└── README.md                      # This file
```

## Configuration

The package uses YAML configuration files. The main configuration file is `config/default_config.yaml`. Key sections:

### Data Configuration

```yaml
data:
  train_data_file_path: data/raw/training.csv
  eval_data_file_path: data/raw/validation.csv
  batch_size: 128
  shuffle_buffer_size: 5000
```

### Feature Configuration

```yaml
features:
  label_column: is_powerup_clicked
  action_weight_column: presented_powerup
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

### Training Configuration

```yaml
training:
  epochs: 20
  initial_learning_rate: 0.001
  lr_decay_rate: 0.7
  lr_decay_steps: 1000
  loss_function: "focal"  # "mse" or "focal"
  focal_loss_gamma: 2.0
  focal_loss_alpha: 0.25
```

### Model Export Configuration

```yaml
model_export:
  save_format: tf  # "keras", "h5", or "tf"
  tflite_enabled: false
  model_directory: data/artifacts/models
  checkpoints:
    enabled: true
    monitor: val_loss
    save_best_only: true
```

## Model Architecture

The Neural Bandit model consists of two main components:

1. **Preprocessing Submodel**: Encodes raw features into a unified representation
   - Numerical features: Normalization layers
   - Categorical features: One-hot encoding or lookup tables
   - Action encoding: Dynamic category encoding with vocabulary mapping

2. **Q-Network (MLP)**: Deep neural network that estimates Q-values
   - Architecture: 7 dense layers with ReLU activation
   - Dropout: 0.2 dropout for regularization
   - Output: Q-values for each action (dimension = action space size)

### Training Process

1. **Feature Preprocessing**: Raw features are normalized/encoded
2. **Q-Value Prediction**: Network predicts Q-values for all actions
3. **Action Selection**: During training, uses the action from the dataset
4. **Loss Computation**: Compares predicted Q-value for chosen action with actual reward
5. **Weighted Updates**: Applies sample weights to handle class imbalance

## Model Export Formats

The package supports exporting models in multiple formats:

- **Keras** (`.keras`): Native Keras format, recommended for Python deployment
- **H5** (`.h5`): Legacy HDF5 format
- **TensorFlow SavedModel** (`tf/`): Standard TensorFlow format
- **TensorFlow Lite** (`.tflite`): For mobile/edge deployment with metadata

TFLite export includes preprocessing metadata (normalization stats, vocabulary, action mappings) in JSON format.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `PYTHONPATH` includes the project root

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

2. **Configuration Not Found**: Set `WORKDIR` environment variable or ensure `config/default_config.yaml` exists

3. **Memory Issues**: Reduce `batch_size` in configuration or use data generators

4. **Model Not Learning**: Check class imbalance - consider using Focal Loss or adjusting sample weights

## References

### Papers of interest

- [Neural Contextual Bandits for Personalized Recommendation](docs/materials/Neural%20Contextual%20Bandits%20for%20Personalized%20Recommendation.pdf)
- [Scalable Neural Contextual Bandit for Recommender Systems](docs/materials/Scalable%20Neural%20Contextual%20Bandit%20for%20Recommender%20Systems.pdf)
- [Customer Lifetime Value in Video Games Using Deep Learning and Parametric Models](https://arxiv.org/abs/1811.12799)

## License

See `LICENSE` file for details.
