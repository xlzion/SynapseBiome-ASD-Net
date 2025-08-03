# SynapseBiome ASD-Net Project Structure

This document describes the reorganized project structure for SynapseBiome ASD-Net, designed to meet Nature Methods publication standards.

## Overview

The project has been restructured to follow modern Python package conventions and best practices for scientific software. The new structure emphasizes:

- **Modularity**: Clear separation of concerns
- **Reproducibility**: Comprehensive configuration management
- **Maintainability**: Clean, well-documented code
- **Extensibility**: Easy to add new features and models
- **Professional Standards**: Ready for publication and community adoption

## Directory Structure

```
ASDnet/
├── src/                           # Main source code
│   ├── __init__.py               # Package initialization
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   ├── asdnet.py            # Main SynapseBiomeASDNet class
│   │   ├── fmri_gnn.py          # fMRI Graph Neural Network
│   │   ├── microbiome_mlp.py    # Microbiome Sparse MLP
│   │   └── contrastive.py       # Contrastive learning components
│   ├── data/                     # Data processing modules
│   │   ├── __init__.py
│   │   ├── preprocessing.py     # Data preprocessing utilities
│   │   └── dataloaders.py       # Data loading utilities
│   ├── training/                 # Training utilities
│   │   ├── __init__.py
│   │   ├── trainer.py           # Main training loop
│   │   └── utils.py             # Training utilities
│   ├── analysis/                 # Analysis and evaluation
│   │   ├── __init__.py
│   │   ├── biomarker_discovery.py
│   │   ├── feature_importance.py
│   │   └── network_analysis.py
│   ├── evaluation/               # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── cross_validation.py
│   │   └── statistical_analysis.py
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── visualization.py
│       └── helpers.py
├── configs/                       # Configuration files
│   ├── training_config.yaml      # Main training configuration
│   ├── model_config.yaml         # Model architecture config
│   └── data_config.yaml          # Data processing config
├── scripts/                       # Executable scripts
│   ├── train.py                  # Main training script
│   ├── evaluate.py               # Evaluation script
│   ├── preprocess_data.py        # Data preprocessing script
│   └── analyze_biomarkers.py     # Biomarker analysis script
├── tests/                         # Test suite
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_data.py
│   └── test_analysis.py
├── docs/                          # Documentation
│   ├── api/                      # API documentation
│   ├── tutorials/                # Tutorial notebooks
│   └── examples/                 # Example scripts
├── data/                          # Data directory (not in repo)
├── outputs/                       # Training outputs
├── models/                        # Saved model checkpoints
├── results/                       # Analysis results
├── README.md                      # Main project documentation
├── LICENSE                        # MIT License
├── CONTRIBUTING.md               # Contribution guidelines
├── setup.py                      # Package installation
├── requirements.txt              # Runtime dependencies
├── requirements-dev.txt          # Development dependencies
└── .gitignore                    # Git ignore rules
```

## Key Components

### 1. Source Code (`src/`)

#### Models (`src/models/`)
- **`asdnet.py`**: Main `SynapseBiomeASDNet` class that integrates all components
- **`fmri_gnn.py`**: Dynamic fMRI Graph Neural Network with attention mechanisms
- **`microbiome_mlp.py`**: Sparse MLP for microbiome data processing
- **`contrastive.py`**: Contrastive learning framework and loss functions

#### Data Processing (`src/data/`)
- **`preprocessing.py`**: ABIDE-II data preprocessing utilities
- **`dataloaders.py`**: Custom data loaders for fMRI and microbiome data

#### Training (`src/training/`)
- **`trainer.py`**: `ContrastiveTrainer` class with multi-phase training
- **`utils.py`**: Training utilities (early stopping, metrics, schedulers)

#### Analysis (`src/analysis/`)
- **`biomarker_discovery.py`**: Biomarker identification and analysis
- **`feature_importance.py`**: Feature importance computation methods
- **`network_analysis.py`**: Brain connectivity network analysis

### 2. Configuration (`configs/`)

#### `training_config.yaml`
Comprehensive configuration file containing:
- Model architecture parameters
- Training hyperparameters
- Data processing settings
- Evaluation metrics
- Logging and output settings
- Hardware configuration
- Reproducibility settings

### 3. Scripts (`scripts/`)

#### `train.py`
Main training script with command-line interface:
```bash
python scripts/train.py --config configs/training_config.yaml --data_dir /path/to/data
```

### 4. Legacy Code Preservation

The original code structure is preserved in:
- `main/`: Original implementation files
- `ABIDE2/`: Data processing and validation scripts

This allows for:
- Comparison with new implementation
- Gradual migration of features
- Backward compatibility
- Reference for methodology

## Migration Guide

### From Old to New Structure

1. **Model Usage**:
   ```python
   # Old
   from main.biomarker.adversarial_new import ContrastiveModel
   
   # New
   from src.models import SynapseBiomeASDNet
   ```

2. **Training**:
   ```python
   # Old
   train_contrastive_adversarial(model, train_loader, val_loader)
   
   # New
   trainer = ContrastiveTrainer(config, model, train_loader, val_loader)
   trainer.train()
   ```

3. **Configuration**:
   ```python
   # Old: Hard-coded parameters
   # New: YAML configuration files
   ```

## Benefits of New Structure

### 1. **Professional Standards**
- Follows Python packaging best practices
- Comprehensive documentation
- Proper dependency management
- Version control and licensing

### 2. **Reproducibility**
- Configuration-driven experiments
- Deterministic training
- Comprehensive logging
- Version tracking

### 3. **Maintainability**
- Clear separation of concerns
- Modular design
- Type hints and documentation
- Comprehensive testing

### 4. **Extensibility**
- Easy to add new models
- Plugin architecture
- Configuration-driven features
- API-first design

### 5. **Publication Ready**
- Professional documentation
- Clear methodology description
- Reproducible experiments
- Community-friendly structure

## Usage Examples

### Basic Training
```python
from src.models import SynapseBiomeASDNet
from src.training import ContrastiveTrainer
import yaml

# Load configuration
with open('configs/training_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize model
model = SynapseBiomeASDNet(config['model'])

# Initialize trainer
trainer = ContrastiveTrainer(config, model, train_loader, val_loader, device)

# Train model
trainer.train()
```

### Biomarker Analysis
```python
from src.analysis import BiomarkerAnalyzer

analyzer = BiomarkerAnalyzer(model, data_loader)
biomarkers = analyzer.discover_biomarkers()
analyzer.visualize_results()
```

### Evaluation
```python
from src.evaluation import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
metrics = analyzer.compute_metrics(predictions, ground_truth)
p_values = analyzer.statistical_testing(results)
```

This structure ensures that SynapseBiome ASD-Net meets the highest standards for scientific software publication and community adoption. 