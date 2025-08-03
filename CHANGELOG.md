# Changelog

All notable changes to SynapseBiome ASD-Net will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-XX

### Added
- **Complete project restructuring** for Nature Methods publication standards
- **Modular architecture** with clear separation of concerns
- **Professional documentation** including comprehensive README and API docs
- **Configuration-driven training** using YAML configuration files
- **Multi-phase training pipeline** (pretrain, contrastive, finetune)
- **Comprehensive logging** with TensorBoard integration
- **Type hints** throughout the codebase for better maintainability
- **Unit tests** framework and initial test suite
- **Development tools** including linting, formatting, and pre-commit hooks
- **Package installation** support via setup.py
- **MIT License** for open-source distribution
- **ABIDE I dataset support** with comprehensive data processing pipeline
- **Multimodal data processor** for combining ABIDE I and ABIDE II datasets
- **Advanced data preprocessing** with quality control and validation
- **QIIME2 integration** for microbiome data processing from BIOM format
- **Comprehensive microbiome processing** with normalization, transformation, and feature selection

### Changed
- **Refactored model architecture** into modular components:
  - `SynapseBiomeASDNet`: Main model class
  - `fMRI3DGNN`: Dynamic fMRI Graph Neural Network
  - `SparseMLP`: Microbiome processing with L1 regularization
  - `ContrastiveModel`: Contrastive learning framework
- **Improved training pipeline** with:
  - Early stopping and learning rate scheduling
  - Mixed precision training support
  - Gradient clipping and regularization
  - Comprehensive metric tracking
- **Enhanced data processing** with:
  - Standardized data loaders for both ABIDE I and ABIDE II
  - Advanced preprocessing utilities with quality control
  - Cross-validation and leave-one-site-out validation
  - Support for multiple brain atlases (CC200, AAL, EZ, etc.)
- **Better error handling** and validation throughout

### Technical Improvements
- **Code quality**: Added type hints, docstrings, and comprehensive comments
- **Performance**: Optimized data loading and training loops
- **Reproducibility**: Deterministic training with seed management
- **Scalability**: Support for multi-GPU training and distributed computing
- **Maintainability**: Clean, modular code structure following Python best practices

### Documentation
- **Comprehensive README** with installation, usage, and citation instructions
- **API documentation** for all public classes and functions
- **Tutorial notebooks** for common use cases
- **Configuration examples** and best practices
- **Contributing guidelines** for community development

### Infrastructure
- **CI/CD pipeline** setup for automated testing
- **Dependency management** with requirements files
- **Development environment** setup instructions
- **Code formatting** with Black and isort
- **Linting** with flake8 and mypy
- **Pre-commit hooks** for code quality

## [0.9.0] - 2024-01-XX (Pre-restructuring)

### Added
- Initial implementation of SynapseBiome ASD-Net
- Dynamic fMRI Graph Neural Network with attention mechanisms
- Sparse MLP for microbiome data processing
- Contrastive learning framework for multimodal fusion
- Biomarker discovery and analysis tools
- ABIDE-II dataset integration
- Basic training and evaluation scripts

### Features
- **Dynamic Graph Construction**: Subject-specific brain connectivity graphs
- **Attention Mechanisms**: Graph attention networks for brain region analysis
- **Sparse Feature Learning**: L1 regularization for microbiome feature selection
- **Contrastive Learning**: Modality-invariant representation learning
- **Biomarker Discovery**: Comprehensive analysis tools for feature importance
- **Cross-validation**: Leave-one-site-out validation for clinical relevance

### Research Contributions
- Novel multimodal fusion approach for ASD diagnosis
- Dynamic graph neural networks for fMRI analysis
- Contrastive learning for modality alignment
- Comprehensive biomarker discovery framework
- Clinical validation on ABIDE-II dataset

## Migration Notes

### From Version 0.9.0 to 1.0.0

The project has undergone a complete restructuring to meet publication standards. Key changes:

1. **Import Changes**:
   ```python
   # Old
   from main.biomarker.adversarial_new import ContrastiveModel
   
   # New
   from src.models import SynapseBiomeASDNet
   ```

2. **Training Interface**:
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

4. **Installation**:
   ```bash
   # Old: Manual dependency installation
   # New: pip install -e .
   ```

### Backward Compatibility

The original implementation is preserved in the `main/` directory for reference and comparison. Users can:

- Compare old and new implementations
- Gradually migrate to the new structure
- Maintain backward compatibility during transition
- Reference original methodology

## Future Roadmap

### Version 1.1.0 (Planned)
- Additional model architectures
- Extended biomarker analysis tools
- Performance optimizations
- Additional datasets support

### Version 1.2.0 (Planned)
- Web interface for model deployment
- Real-time prediction capabilities
- Clinical integration tools
- Extended documentation and tutorials

### Long-term Goals
- Community-driven development
- Integration with clinical workflows
- Support for additional neuroimaging modalities
- International collaboration and validation studies

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

For questions and support:
- GitHub Issues: [Create an issue](https://github.com/your-username/synapsebiome-asdnet/issues)
- Documentation: [Read the docs](https://synapsebiome-asdnet.readthedocs.io/)
- Email: your.email@institution.edu 