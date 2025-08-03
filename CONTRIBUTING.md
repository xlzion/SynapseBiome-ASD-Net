# Contributing to SynapseBiome ASD-Net

Thank you for your interest in contributing to SynapseBiome ASD-Net! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please read it before contributing.

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker to report bugs
- Include a clear and descriptive title
- Provide detailed steps to reproduce the bug
- Include system information (OS, Python version, etc.)
- Attach relevant error messages and logs

### Suggesting Enhancements

- Use the GitHub issue tracker for feature requests
- Clearly describe the enhancement and its benefits
- Provide use cases and examples if applicable
- Consider the impact on existing functionality

### Code Contributions

#### Setting Up Development Environment

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment:
   ```bash
   conda create -n asdnet-dev python=3.8
   conda activate asdnet-dev
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
5. Install the package in development mode:
   ```bash
   pip install -e .
   ```

#### Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass:
   ```bash
   pytest tests/
   ```
5. Run linting and formatting:
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```
6. Commit your changes with a descriptive message
7. Push to your fork and create a pull request

#### Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and concise
- Use meaningful variable and function names

#### Testing

- Write unit tests for new functionality
- Ensure test coverage for critical components
- Use pytest for testing
- Include integration tests for complex workflows

#### Documentation

- Update documentation for new features
- Include code examples in docstrings
- Update README.md if necessary
- Add comments for complex algorithms

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Update documentation as needed
3. Add tests for new functionality
4. Ensure all tests pass
5. Update the CHANGELOG.md with a description of your changes
6. Request review from maintainers

## Review Process

- All pull requests require review from at least one maintainer
- Address feedback and requested changes
- Maintainers may request additional tests or documentation
- Once approved, your changes will be merged

## Release Process

- Releases are made by maintainers
- Version numbers follow semantic versioning
- Release notes are generated from the CHANGELOG.md
- Documentation is updated for new releases

## Getting Help

- Check existing issues and pull requests
- Join our community discussions
- Contact maintainers for questions

Thank you for contributing to SynapseBiome ASD-Net! 