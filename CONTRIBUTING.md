# Contributing to Qwen-Image-Edit-LoRA-Trainer

Thank you for your interest in contributing to Qwen-Image-Edit-LoRA-Trainer! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Qwen-Image-Edit-LoRA-Trainer.git
   cd Qwen-Image-Edit-LoRA-Trainer
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/nbeerbower/Qwen-Image-Edit-LoRA-Trainer.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"  # Install development dependencies
   ```

3. Install pre-commit hooks (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Making Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines below

3. Add tests for any new functionality

4. Ensure all tests pass:
   ```bash
   pytest
   ```

5. Commit your changes with a descriptive commit message:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

## Testing

We use pytest for testing. All new features should include tests.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_utils.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest -m unit

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Use fixtures for common setup code

Example:
```python
def test_calculate_dimensions_square():
    """Test calculate_dimensions with square aspect ratio."""
    width, height = calculate_dimensions(1024 * 1024, 1.0)
    assert width == 1024
    assert height == 1024
```

## Code Style

We follow PEP 8 with some modifications:

- Line length: 100 characters
- Use type hints for function arguments and return values
- Write docstrings for all public functions and classes
- Use meaningful variable names

### Formatting Tools

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run formatters before committing:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Run linter
flake8 src/ tests/

# Type check
mypy src/
```

### Docstring Format

Use Google-style docstrings:

```python
def function_name(arg1: int, arg2: str) -> bool:
    """Short description of function.

    Longer description if needed. Can span multiple lines
    and include more details about the function.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided
    """
    pass
```

## Submitting Changes

1. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Create a Pull Request (PR) on GitHub

3. In your PR description:
   - Describe what changes you made and why
   - Reference any related issues (e.g., "Fixes #123")
   - Include screenshots for UI changes
   - List any breaking changes

4. Wait for review and address any feedback

### PR Checklist

Before submitting your PR, ensure:

- [ ] Code follows the project's style guidelines
- [ ] Tests have been added for new features
- [ ] All tests pass
- [ ] Documentation has been updated (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files are included (check `.gitignore`)

## Types of Contributions

We welcome many types of contributions:

### Bug Reports

When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, package versions)
- Relevant error messages or logs

### Feature Requests

When proposing new features:
- Describe the use case
- Explain why this feature would be useful
- Suggest potential implementation approaches
- Consider backward compatibility

### Documentation

Documentation improvements are always welcome:
- Fix typos or clarify unclear sections
- Add examples
- Improve API documentation
- Add tutorials or guides

### Code Contributions

- Bug fixes
- New features
- Performance improvements
- Code refactoring
- Test coverage improvements

## Questions?

If you have questions about contributing, please:
- Check existing issues and discussions
- Open a new issue with the "question" label
- Reach out to the maintainers

Thank you for contributing to Qwen-Image-Edit-LoRA-Trainer!
