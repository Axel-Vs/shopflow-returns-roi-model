# Tests Directory

This directory contains unit tests for the project.

## Test Structure

```
tests/
├── test_preprocessing.py      # Tests for data preprocessing
├── test_evaluation.py         # Tests for evaluation metrics
├── test_models.py            # Tests for model training
└── test_utils.py             # Tests for utility functions
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py
```

## Writing Tests

Use pytest framework. Example:

```python
def test_preprocessor():
    from src.utils.preprocessing import DataPreprocessor
    preprocessor = DataPreprocessor()
    # Add test assertions
```
