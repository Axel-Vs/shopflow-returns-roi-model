# Intelligent Router Documentation

## Overview

The Intelligent Router is a smart prediction routing system for the ShopFlow returns prediction model. It applies category-specific prediction thresholds to dramatically improve recall and business outcomes.

## Key Concept

Traditional ML models use a single prediction threshold (typically 0.5) for all predictions. However, different product categories have vastly different return patterns and model performance:

- **Fashion**: High return rate (31.8%), good model performance (62.4% recall with 0.5 threshold)
- **Electronics**: Low return rate (17.1%), poor model performance (1.9% recall with 0.5 threshold)
- **Home_Decor**: Medium return rate (17.3%), moderate model performance (12.7% recall with 0.5 threshold)

The Intelligent Router solves this by applying **different prediction thresholds per category**, optimized for business outcomes.

## Performance Results

### Compared to Enhanced Model (0.5 threshold)

| Metric | Enhanced Model | Intelligent Router | Improvement |
|--------|----------------|-------------------|-------------|
| **Recall** | 44.6% | **82.6%** | **+38.0%** |
| **Precision** | 32.6% | 27.1% | -5.5% |
| **F1-Score** | 0.376 | 0.408 | +0.032 |
| **Returns Caught** | 225/505 | 417/505 | +192 |
| **Net Profit** (per 2K) | -$5,256.75 | -$2,763.75 | +$2,493 |
| **Annual Benefit** | - | +$124,650 | - |

## Quick Start

```python
from intelligent_router import create_intelligent_router

# Load router
router = create_intelligent_router(
    model_path='../models/random_forest_model.pkl',
    preprocessor_path='../models/preprocessor.pkl'
)

# Make predictions
predictions, probabilities, strategies = router.predict_with_strategy(new_orders)
```

## Key Features

1. **Category-Specific Thresholds**: Different thresholds per category
2. **Intervention Routing**: Routes to none/email/phone/discount strategies
3. **Threshold Optimization**: Auto-optimize for recall, F1, or custom metrics
4. **Per-Category Evaluation**: Detailed metrics by category
5. **Easy Integration**: Drop-in replacement for standard predictions

## Documentation

- **Source Code**: `src/intelligent_router.py`
- **Demo**: `src/demo_intelligent_router.py`
- **Optimization**: `src/optimize_router_thresholds.py`
- **Tests**: `tests/test_intelligent_router.py`
- **Main README**: `README.md`

## Run Demo

```bash
cd src
python demo_intelligent_router.py
```

## Run Tests

```bash
pytest tests/test_intelligent_router.py -v
```

**Results:** 16 tests passed, 94% code coverage, 0 security vulnerabilities

---

For detailed documentation, see the README.md file and inline code documentation.
