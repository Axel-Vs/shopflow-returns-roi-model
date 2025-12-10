# ShopFlow Returns ROI Model

E-commerce returns prediction and ROI modeling for ShopFlow. This repository contains machine learning models to predict product returns and calculate the business impact with real financial metrics.

## 🎯 Why This Project?

E-commerce returns cost businesses billions annually. This project:

- ✅ **Predicts which orders are likely to be returned** (44.6% recall, improving to 70%+ target)
- ✅ **Calculates actual financial impact** using real business costs ($18/return, $3/intervention)
- ✅ **Provides actionable insights** by product category (Fashion: 62% success, Electronics: needs work)
- ✅ **Prioritizes business metrics** (Net Profit, Catch Rate, Cost Per Success) over ML vanity metrics
- ✅ **Currently preventing $169K/year in losses** despite being unprofitable
- 🎯 **Clear path to $100K+ annual profit** with threshold optimization and improved interventions

**The Model Today:**
- Catches 225 of 505 returns (44.6%)
- Saves $3,395 vs doing nothing
- Losing $5,696 per 2,000 orders (still better than baseline!)

**The Model Tomorrow (with improvements):**
- Target 70%+ recall
- Generate $2,000+ profit per 2,000 orders
- $100K-$250K annual profit at scale

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Key Results](#key-results)
- [Business Impact & Financial Analysis](#business-impact--financial-analysis)
- [Output Files](#output-files)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Roadmap to Profitability](#roadmap-to-profitability)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
shopflow-returns-roi-model/
├── data/                          # Data files
│   ├── ecommerce_returns_train.csv
│   └── ecommerce_returns_test.csv
├── src/                           # Source code
│   ├── baseline_model.py         # Original baseline model (unchanged)
│   ├── enhanced_model.py         # Enhanced model with improvements
│   ├── compare_models.py         # Model comparison script
│   ├── config.py                 # Configuration parameters
│   └── utils/                    # Utility modules
│       ├── preprocessing.py      # Data preprocessing utilities
│       ├── evaluation.py         # Model evaluation utilities
│       └── visualization.py      # Visualization utilities
├── models/                        # Saved model artifacts
├── outputs/                       # Results and reports
├── notebooks/                     # Jupyter notebooks for analysis
├── tests/                         # Unit tests
├── pyproject.toml                # Project dependencies
└── README.md                     # This file
```

## Features

### Baseline Model
- Simple logistic regression
- Basic preprocessing
- 9 features
- **Issues**: Severe class imbalance (only predicts "No Return")

### Enhanced Model
- ✅ **Feature Engineering**: 18 features (vs 9 in baseline)
  - Price-based features (high/low price indicators)
  - Customer behavior features (return rate, frequent buyer)
  - Product quality indicators
  - Interaction features
- ✅ **Class Imbalance Handling**: `class_weight='balanced'`
- ✅ **Multiple Models**: Random Forest & Logistic Regression
- ✅ **Comprehensive Evaluation**: 
  - Standard metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Business metrics (cost saved, savings rate, intervention rate)
- ✅ **Better Preprocessing**: Robust pipeline with proper encoding

## Installation

### Prerequisites
- Python 3.12+
- pip

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Axel-Vs/shopflow-returns-roi-model.git
cd shopflow-returns-roi-model
```

2. **Create and activate a virtual environment** (recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

3. **Install dependencies**

Choose one of the following options:

#### Option 1: Install core dependencies only
```bash
pip install -e .
```

#### Option 2: Install with development tools (recommended for development)
```bash
pip install -e ".[dev]"
```

#### Option 3: Install with advanced ML libraries
```bash
pip install -e ".[advanced]"
```

#### Option 4: Install everything (recommended for full functionality)
```bash
pip install -e ".[dev,advanced]"
```

## Quick Start

### 🚀 Fastest Way: Run the Comprehensive Evaluation Notebook

The **recommended way** to explore the project is through the comprehensive evaluation notebook:

```bash
# Make sure you're in the project root
jupyter notebook notebooks/model_evaluation_analysis.ipynb
```

**What you'll see:**
- ✅ Complete financial impact analysis with real business costs
- ✅ 3 business-aligned metrics (Net Profit, Catch Rate, Cost Per Success)
- ✅ Confusion matrix with financial breakdown
- ✅ Performance by product category
- ✅ Model weakness identification
- ✅ ROC curves and visualizations
- ✅ Action plan with prioritized improvements

**Note**: The notebook loads pre-trained models from the `models/` directory. If you want to retrain models, follow the steps below.

---

## Usage

### 1. Train the Baseline Model (Original)
```bash
cd src
python baseline_model.py
```

**Expected Output:**
- Accuracy: ~74.75%
- ⚠️ **Problem**: Predicts only "No Return" class due to class imbalance
- Precision/Recall for returns: 0.00
- **Not suitable for business use**

### 2. Train the Enhanced Model (Recommended)
```bash
cd src
python enhanced_model.py
```

**What it does:**
- Trains Random Forest and Logistic Regression models
- Handles class imbalance with `class_weight='balanced'`
- Creates 18 engineered features (vs 9 in baseline)
- Saves models to `models/` directory
- Generates feature importance report
- Calculates business metrics

**Current Results:**
- **Accuracy**: 62.7%
- **Recall**: 44.6% (catches 44.6% of returns)
- **Precision**: 32.6%
- **F1-Score**: 37.6%
- **ROC-AUC**: 0.601

### 3. Compare All Models
```bash
cd src
python compare_models.py
```

**What it does:**
- Loads all trained models (baseline + enhanced)
- Evaluates on test set
- Creates detailed comparison table
- Shows performance by product category
- Saves results to `outputs/`

**Key Findings:**
- Fashion category: 62.4% recall (best performance)
- Electronics category: 1.9% recall (needs improvement)
- Home_Decor category: 12.7% recall (moderate)

### 4. Run Comprehensive Business Analysis
```bash
jupyter notebook notebooks/model_evaluation_analysis.ipynb
```

**What you'll get:**
- Detailed financial impact calculations
- Business metrics dashboard
- Confusion matrix with dollar values
- Category-specific performance analysis
- Visualizations saved to `outputs/`
- Actionable recommendations

## Key Results

### Model Performance Comparison

| Metric | Baseline | Enhanced (Random Forest) | Improvement |
|--------|----------|--------------------------|-------------|
| Accuracy | 74.75% | 62.70% | -12.05% |
| Precision | 0.00% | 32.56% | ∞ |
| Recall | 0.00% | 44.55% | ∞ |
| F1-Score | 0.00 | 0.376 | ∞ |
| ROC-AUC | N/A | 0.601 | N/A |

**Why lower accuracy is actually better**: The baseline has high accuracy because it predicts "No Return" for everything due to class imbalance (75% of orders don't return). However, it's **useless for business** because it never catches any returns (0% recall). The enhanced model sacrifices some accuracy to actually predict returns (44.6% recall), making it **profitable and actionable**.

### Business Metrics (Current Model)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Net Profit** | **-$5,695.50** per 2,000 orders | $2,000+ | ❌ Unprofitable |
| **Catch Rate (Recall)** | **44.6%** | 65-70% | ⚠️ Below target |
| **Cost Per Success** | **$9.21** | $4-6 | ⚠️ Too high |

**Financial Reality:**
- Return cost: **$18** (actual business cost)
- Intervention cost: **$3** (customer service, communications)
- Intervention effectiveness: **35%** (35% of interventions prevent returns)
- Net savings per successful intervention: **$3.30**

**Current Impact (per 2,000 orders):**
- Revenue from prevented returns: **$742.50** (225 successful interventions)
- Costs: **$6,438** ($1,398 from false positives + $5,040 from missed returns)
- **Net Loss: -$5,695.50**

**However**: Still saves **$3,394.50** compared to doing nothing (37% reduction in losses)!

### Performance by Product Category

| Category | Return Rate | Samples | Recall | Status |
|----------|-------------|---------|--------|--------|
| **Fashion** | 31.3% | 1,104 | 62.4% | ✅ Best |
| **Home_Decor** | 19.0% | 289 | 12.7% | ⚠️ Moderate |
| **Electronics** | 17.1% | 607 | 1.9% | ❌ Worst |

## Configuration

Edit `src/config.py` to customize:
- Model hyperparameters (Random Forest: n_estimators, max_depth, etc.)
- Business cost parameters (return cost, intervention cost, effectiveness)
- Feature engineering settings
- Evaluation metrics and thresholds

**Current Business Parameters:**
```python
RETURN_COST = 18              # Cost when a return happens
INTERVENTION_COST = 3         # Cost of customer service intervention
INTERVENTION_EFFECTIVENESS = 0.35  # 35% of interventions prevent returns
```

## Output Files

After running the models and evaluation notebook, you'll find these outputs:

### In `models/` directory:
- `random_forest_model.pkl` - Trained Random Forest model
- `logistic_regression_model.pkl` - Trained Logistic Regression model
- `preprocessor.pkl` - Fitted preprocessing pipeline

### In `outputs/` directory:
- `feature_importance.csv` - Top features from Random Forest
- `model_comparison.csv` - Comparison of enhanced models
- `full_model_comparison.csv` - Including baseline model
- `performance_by_category.csv` - Category-specific metrics
- `financial_impact_analysis.png` - 4-panel financial visualization dashboard
- `confusion_matrix.png` - Confusion matrix with financial heatmap
- `category_performance.png` - Performance charts by product category
- `roc_curve.png` - ROC curve for model evaluation

## Troubleshooting

### Common Issues

**1. Module not found errors**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -e ".[dev]"
```

**2. Jupyter kernel not found**
```bash
# Install ipykernel in your virtual environment
pip install ipykernel
python -m ipykernel install --user --name=shopflow-roi
```

**3. Models not found when running notebook**
```bash
# Train the enhanced model first
cd src
python enhanced_model.py
cd ..
```

**4. Data files not found**
```bash
# Make sure you're running from the project root
pwd  # Should show: .../shopflow-returns-roi-model

# Check data files exist
ls data/
# Should show: ecommerce_returns_train.csv, ecommerce_returns_test.csv
```

## Development

### Run tests
```bash
pytest tests/
```

### Format code
```bash
black src/
```

### Type checking
```bash
mypy src/
```

### Lint code
```bash
flake8 src/
```

## Business Impact & Financial Analysis

### Current Model Performance

**Confusion Matrix Breakdown (per 2,000 orders):**
- **True Negatives (TN)**: 1,029 - Correctly predicted no return ($0 cost)
- **False Positives (FP)**: 466 - Wasted interventions ($1,398 cost)
- **False Negatives (FN)**: 280 - Missed returns ($5,040 cost) 🚨 **Most expensive!**
- **True Positives (TP)**: 225 - Prevented returns ($742.50 revenue)

**Key Insight**: False Negatives cost **6× more** than False Positives ($18 vs $3), so we prioritize **recall over precision**.

### Financial Model

The system calculates:
1. **Net Profit per Order Batch** (PRIMARY metric)
   - Formula: `(TP × $3.30) - (FP × $3) - (FN × $18)`
   - Current: **-$5,695.50** per 2,000 orders
   
2. **Catch Rate / Recall** (OPERATIONAL metric)
   - Formula: `TP / (TP + FN)`
   - Current: **44.6%** (need 65-70% for profitability)
   
3. **Cost Per Successful Intervention** (EFFICIENCY metric)
   - Formula: `(Total Interventions × $3) / TP`
   - Current: **$9.21** (need <$6 for breakeven)

### Annualized Projections (100K orders/year)

- **Current**: Losing **$284,775/year**
- **vs No Model**: Would lose $454,500/year
- **Net Benefit**: Preventing **$169,725/year** in additional losses
- **Potential if profitable**: Could generate $100K-$250K profit/year

## Roadmap to Profitability

Based on the comprehensive evaluation, here's the prioritized action plan:

### 🚨 Immediate Actions (Week 1)
1. **Lower Prediction Threshold** (0.5 → 0.35-0.40)
   - Increase recall from 44.6% → 65%+
   - Accept more false positives to prevent costly false negatives
   - Expected impact: Catch 103 more returns

### 📈 Short-term (Month 1-2)
2. **Improve Intervention Effectiveness** (35% → 50%+)
   - Better customer communications
   - Enhanced sizing tools/guides
   - Virtual try-on features
   - Instant exchange offers
   - Expected impact: Doubles profitability per successful intervention

3. **Category-Specific Models**
   - Focus on Electronics (currently only 1.9% recall)
   - Develop Fashion-specific features (already performing well at 62.4% recall)
   - Expected impact: 15-20% increase in overall recall

### 🔧 Medium-term (Month 2-3)
4. **Reduce Intervention Costs** ($3 → $1.50-$2)
   - Automate email/SMS interventions
   - Self-service tools
   - Chatbot integration
   - Expected impact: Halves cost per intervention

5. **Advanced Resampling**: SMOTE, ADASYN for better class balance
6. **Add Behavioral Features**: 
   - Time spent on product page
   - Number of reviews read
   - Product comparisons made
   - Cart abandonment history

### 🎯 Long-term (Quarter 2)
7. **Hyperparameter Tuning**: GridSearchCV with business metrics as scoring
8. **Try Advanced Models**: XGBoost, LightGBM with custom loss functions
9. **Ensemble Methods**: Combine multiple models
10. **A/B Testing**: Deploy in production with control group
11. **Real-time Predictions**: Create API endpoint for live scoring

### 💰 Expected Impact if All Improvements Implemented
- Net Profit: **-$5,695** → **+$2,000+** per 2,000 orders
- Annual Impact: **-$285K** → **+$100K** (swing of **$385K!**)
- Recall: **44.6%** → **70%+**
- Cost Per Success: **$9.21** → **<$6**

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Author

Axel Vargas

## Acknowledgments

- Dataset: E-commerce returns synthetic data
- Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn
