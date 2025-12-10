# shopflow-returns-roi-model
Predict returns to maximize ROI

## Overview
This repository contains machine learning models to predict e-commerce returns and maximize ROI.

## Baseline Model
The baseline model (`baseline_model.py`) implements a simple Logistic Regression classifier for predicting returns.

### Features
- Data preprocessing (handling missing values, encoding categorical variables)
- Feature scaling using StandardScaler
- Logistic Regression model training
- Model evaluation (accuracy, classification report)
- Model persistence using joblib

### Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Usage
1. Prepare your data files:
   - `ecommerce_returns_train.csv` - Training dataset
   - `ecommerce_returns_test.csv` - Test dataset

2. Run the baseline model:
```bash
python baseline_model.py
```

3. The script will:
   - Load and preprocess the data
   - Train a Logistic Regression model
   - Evaluate performance on train and test sets
   - Save the trained model as `logistic_regression_model.pkl`
   - Save the scaler as `scaler.pkl`
   - Save the label encoders as `label_encoders.pkl`

### Expected Output
- Training and test accuracy scores
- Classification report with precision, recall, and F1-scores
- Top 10 most important features based on model coefficients

### Data Format
The CSV files should contain:
- Feature columns (numeric and categorical)
- Target column (one of: 'returned', 'return', 'is_returned', 'target', 'label')

If the target column name differs, the script will use the last column as the target.
