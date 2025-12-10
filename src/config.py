"""
Configuration file for model parameters and settings
"""

# Data paths
DATA_DIR = '../data'
MODELS_DIR = '../models'
OUTPUTS_DIR = '../outputs'

TRAIN_DATA = f'{DATA_DIR}/ecommerce_returns_train.csv'
TEST_DATA = f'{DATA_DIR}/ecommerce_returns_test.csv'

# Model parameters
RANDOM_SEED = 42

# Random Forest parameters
RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# Logistic Regression parameters
LOGISTIC_REGRESSION_PARAMS = {
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED
}

# XGBoost parameters (for future use)
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 3,  # Adjust based on class imbalance
    'random_state': RANDOM_SEED
}

# Feature engineering settings
FEATURE_ENGINEERING = {
    'create_interactions': True,
    'create_ratios': True,
    'create_binned_features': True
}

# Business metrics parameters
BUSINESS_PARAMS = {
    'cost_of_return': 25.0,
    'cost_of_false_positive': 5.0,
    'revenue_per_sale': 100.0,
    'intervention_threshold': 0.5
}

# Evaluation settings
EVALUATION_SETTINGS = {
    'cv_folds': 5,
    'scoring_metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    'save_predictions': True,
    'save_probabilities': True
}

# Visualization settings
VISUALIZATION_SETTINGS = {
    'figure_size': (10, 6),
    'dpi': 300,
    'save_format': 'png',
    'color_palette': 'husl'
}
