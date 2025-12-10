"""
Enhanced Model - Random Forest with Class Imbalance Handling
Improved version with better preprocessing, evaluation, and class balance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import DataPreprocessor
from utils.evaluation import ModelEvaluator
from utils.visualization import ModelVisualizer

# Set random seed for reproducibility
np.random.seed(42)

def main():
    print("="*60)
    print("ENHANCED MODEL - Returns Prediction with Class Balance")
    print("="*60)
    
    # Load data
    print("\n[1/6] Loading data...")
    train = pd.read_csv('../data/ecommerce_returns_train.csv')
    test = pd.read_csv('../data/ecommerce_returns_test.csv')
    
    print(f"  Training samples: {len(train)}")
    print(f"  Test samples: {len(test)}")
    print(f"  Training class distribution:")
    print(f"    No Return: {(train['is_return'] == 0).sum()} ({(train['is_return'] == 0).mean()*100:.1f}%)")
    print(f"    Return: {(train['is_return'] == 1).sum()} ({(train['is_return'] == 1).mean()*100:.1f}%)")
    
    # Preprocess data
    print("\n[2/6] Preprocessing data with feature engineering...")
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train)
    X_test, y_test = preprocessor.transform(test)
    
    print(f"  Features: {len(preprocessor.get_feature_names())}")
    print(f"  Feature names: {', '.join(preprocessor.get_feature_names()[:5])}...")
    
    # Train Random Forest with class weight balancing
    print("\n[3/6] Training Random Forest with class_weight='balanced'...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("  ✓ Random Forest trained")
    
    # Train Logistic Regression with class weight balancing
    print("\n[4/6] Training Logistic Regression with class_weight='balanced'...")
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
        random_state=42
    )
    lr_model.fit(X_train, y_train)
    print("  ✓ Logistic Regression trained")
    
    # Evaluate Random Forest
    print("\n[5/6] Evaluating models...")
    print("\n" + "─"*60)
    print("RANDOM FOREST RESULTS")
    print("─"*60)
    
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    rf_evaluator = ModelEvaluator("Random Forest (Balanced)")
    rf_evaluator.print_evaluation_report(y_test, rf_pred, rf_pred_proba)
    
    # Evaluate Logistic Regression
    print("\n" + "─"*60)
    print("LOGISTIC REGRESSION RESULTS")
    print("─"*60)
    
    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    lr_evaluator = ModelEvaluator("Logistic Regression (Balanced)")
    lr_evaluator.print_evaluation_report(y_test, lr_pred, lr_pred_proba)
    
    # Feature importance for Random Forest
    print("\n[6/6] Analyzing feature importance...")
    feature_importance = rf_model.feature_importances_
    feature_names = preprocessor.get_feature_names()
    
    # Get top 10 features
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    print("\nTop 10 Most Important Features:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:30s} {feature_importance[idx]:.4f}")
    
    # Save models and preprocessor
    print("\n" + "="*60)
    print("Saving models and artifacts...")
    print("="*60)
    
    os.makedirs('../models', exist_ok=True)
    
    joblib.dump(rf_model, '../models/random_forest_model.pkl')
    joblib.dump(lr_model, '../models/logistic_regression_balanced_model.pkl')
    joblib.dump(preprocessor, '../models/preprocessor.pkl')
    
    # Save feature importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    feature_importance_df.to_csv('../outputs/feature_importance.csv', index=False)
    
    print("  ✓ Random Forest model saved")
    print("  ✓ Logistic Regression model saved")
    print("  ✓ Preprocessor saved")
    print("  ✓ Feature importance saved")
    
    # Model comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_data = {
        'Model': ['Random Forest', 'Logistic Regression'],
        'Accuracy': [rf_evaluator.metrics['accuracy'], lr_evaluator.metrics['accuracy']],
        'Precision': [rf_evaluator.metrics['precision'], lr_evaluator.metrics['precision']],
        'Recall': [rf_evaluator.metrics['recall'], lr_evaluator.metrics['recall']],
        'F1-Score': [rf_evaluator.metrics['f1_score'], lr_evaluator.metrics['f1_score']],
        'ROC-AUC': [rf_evaluator.metrics['roc_auc'], lr_evaluator.metrics['roc_auc']],
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv('../outputs/model_comparison.csv', index=False)
    
    print("\n" + "="*60)
    print("✓ ENHANCED MODEL TRAINING COMPLETE")
    print("="*60)
    print("\nKey Improvements over Baseline:")
    print("  • Feature engineering (18 features vs 9)")
    print("  • Class weight balancing to handle imbalanced data")
    print("  • Multiple model comparison")
    print("  • Comprehensive evaluation metrics")
    print("  • Business impact analysis")
    print("\nNext steps:")
    print("  • Review feature importance")
    print("  • Analyze model predictions")
    print("  • Consider hyperparameter tuning")
    print("  • Explore ensemble methods")
    print("="*60)


if __name__ == "__main__":
    main()
