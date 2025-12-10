"""
Model Comparison Script
Compare baseline model with enhanced models
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.evaluation import ModelEvaluator
from utils.preprocessing import DataPreprocessor

def load_baseline_predictions():
    """Load baseline model predictions"""
    # Load baseline artifacts
    baseline_model = joblib.load('../src/baseline_model.pkl')
    baseline_scaler = joblib.load('../src/scaler.pkl')
    
    # Load test data
    test = pd.read_csv('../data/ecommerce_returns_test.csv')
    
    # Preprocess using baseline method
    from sklearn.preprocessing import LabelEncoder
    df_processed = test.copy()
    
    le_category = LabelEncoder()
    df_processed['product_category_encoded'] = le_category.fit_transform(
        df_processed['product_category']
    )
    
    if df_processed['size_purchased'].notna().any():
        most_common_size = df_processed['size_purchased'].mode()[0]
        df_processed.loc[:, 'size_purchased'] = df_processed['size_purchased'].fillna(most_common_size)
        
        le_size = LabelEncoder()
        df_processed['size_encoded'] = le_size.fit_transform(
            df_processed['size_purchased']
        )
    
    feature_cols = [
        'customer_age', 'customer_tenure_days',
        'product_category_encoded', 'product_price',
        'days_since_last_purchase', 'previous_returns',
        'product_rating', 'size_encoded', 'discount_applied'
    ]
    
    X = df_processed[feature_cols]
    X_scaled = baseline_scaler.transform(X)
    
    y_pred = baseline_model.predict(X_scaled)
    y_pred_proba = baseline_model.predict_proba(X_scaled)[:, 1]
    
    return test['is_return'].values, y_pred, y_pred_proba


def main():
    print("="*70)
    print("MODEL COMPARISON ANALYSIS")
    print("="*70)
    
    # Load test data
    test = pd.read_csv('../data/ecommerce_returns_test.csv')
    y_test = test['is_return'].values
    
    # Evaluate Baseline Model
    print("\n[1/3] Evaluating Baseline Model...")
    y_test_baseline, y_pred_baseline, y_pred_proba_baseline = load_baseline_predictions()
    
    baseline_evaluator = ModelEvaluator("Baseline Logistic Regression")
    baseline_evaluator.print_evaluation_report(y_test_baseline, y_pred_baseline, y_pred_proba_baseline)
    
    # Evaluate Random Forest
    print("\n[2/3] Evaluating Enhanced Random Forest...")
    rf_model = joblib.load('../models/random_forest_model.pkl')
    preprocessor = joblib.load('../models/preprocessor.pkl')
    
    X_test, y_test = preprocessor.transform(test)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    rf_evaluator = ModelEvaluator("Random Forest (Balanced)")
    rf_evaluator.print_evaluation_report(y_test, rf_pred, rf_pred_proba)
    
    # Evaluate Enhanced Logistic Regression
    print("\n[3/3] Evaluating Enhanced Logistic Regression...")
    lr_model = joblib.load('../models/logistic_regression_balanced_model.pkl')
    
    lr_pred = lr_model.predict(X_test)
    lr_pred_proba = lr_model.predict_proba(X_test)[:, 1]
    
    lr_evaluator = ModelEvaluator("Logistic Regression (Balanced)")
    lr_evaluator.print_evaluation_report(y_test, lr_pred, lr_pred_proba)
    
    # Create comparison table
    print("\n" + "="*70)
    print("FINAL COMPARISON - ALL MODELS")
    print("="*70)
    
    comparison_df = pd.DataFrame({
        'Model': [
            'Baseline LR',
            'Enhanced LR',
            'Random Forest'
        ],
        'Accuracy': [
            baseline_evaluator.metrics['accuracy'],
            lr_evaluator.metrics['accuracy'],
            rf_evaluator.metrics['accuracy']
        ],
        'Precision': [
            baseline_evaluator.metrics['precision'],
            lr_evaluator.metrics['precision'],
            rf_evaluator.metrics['precision']
        ],
        'Recall': [
            baseline_evaluator.metrics['recall'],
            lr_evaluator.metrics['recall'],
            rf_evaluator.metrics['recall']
        ],
        'F1-Score': [
            baseline_evaluator.metrics['f1_score'],
            lr_evaluator.metrics['f1_score'],
            rf_evaluator.metrics['f1_score']
        ],
        'ROC-AUC': [
            baseline_evaluator.metrics.get('roc_auc', 0),
            lr_evaluator.metrics.get('roc_auc', 0),
            rf_evaluator.metrics.get('roc_auc', 0)
        ],
        'Cost Saved ($)': [
            baseline_evaluator.metrics['cost_saved'],
            lr_evaluator.metrics['cost_saved'],
            rf_evaluator.metrics['cost_saved']
        ]
    })
    
    print("\n" + comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Save comparison
    comparison_df.to_csv('../outputs/full_model_comparison.csv', index=False)
    print("\n✓ Comparison saved to outputs/full_model_comparison.csv")
    
    # Highlight improvements
    print("\n" + "="*70)
    print("KEY IMPROVEMENTS")
    print("="*70)
    
    best_model = comparison_df.loc[comparison_df['F1-Score'].idxmax()]
    print(f"\n🏆 Best Model: {best_model['Model']}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {best_model['Accuracy']:.4f}")
    print(f"  Precision: {best_model['Precision']:.4f}")
    print(f"  Recall:    {best_model['Recall']:.4f}")
    print(f"  F1-Score:  {best_model['F1-Score']:.4f}")
    print(f"  ROC-AUC:   {best_model['ROC-AUC']:.4f}")
    print(f"  Cost Saved: ${best_model['Cost Saved ($)']:,.2f}")
    
    # Calculate improvements
    baseline_f1 = comparison_df.loc[0, 'F1-Score']
    best_f1 = best_model['F1-Score']
    improvement = ((best_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else float('inf')
    
    print(f"\n📈 F1-Score Improvement: {improvement:.1f}% over baseline")
    
    baseline_cost = comparison_df.loc[0, 'Cost Saved ($)']
    best_cost = best_model['Cost Saved ($)']
    cost_improvement = best_cost - baseline_cost
    
    print(f"💰 Additional Cost Savings: ${cost_improvement:,.2f}")
    
    print("\n" + "="*70)
    
    # Performance by Product Category
    print("\n" + "="*70)
    print("PERFORMANCE BY PRODUCT CATEGORY")
    print("="*70)
    
    # Analyze by category
    categories = test['product_category'].unique()
    category_results = []
    
    for category in sorted(categories):
        category_mask = test['product_category'] == category
        category_indices = test[category_mask].index
        
        # Get predictions for this category
        y_true_cat = y_test[category_indices]
        rf_pred_cat = rf_pred[category_indices]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        cat_accuracy = accuracy_score(y_true_cat, rf_pred_cat)
        cat_precision = precision_score(y_true_cat, rf_pred_cat, zero_division=0)
        cat_recall = recall_score(y_true_cat, rf_pred_cat, zero_division=0)
        cat_f1 = f1_score(y_true_cat, rf_pred_cat, zero_division=0)
        
        # Count samples and returns
        n_samples = len(y_true_cat)
        n_returns = y_true_cat.sum()
        return_rate = (n_returns / n_samples * 100) if n_samples > 0 else 0
        
        category_results.append({
            'Category': category,
            'Samples': n_samples,
            'Return_Rate_%': return_rate,
            'Accuracy': cat_accuracy,
            'Precision': cat_precision,
            'Recall': cat_recall,
            'F1-Score': cat_f1
        })
    
    category_df = pd.DataFrame(category_results)
    print("\n" + category_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Save category analysis
    category_df.to_csv('../outputs/performance_by_category.csv', index=False)
    print("\n✓ Category analysis saved to outputs/performance_by_category.csv")
    
    # Identify best and worst performing categories
    print("\n" + "─"*70)
    print("CATEGORY INSIGHTS")
    print("─"*70)
    
    best_category = category_df.loc[category_df['F1-Score'].idxmax()]
    worst_category = category_df.loc[category_df['F1-Score'].idxmin()]
    highest_return_rate = category_df.loc[category_df['Return_Rate_%'].idxmax()]
    
    print(f"\n🏆 Best Model Performance:")
    print(f"   Category: {best_category['Category']}")
    print(f"   F1-Score: {best_category['F1-Score']:.4f}")
    print(f"   Recall: {best_category['Recall']:.4f}")
    
    print(f"\n⚠️  Weakest Model Performance:")
    print(f"   Category: {worst_category['Category']}")
    print(f"   F1-Score: {worst_category['F1-Score']:.4f}")
    print(f"   Recall: {worst_category['Recall']:.4f}")
    
    print(f"\n📈 Highest Return Rate:")
    print(f"   Category: {highest_return_rate['Category']}")
    print(f"   Return Rate: {highest_return_rate['Return_Rate_%']:.2f}%")
    print(f"   Samples: {int(highest_return_rate['Samples'])}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
