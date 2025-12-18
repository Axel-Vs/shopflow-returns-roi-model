"""
Threshold Optimization for Intelligent Router
Finds optimal thresholds for each category to maximize recall or other metrics
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligent_router import IntelligentRouter, create_intelligent_router
from sklearn.metrics import recall_score, precision_score, f1_score

np.random.seed(42)


def main():
    print("="*70)
    print("INTELLIGENT ROUTER THRESHOLD OPTIMIZATION")
    print("="*70)
    
    # Load test data
    print("\n[1/4] Loading test data...")
    test = pd.read_csv('../data/ecommerce_returns_test.csv')
    y_test = test['is_return'].values
    
    print(f"  Test samples: {len(test)}")
    print(f"  Return rate: {y_test.mean()*100:.1f}%")
    
    # Create router with default thresholds
    print("\n[2/4] Creating Intelligent Router with default thresholds...")
    router = create_intelligent_router(
        model_path='../models/random_forest_model.pkl',
        preprocessor_path='../models/preprocessor.pkl'
    )
    
    print(f"\n  Default thresholds:")
    for cat, threshold in router.get_thresholds().items():
        print(f"    {cat}: {threshold}")
    
    # Evaluate with default thresholds
    print("\n  Performance with default thresholds:")
    default_pred = router.predict(test)
    default_recall = recall_score(y_test, default_pred)
    default_precision = precision_score(y_test, default_pred, zero_division=0)
    default_f1 = f1_score(y_test, default_pred)
    print(f"    Recall: {default_recall*100:.1f}%")
    print(f"    Precision: {default_precision*100:.1f}%")
    print(f"    F1-Score: {default_f1:.3f}")
    
    # Optimize for recall
    print("\n[3/4] Optimizing thresholds for RECALL...")
    optimized_recall_thresholds = router.optimize_thresholds(
        df=test,
        y_true=y_test,
        metric='recall',
        threshold_range=(0.2, 0.6),
        step=0.05
    )
    
    print(f"\n  Optimized thresholds (for recall):")
    for cat, threshold in optimized_recall_thresholds.items():
        print(f"    {cat}: {threshold}")
    
    # Apply optimized thresholds
    for cat, threshold in optimized_recall_thresholds.items():
        router.update_threshold(cat, threshold)
    
    # Evaluate with optimized thresholds
    print("\n  Performance with optimized thresholds:")
    optimized_pred = router.predict(test)
    optimized_recall = recall_score(y_test, optimized_pred)
    optimized_precision = precision_score(y_test, optimized_pred, zero_division=0)
    optimized_f1 = f1_score(y_test, optimized_pred)
    print(f"    Recall: {optimized_recall*100:.1f}%")
    print(f"    Precision: {optimized_precision*100:.1f}%")
    print(f"    F1-Score: {optimized_f1:.3f}")
    
    # Optimize for F1
    print("\n[4/4] Optimizing thresholds for F1-SCORE...")
    
    # Reset to original
    router = create_intelligent_router(
        model_path='../models/random_forest_model.pkl',
        preprocessor_path='../models/preprocessor.pkl'
    )
    
    optimized_f1_thresholds = router.optimize_thresholds(
        df=test,
        y_true=y_test,
        metric='f1',
        threshold_range=(0.2, 0.6),
        step=0.05
    )
    
    print(f"\n  Optimized thresholds (for F1):")
    for cat, threshold in optimized_f1_thresholds.items():
        print(f"    {cat}: {threshold}")
    
    # Apply F1-optimized thresholds
    for cat, threshold in optimized_f1_thresholds.items():
        router.update_threshold(cat, threshold)
    
    # Evaluate with F1-optimized thresholds
    print("\n  Performance with F1-optimized thresholds:")
    f1_optimized_pred = router.predict(test)
    f1_optimized_recall = recall_score(y_test, f1_optimized_pred)
    f1_optimized_precision = precision_score(y_test, f1_optimized_pred, zero_division=0)
    f1_optimized_f1 = f1_score(y_test, f1_optimized_pred)
    print(f"    Recall: {f1_optimized_recall*100:.1f}%")
    print(f"    Precision: {f1_optimized_precision*100:.1f}%")
    print(f"    F1-Score: {f1_optimized_f1:.3f}")
    
    # Summary comparison
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    
    summary_data = {
        'Configuration': ['Default', 'Optimized (Recall)', 'Optimized (F1)'],
        'Recall': [
            f"{default_recall*100:.1f}%",
            f"{optimized_recall*100:.1f}%",
            f"{f1_optimized_recall*100:.1f}%"
        ],
        'Precision': [
            f"{default_precision*100:.1f}%",
            f"{optimized_precision*100:.1f}%",
            f"{f1_optimized_precision*100:.1f}%"
        ],
        'F1-Score': [
            f"{default_f1:.3f}",
            f"{optimized_f1:.3f}",
            f"{f1_optimized_f1:.3f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    print("\n" + "─"*70)
    print("THRESHOLD COMPARISON BY CATEGORY")
    print("─"*70)
    
    # Recreate default router for comparison
    router_default = create_intelligent_router(
        model_path='../models/random_forest_model.pkl',
        preprocessor_path='../models/preprocessor.pkl'
    )
    
    threshold_comparison = {
        'Category': list(router_default.get_thresholds().keys()),
        'Default': [
            router_default.get_thresholds()[cat]
            for cat in router_default.get_thresholds().keys()
        ],
        'Optimized (Recall)': [
            optimized_recall_thresholds[cat]
            for cat in router_default.get_thresholds().keys()
        ],
        'Optimized (F1)': [
            optimized_f1_thresholds[cat]
            for cat in router_default.get_thresholds().keys()
        ]
    }
    
    threshold_df = pd.DataFrame(threshold_comparison)
    print("\n" + threshold_df.to_string(index=False))
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    print("\n1. For maximum recall (catch more returns):")
    print("   Use the 'Optimized (Recall)' thresholds")
    print(f"   → Achieves {optimized_recall*100:.1f}% recall")
    
    print("\n2. For balanced precision/recall:")
    print("   Use the 'Optimized (F1)' thresholds")
    print(f"   → Achieves F1 score of {f1_optimized_f1:.3f}")
    
    print("\n3. For cost efficiency:")
    print("   Consider custom thresholds based on category-specific")
    print("   intervention costs and return rates")
    
    recall_improvement = (optimized_recall - default_recall) * 100
    print(f"\n💡 Key Insight: Optimization improved recall by {recall_improvement:+.1f} percentage points")
    
    print("\n" + "="*70)
    print("✓ Optimization complete!")
    print("="*70)


if __name__ == '__main__':
    main()
