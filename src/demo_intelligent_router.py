"""
Demonstration of Intelligent Router
Shows how to use the router and compare with baseline predictions
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intelligent_router import IntelligentRouter, create_intelligent_router
from utils.evaluation import ModelEvaluator
from sklearn.metrics import confusion_matrix, classification_report

np.random.seed(42)


def main():
    print("="*70)
    print("INTELLIGENT ROUTER DEMONSTRATION")
    print("="*70)
    
    # Load test data
    print("\n[1/5] Loading test data...")
    test = pd.read_csv('../data/ecommerce_returns_test.csv')
    y_test = test['is_return'].values
    
    print(f"  Test samples: {len(test)}")
    print(f"  Categories: {test['product_category'].unique()}")
    print(f"  Return rate: {y_test.mean()*100:.1f}%")
    
    # Create intelligent router
    print("\n[2/5] Creating Intelligent Router...")
    router = create_intelligent_router(
        model_path='../models/random_forest_model.pkl',
        preprocessor_path='../models/preprocessor.pkl'
    )
    
    print(f"  Loaded model and preprocessor")
    print(f"  Category thresholds:")
    for cat, threshold in router.get_thresholds().items():
        print(f"    {cat}: {threshold}")
    
    # Make predictions with router
    print("\n[3/5] Making predictions with Intelligent Router...")
    router_pred, router_proba, strategies = router.predict_with_strategy(test)
    
    print(f"  Predictions made: {len(router_pred)}")
    print(f"  Positive predictions: {router_pred.sum()} ({router_pred.sum()/len(router_pred)*100:.1f}%)")
    
    # Strategy distribution
    print(f"\n  Intervention strategies recommended:")
    strategy_counts = pd.Series(strategies).value_counts()
    for strategy, count in strategy_counts.items():
        print(f"    {strategy}: {count} ({count/len(strategies)*100:.1f}%)")
    
    # Evaluate router performance
    print("\n[4/5] Evaluating Router Performance...")
    print("\n" + "─"*70)
    print("OVERALL PERFORMANCE WITH INTELLIGENT ROUTER")
    print("─"*70)
    
    evaluator = ModelEvaluator("Intelligent Router")
    evaluator.print_evaluation_report(y_test, router_pred, router_proba)
    
    # Compare by category
    print("\n" + "─"*70)
    print("PERFORMANCE BY CATEGORY")
    print("─"*70)
    
    category_results = router.evaluate_by_category(test, y_test)
    print(category_results.to_string(index=False))
    
    # Get baseline predictions (standard 0.5 threshold)
    print("\n[5/5] Comparing with baseline (0.5 threshold)...")
    baseline_proba = router.predict_proba(test)
    baseline_pred = (baseline_proba >= 0.5).astype(int)
    
    print("\n" + "─"*70)
    print("BASELINE PERFORMANCE (Fixed 0.5 threshold)")
    print("─"*70)
    
    baseline_evaluator = ModelEvaluator("Baseline (0.5 threshold)")
    baseline_evaluator.print_evaluation_report(y_test, baseline_pred, baseline_proba)
    
    # Comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    from sklearn.metrics import recall_score, precision_score, f1_score
    
    router_recall = recall_score(y_test, router_pred)
    baseline_recall = recall_score(y_test, baseline_pred)
    recall_improvement = (router_recall - baseline_recall) * 100
    
    router_precision = precision_score(y_test, router_pred, zero_division=0)
    baseline_precision = precision_score(y_test, baseline_pred, zero_division=0)
    
    router_f1 = f1_score(y_test, router_pred)
    baseline_f1 = f1_score(y_test, baseline_pred)
    
    print(f"\nRecall (Catch Rate):")
    print(f"  Baseline: {baseline_recall*100:.1f}%")
    print(f"  Router:   {router_recall*100:.1f}%")
    print(f"  Improvement: {recall_improvement:+.1f} percentage points")
    
    print(f"\nPrecision:")
    print(f"  Baseline: {baseline_precision*100:.1f}%")
    print(f"  Router:   {router_precision*100:.1f}%")
    
    print(f"\nF1-Score:")
    print(f"  Baseline: {baseline_f1:.3f}")
    print(f"  Router:   {router_f1:.3f}")
    
    # Business impact
    print("\n" + "─"*70)
    print("BUSINESS IMPACT ESTIMATION")
    print("─"*70)
    
    # Calculate confusion matrices
    cm_baseline = confusion_matrix(y_test, baseline_pred)
    cm_router = confusion_matrix(y_test, router_pred)
    
    tn_base, fp_base, fn_base, tp_base = cm_baseline.ravel()
    tn_router, fp_router, fn_router, tp_router = cm_router.ravel()
    
    # Business costs (from README)
    return_cost = 18.0  # Cost when a return happens
    intervention_cost = 3.0  # Cost of intervention
    intervention_effectiveness = 0.35  # 35% of interventions prevent returns
    
    # Calculate business metrics
    # Revenue = successful interventions * net savings per intervention
    revenue_base = tp_base * intervention_effectiveness * (return_cost - intervention_cost)
    revenue_router = tp_router * intervention_effectiveness * (return_cost - intervention_cost)
    
    # Costs = FP interventions + FN missed returns
    cost_base = (fp_base * intervention_cost) + (fn_base * return_cost)
    cost_router = (fp_router * intervention_cost) + (fn_router * return_cost)
    
    net_profit_base = revenue_base - cost_base
    net_profit_router = revenue_router - cost_router
    
    print(f"\nPer {len(test)} orders:")
    print(f"\nBaseline (0.5 threshold):")
    print(f"  True Positives: {tp_base}")
    print(f"  False Positives: {fp_base}")
    print(f"  False Negatives: {fn_base}")
    print(f"  Revenue: ${revenue_base:.2f}")
    print(f"  Costs: ${cost_base:.2f}")
    print(f"  Net Profit: ${net_profit_base:.2f}")
    
    print(f"\nIntelligent Router:")
    print(f"  True Positives: {tp_router}")
    print(f"  False Positives: {fp_router}")
    print(f"  False Negatives: {fn_router}")
    print(f"  Revenue: ${revenue_router:.2f}")
    print(f"  Costs: ${cost_router:.2f}")
    print(f"  Net Profit: ${net_profit_router:.2f}")
    
    profit_improvement = net_profit_router - net_profit_base
    print(f"\n💰 Profit Improvement: ${profit_improvement:+.2f}")
    
    if profit_improvement > 0:
        print(f"   ({profit_improvement/abs(net_profit_base)*100:+.1f}% better)")
    
    # Annualized projection
    orders_per_year = 100000
    scaling_factor = orders_per_year / len(test)
    annual_improvement = profit_improvement * scaling_factor
    
    print(f"\n📊 Annualized Projection (100K orders/year):")
    print(f"   Additional profit: ${annual_improvement:+,.0f}/year")
    
    print("\n" + "="*70)
    print("✓ Demonstration complete!")
    print("="*70)
    
    # Save router
    print("\nSaving Intelligent Router to models/intelligent_router.pkl...")
    router.save('../models/intelligent_router.pkl')
    print("✓ Router saved successfully")


if __name__ == '__main__':
    main()
