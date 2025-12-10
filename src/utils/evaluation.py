"""
Model Evaluation Utilities
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation for returns prediction"""
    
    def __init__(self, model_name: str = "Model"):
        self.model_name = model_name
        self.metrics = {}
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_pred_proba: np.ndarray = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of model predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Basic metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        })
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except:
                metrics['roc_auc'] = None
        
        # Business metrics
        metrics.update(self._calculate_business_metrics(y_true, y_pred))
        
        self.metrics = metrics
        return metrics
    
    def _calculate_business_metrics(self, y_true: np.ndarray, 
                                    y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate business-relevant metrics"""
        # Cost assumptions (can be customized)
        cost_of_return = 25.0  # Average cost per return
        cost_of_false_positive = 5.0  # Cost of unnecessary intervention
        revenue_per_sale = 100.0  # Average revenue
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate costs
        cost_missed_returns = fn * cost_of_return
        cost_false_alarms = fp * cost_of_false_positive
        cost_prevented_returns = tp * cost_of_return
        
        total_cost_saved = cost_prevented_returns - cost_false_alarms
        potential_savings = (tp + fn) * cost_of_return
        
        return {
            'cost_saved': total_cost_saved,
            'potential_savings': potential_savings,
            'savings_rate': (total_cost_saved / potential_savings * 100) if potential_savings > 0 else 0,
            'intervention_rate': ((tp + fp) / len(y_true) * 100) if len(y_true) > 0 else 0,
        }
    
    def print_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                               y_pred_proba: np.ndarray = None) -> None:
        """Print a formatted evaluation report"""
        metrics = self.evaluate(y_true, y_pred, y_pred_proba)
        
        print(f"\n{'='*60}")
        print(f"{self.model_name} - Evaluation Report")
        print(f"{'='*60}\n")
        
        # Classification metrics
        print("Classification Metrics:")
        print(f"  Accuracy:     {metrics['accuracy']:.4f}")
        print(f"  Precision:    {metrics['precision']:.4f}")
        print(f"  Recall:       {metrics['recall']:.4f}")
        print(f"  F1-Score:     {metrics['f1_score']:.4f}")
        print(f"  Specificity:  {metrics['specificity']:.4f}")
        if metrics.get('roc_auc'):
            print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {metrics['true_negatives']:>6}")
        print(f"  False Positives: {metrics['false_positives']:>6}")
        print(f"  False Negatives: {metrics['false_negatives']:>6}")
        print(f"  True Positives:  {metrics['true_positives']:>6}")
        
        # Business metrics
        print(f"\nBusiness Impact:")
        print(f"  Cost Saved:           ${metrics['cost_saved']:>10,.2f}")
        print(f"  Potential Savings:    ${metrics['potential_savings']:>10,.2f}")
        print(f"  Savings Rate:         {metrics['savings_rate']:>10.2f}%")
        print(f"  Intervention Rate:    {metrics['intervention_rate']:>10.2f}%")
        
        print(f"\n{'='*60}\n")
        
        # Detailed classification report
        print("Detailed Classification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['No Return', 'Return'],
                                   zero_division=0))
    
    def compare_models(self, evaluators: list) -> pd.DataFrame:
        """
        Compare multiple model evaluators
        
        Args:
            evaluators: List of ModelEvaluator instances with metrics
            
        Returns:
            DataFrame comparing all models
        """
        comparison_data = []
        
        for evaluator in evaluators:
            if evaluator.metrics:
                comparison_data.append({
                    'Model': evaluator.model_name,
                    'Accuracy': evaluator.metrics['accuracy'],
                    'Precision': evaluator.metrics['precision'],
                    'Recall': evaluator.metrics['recall'],
                    'F1-Score': evaluator.metrics['f1_score'],
                    'ROC-AUC': evaluator.metrics.get('roc_auc', np.nan),
                    'Cost Saved': evaluator.metrics['cost_saved'],
                    'Savings Rate': evaluator.metrics['savings_rate'],
                })
        
        df = pd.DataFrame(comparison_data)
        return df
