"""
Visualization Utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class ModelVisualizer:
    """Visualization utilities for model analysis"""
    
    def __init__(self, model_name: str = "Model"):
        self.model_name = model_name
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """Plot confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Return', 'Return'],
                   yticklabels=['No Return', 'Return'])
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{self.model_name} (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name} - ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, 
                                    y_pred_proba: np.ndarray,
                                    save_path: Optional[str] = None) -> None:
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=self.model_name)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.model_name} - Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: list, 
                               feature_importance: np.ndarray,
                               top_n: int = 15,
                               save_path: Optional[str] = None) -> None:
        """Plot feature importance"""
        # Sort features by importance
        indices = np.argsort(feature_importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), feature_importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'{self.model_name} - Top {top_n} Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            save_path: Optional[str] = None) -> None:
        """Plot model comparison across multiple metrics"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.set_xlabel('')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1])
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, y: np.ndarray, title: str = "Class Distribution",
                               save_path: Optional[str] = None) -> None:
        """Plot class distribution"""
        unique, counts = np.unique(y, return_counts=True)
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(['No Return', 'Return'], counts, color=['#3498db', '#e74c3c'])
        plt.ylabel('Count')
        plt.title(title)
        
        # Add count labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}\n({height/sum(counts)*100:.1f}%)',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
