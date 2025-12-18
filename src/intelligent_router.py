"""
Intelligent Router for Returns Prediction
Routes predictions through category-specific thresholds and strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')


class IntelligentRouter:
    """
    Intelligent routing system for returns prediction that applies
    category-specific thresholds and intervention strategies.
    
    The router improves overall model performance by:
    1. Applying different prediction thresholds per product category
    2. Routing high-risk orders to specialized intervention strategies
    3. Optimizing for business metrics (recall, cost efficiency)
    """
    
    def __init__(
        self,
        model: Any,
        preprocessor: Any,
        category_thresholds: Optional[Dict[str, float]] = None,
        default_threshold: float = 0.5
    ):
        """
        Initialize the intelligent router.
        
        Args:
            model: Trained prediction model (must have predict_proba method)
            preprocessor: Fitted data preprocessor
            category_thresholds: Dict mapping category names to prediction thresholds
            default_threshold: Default threshold for categories without specific settings
        """
        self.model = model
        self.preprocessor = preprocessor
        self.default_threshold = default_threshold
        
        # Default category-specific thresholds optimized for recall
        # Lower thresholds for categories with poor performance
        self.category_thresholds = category_thresholds or {
            'Fashion': 0.45,        # Already good performance, keep moderate
            'Electronics': 0.30,     # Very poor performance, aggressive threshold
            'Home_Decor': 0.35,     # Poor performance, lower threshold
        }
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using category-specific thresholds.
        
        Args:
            df: DataFrame with features including 'product_category'
            
        Returns:
            Binary predictions (0 or 1) using intelligent routing
        """
        # Get probability predictions
        probabilities = self.predict_proba(df)
        
        # Get categories for each sample
        categories = df['product_category'].values
        
        # Apply category-specific thresholds
        predictions = np.zeros(len(df), dtype=int)
        for category in self.category_thresholds.keys():
            mask = categories == category
            threshold = self.category_thresholds.get(category, self.default_threshold)
            predictions[mask] = (probabilities[mask] >= threshold).astype(int)
        
        # Handle any unconfigured categories with default threshold
        unconfigured_mask = ~np.isin(categories, list(self.category_thresholds.keys()))
        if unconfigured_mask.any():
            predictions[unconfigured_mask] = (
                probabilities[unconfigured_mask] >= self.default_threshold
            ).astype(int)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions from the model.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Array of probabilities for the positive class
        """
        # Preprocess data
        X, _ = self.preprocessor.transform(df)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return probabilities
    
    def predict_with_strategy(
        self, 
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions and recommend intervention strategies.
        
        Args:
            df: DataFrame with features including 'product_category'
            
        Returns:
            Tuple of (predictions, probabilities, strategies)
            - predictions: Binary predictions (0 or 1)
            - probabilities: Probability scores
            - strategies: Recommended intervention strategies
        """
        probabilities = self.predict_proba(df)
        predictions = self.predict(df)
        
        # Determine intervention strategies based on probability and category
        strategies = self._determine_strategies(df, probabilities)
        
        return predictions, probabilities, strategies
    
    def _determine_strategies(
        self, 
        df: pd.DataFrame, 
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        Determine intervention strategy for each order.
        
        Strategies:
        - 'none': No intervention needed (low risk)
        - 'email': Automated email intervention (medium risk)
        - 'phone': Phone call intervention (high risk)
        - 'discount': Offer discount/incentive (very high risk)
        
        Args:
            df: DataFrame with features including 'product_category'
            probabilities: Prediction probabilities
            
        Returns:
            Array of strategy recommendations
        """
        strategies = np.full(len(df), 'none', dtype=object)
        categories = df['product_category'].values
        
        for i in range(len(df)):
            prob = probabilities[i]
            category = categories[i]
            threshold = self.category_thresholds.get(
                category, 
                self.default_threshold
            )
            
            # Below threshold - no intervention
            if prob < threshold:
                strategies[i] = 'none'
            # Low-medium risk - automated email
            elif prob < threshold + 0.15:
                strategies[i] = 'email'
            # Medium-high risk - phone intervention
            elif prob < threshold + 0.30:
                strategies[i] = 'phone'
            # Very high risk - offer discount/incentive
            else:
                strategies[i] = 'discount'
        
        return strategies
    
    def evaluate_by_category(
        self, 
        df: pd.DataFrame, 
        y_true: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate router performance by category.
        
        Args:
            df: DataFrame with features including 'product_category'
            y_true: True labels
            
        Returns:
            DataFrame with per-category performance metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        predictions = self.predict(df)
        probabilities = self.predict_proba(df)
        categories = df['product_category'].values
        
        results = []
        for category in np.unique(categories):
            mask = categories == category
            
            if mask.sum() == 0:
                continue
            
            y_true_cat = y_true[mask]
            y_pred_cat = predictions[mask]
            
            # Calculate metrics
            metrics = {
                'category': category,
                'threshold': self.category_thresholds.get(
                    category, 
                    self.default_threshold
                ),
                'samples': mask.sum(),
                'return_rate': y_true_cat.mean(),
                'accuracy': accuracy_score(y_true_cat, y_pred_cat),
                'precision': precision_score(
                    y_true_cat, y_pred_cat, zero_division=0
                ),
                'recall': recall_score(
                    y_true_cat, y_pred_cat, zero_division=0
                ),
                'f1_score': f1_score(
                    y_true_cat, y_pred_cat, zero_division=0
                ),
            }
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def update_threshold(self, category: str, threshold: float) -> None:
        """
        Update threshold for a specific category.
        
        Args:
            category: Product category name
            threshold: New prediction threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        
        self.category_thresholds[category] = threshold
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current category thresholds."""
        return self.category_thresholds.copy()
    
    def optimize_thresholds(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        metric: str = 'recall',
        threshold_range: Tuple[float, float] = (0.2, 0.6),
        step: float = 0.05
    ) -> Dict[str, float]:
        """
        Optimize thresholds for each category to maximize a specific metric.
        
        Args:
            df: DataFrame with features including 'product_category'
            y_true: True labels
            metric: Metric to optimize ('recall', 'f1', 'precision')
            threshold_range: Range of thresholds to try (min, max)
            step: Step size for threshold search
            
        Returns:
            Dictionary of optimized thresholds per category
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score
        )
        
        metric_funcs = {
            'recall': recall_score,
            'f1': f1_score,
            'precision': precision_score,
        }
        
        if metric not in metric_funcs:
            raise ValueError(f"Metric must be one of {list(metric_funcs.keys())}")
        
        metric_func = metric_funcs[metric]
        categories = df['product_category'].values
        probabilities = self.predict_proba(df)
        
        optimized_thresholds = {}
        
        # Optimize threshold for each category
        for category in np.unique(categories):
            mask = categories == category
            y_true_cat = y_true[mask]
            proba_cat = probabilities[mask]
            
            best_threshold = self.default_threshold
            best_score = -1.0  # Initialize to -1 to ensure first valid threshold is selected
            
            # Search for best threshold
            thresholds = np.arange(
                threshold_range[0], 
                threshold_range[1] + step, 
                step
            )
            
            for threshold in thresholds:
                y_pred_cat = (proba_cat >= threshold).astype(int)
                score = metric_func(y_true_cat, y_pred_cat, zero_division=0)
                
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            optimized_thresholds[category] = round(best_threshold, 2)
        
        return optimized_thresholds
    
    def save(self, filepath: str) -> None:
        """Save the router to a file."""
        joblib.dump(self, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'IntelligentRouter':
        """Load a router from a file."""
        return joblib.load(filepath)


def create_intelligent_router(
    model_path: str,
    preprocessor_path: str,
    category_thresholds: Optional[Dict[str, float]] = None
) -> IntelligentRouter:
    """
    Convenience function to create an intelligent router from saved artifacts.
    
    Args:
        model_path: Path to saved model file
        preprocessor_path: Path to saved preprocessor file
        category_thresholds: Optional dict of category-specific thresholds
        
    Returns:
        Configured IntelligentRouter instance
    """
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    return IntelligentRouter(
        model=model,
        preprocessor=preprocessor,
        category_thresholds=category_thresholds
    )
