"""
Data Preprocessing Utilities
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Advanced data preprocessing pipeline for returns prediction"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """Fit the preprocessor on training data"""
        df_processed = df.copy()
        
        # Encode product_category
        self.label_encoders['product_category'] = LabelEncoder()
        df_processed['product_category_encoded'] = self.label_encoders['product_category'].fit_transform(
            df_processed['product_category']
        )
        
        # Handle missing sizes
        if df_processed['size_purchased'].notna().any():
            self.most_common_size = df_processed['size_purchased'].mode()[0] if len(df_processed['size_purchased'].mode()) > 0 else 'M'
            df_processed.loc[:, 'size_purchased'] = df_processed['size_purchased'].fillna(self.most_common_size)
            
            self.label_encoders['size'] = LabelEncoder()
            df_processed['size_encoded'] = self.label_encoders['size'].fit_transform(
                df_processed['size_purchased']
            )
        else:
            self.most_common_size = None
            df_processed['size_encoded'] = 0
        
        # Feature engineering
        df_processed = self._engineer_features(df_processed)
        
        # Select features
        self.feature_names = self._get_feature_columns(df_processed)
        X = df_processed[self.feature_names]
        
        # Fit scaler
        self.scaler.fit(X)
        
        return self
    
    def transform(self, df: pd.DataFrame, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Transform data using fitted preprocessor"""
        df_processed = df.copy()
        
        # Encode product_category
        if fit:
            df_processed['product_category_encoded'] = self.label_encoders['product_category'].fit_transform(
                df_processed['product_category']
            )
        else:
            df_processed['product_category_encoded'] = self.label_encoders['product_category'].transform(
                df_processed['product_category']
            )
        
        # Handle missing sizes
        if self.most_common_size is not None:
            df_processed.loc[:, 'size_purchased'] = df_processed['size_purchased'].fillna(self.most_common_size)
            
            if fit:
                df_processed['size_encoded'] = self.label_encoders['size'].fit_transform(
                    df_processed['size_purchased']
                )
            else:
                # Handle unseen categories
                df_processed['size_encoded'] = df_processed['size_purchased'].map(
                    lambda x: self.label_encoders['size'].transform([x])[0] 
                    if x in self.label_encoders['size'].classes_ else -1
                )
        else:
            df_processed['size_encoded'] = 0
        
        # Feature engineering
        df_processed = self._engineer_features(df_processed)
        
        # Select features
        X = df_processed[self.feature_names]
        y = df_processed['is_return']
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled, y.values
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df, fit=True)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        df = df.copy()
        
        # Price-related features
        df['is_high_price'] = (df['product_price'] > df['product_price'].quantile(0.75)).astype(int)
        df['is_low_price'] = (df['product_price'] < df['product_price'].quantile(0.25)).astype(int)
        
        # Customer behavior features
        df['return_rate'] = df['previous_returns'] / (df['customer_tenure_days'] / 30 + 1)
        df['is_frequent_buyer'] = (df['days_since_last_purchase'] < 30).astype(int)
        df['is_new_customer'] = (df['customer_tenure_days'] < 90).astype(int)
        
        # Product quality indicators
        df['is_low_rating'] = (df['product_rating'] < 3.5).astype(int)
        df['is_high_rating'] = (df['product_rating'] >= 4.5).astype(int)
        
        # Interaction features
        df['price_discount_interaction'] = df['product_price'] * df['discount_applied']
        df['age_tenure_interaction'] = df['customer_age'] * df['customer_tenure_days']
        
        return df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get list of feature columns"""
        base_features = [
            'customer_age', 'customer_tenure_days',
            'product_category_encoded', 'product_price',
            'days_since_last_purchase', 'previous_returns',
            'product_rating', 'size_encoded', 'discount_applied'
        ]
        
        engineered_features = [
            'is_high_price', 'is_low_price', 'return_rate',
            'is_frequent_buyer', 'is_new_customer',
            'is_low_rating', 'is_high_rating',
            'price_discount_interaction', 'age_tenure_interaction'
        ]
        
        return base_features + engineered_features
    
    def get_feature_names(self) -> list:
        """Return feature names"""
        return self.feature_names
