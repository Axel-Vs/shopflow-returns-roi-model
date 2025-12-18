"""
Unit tests for Intelligent Router
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
import joblib
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from intelligent_router import IntelligentRouter, create_intelligent_router


@pytest.fixture
def mock_model():
    """Create a mock model with predict_proba method"""
    model = Mock()
    model.predict_proba = Mock(
        return_value=np.array([
            [0.6, 0.4],
            [0.3, 0.7],
            [0.8, 0.2],
            [0.4, 0.6],
            [0.5, 0.5],
        ])
    )
    return model


@pytest.fixture
def mock_preprocessor():
    """Create a mock preprocessor"""
    preprocessor = Mock()
    preprocessor.transform = Mock(
        return_value=(
            np.random.randn(5, 10),  # X
            np.array([0, 1, 0, 1, 1])  # y
        )
    )
    preprocessor.get_feature_names = Mock(
        return_value=['feature1', 'feature2', 'feature3']
    )
    return preprocessor


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'product_category': ['Fashion', 'Electronics', 'Home_Decor', 'Fashion', 'Electronics'],
        'product_price': [50, 200, 100, 75, 150],
        'customer_age': [25, 35, 45, 30, 40],
        'is_return': [0, 1, 0, 1, 1],
    })


@pytest.fixture
def router(mock_model, mock_preprocessor):
    """Create IntelligentRouter instance"""
    return IntelligentRouter(
        model=mock_model,
        preprocessor=mock_preprocessor,
        category_thresholds={
            'Fashion': 0.45,
            'Electronics': 0.30,
            'Home_Decor': 0.35,
        }
    )


class TestIntelligentRouterInitialization:
    """Test router initialization"""
    
    def test_initialization_with_custom_thresholds(self, mock_model, mock_preprocessor):
        """Test router can be initialized with custom thresholds"""
        custom_thresholds = {
            'Fashion': 0.5,
            'Electronics': 0.3,
        }
        router = IntelligentRouter(
            model=mock_model,
            preprocessor=mock_preprocessor,
            category_thresholds=custom_thresholds
        )
        
        assert router.category_thresholds == custom_thresholds
        assert router.default_threshold == 0.5
    
    def test_initialization_with_default_thresholds(self, mock_model, mock_preprocessor):
        """Test router uses default thresholds if none provided"""
        router = IntelligentRouter(
            model=mock_model,
            preprocessor=mock_preprocessor
        )
        
        assert 'Fashion' in router.category_thresholds
        assert 'Electronics' in router.category_thresholds
        assert 'Home_Decor' in router.category_thresholds


class TestPrediction:
    """Test prediction functionality"""
    
    def test_predict_proba(self, router, sample_data):
        """Test probability prediction"""
        probabilities = router.predict_proba(sample_data)
        
        assert isinstance(probabilities, np.ndarray)
        assert len(probabilities) == len(sample_data)
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_predict_with_category_thresholds(self, router, sample_data):
        """Test predictions use category-specific thresholds"""
        predictions = router.predict(sample_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_with_strategy(self, router, sample_data):
        """Test prediction with intervention strategies"""
        predictions, probabilities, strategies = router.predict_with_strategy(sample_data)
        
        assert len(predictions) == len(sample_data)
        assert len(probabilities) == len(sample_data)
        assert len(strategies) == len(sample_data)
        
        # Check strategy types
        valid_strategies = {'none', 'email', 'phone', 'discount'}
        assert all(s in valid_strategies for s in strategies)


class TestStrategyDetermination:
    """Test intervention strategy logic"""
    
    def test_strategy_none_for_low_probability(self, router, sample_data):
        """Test 'none' strategy for low probability predictions"""
        # Mock low probabilities
        router.model.predict_proba = Mock(
            return_value=np.array([
                [0.9, 0.1],  # Very low probability
                [0.85, 0.15],
                [0.9, 0.1],
                [0.85, 0.15],
                [0.9, 0.1],
            ])
        )
        
        _, _, strategies = router.predict_with_strategy(sample_data)
        
        # Most should be 'none' due to low probability
        assert strategies.tolist().count('none') > 0
    
    def test_strategy_escalation_with_probability(self, router, sample_data):
        """Test strategy escalates with probability"""
        # Mock varying probabilities
        router.model.predict_proba = Mock(
            return_value=np.array([
                [0.3, 0.7],   # High probability
                [0.1, 0.9],   # Very high probability
                [0.6, 0.4],   # Medium probability
                [0.5, 0.5],   # Medium probability
                [0.8, 0.2],   # Low probability
            ])
        )
        
        _, _, strategies = router.predict_with_strategy(sample_data)
        
        # Should have different strategies based on probabilities
        unique_strategies = set(strategies)
        assert len(unique_strategies) >= 2  # At least 2 different strategies


class TestThresholdManagement:
    """Test threshold update and retrieval"""
    
    def test_get_thresholds(self, router):
        """Test getting current thresholds"""
        thresholds = router.get_thresholds()
        
        assert isinstance(thresholds, dict)
        assert 'Fashion' in thresholds
        assert thresholds['Fashion'] == 0.45
    
    def test_update_threshold(self, router):
        """Test updating threshold for a category"""
        router.update_threshold('Fashion', 0.55)
        
        assert router.category_thresholds['Fashion'] == 0.55
    
    def test_update_threshold_invalid(self, router):
        """Test updating threshold with invalid value raises error"""
        with pytest.raises(ValueError):
            router.update_threshold('Fashion', 1.5)
        
        with pytest.raises(ValueError):
            router.update_threshold('Fashion', -0.1)


class TestCategoryEvaluation:
    """Test category-specific evaluation"""
    
    def test_evaluate_by_category(self, router, sample_data):
        """Test per-category evaluation"""
        y_true = sample_data['is_return'].values
        
        results = router.evaluate_by_category(sample_data, y_true)
        
        assert isinstance(results, pd.DataFrame)
        assert 'category' in results.columns
        assert 'recall' in results.columns
        assert 'precision' in results.columns
        assert len(results) > 0


class TestThresholdOptimization:
    """Test threshold optimization"""
    
    def test_optimize_thresholds_recall(self, router, sample_data):
        """Test threshold optimization for recall"""
        y_true = sample_data['is_return'].values
        
        optimized = router.optimize_thresholds(
            df=sample_data,
            y_true=y_true,
            metric='recall',
            threshold_range=(0.2, 0.6),
            step=0.1
        )
        
        assert isinstance(optimized, dict)
        assert len(optimized) > 0
        assert all(0.0 <= t <= 1.0 for t in optimized.values())
    
    def test_optimize_thresholds_f1(self, router, sample_data):
        """Test threshold optimization for F1"""
        y_true = sample_data['is_return'].values
        
        optimized = router.optimize_thresholds(
            df=sample_data,
            y_true=y_true,
            metric='f1',
            threshold_range=(0.2, 0.6),
            step=0.1
        )
        
        assert isinstance(optimized, dict)
        assert all(isinstance(t, float) for t in optimized.values())
    
    def test_optimize_thresholds_invalid_metric(self, router, sample_data):
        """Test invalid metric raises error"""
        y_true = sample_data['is_return'].values
        
        with pytest.raises(ValueError):
            router.optimize_thresholds(
                df=sample_data,
                y_true=y_true,
                metric='invalid_metric'
            )


class TestSaveLoad:
    """Test save and load functionality"""
    
    @pytest.mark.skip(reason="Cannot pickle mock objects; tested with integration test")
    def test_save_and_load(self, router, tmp_path):
        """Test saving and loading router"""
        filepath = tmp_path / "test_router.pkl"
        
        # Save
        router.save(str(filepath))
        assert filepath.exists()
        
        # Load
        loaded_router = IntelligentRouter.load(str(filepath))
        
        # Verify thresholds are preserved
        assert loaded_router.get_thresholds() == router.get_thresholds()
        assert loaded_router.default_threshold == router.default_threshold


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_predict_with_unknown_category(self, router, sample_data):
        """Test prediction with unknown category uses default threshold"""
        sample_data_with_unknown = sample_data.copy()
        sample_data_with_unknown.loc[0, 'product_category'] = 'Unknown_Category'
        
        predictions = router.predict(sample_data_with_unknown)
        
        # Should not raise error and return predictions
        assert len(predictions) == len(sample_data_with_unknown)
    
    def test_empty_dataframe(self, router):
        """Test router handles empty DataFrame"""
        empty_df = pd.DataFrame(columns=['product_category', 'is_return'])
        
        # Update mock to return empty array for empty input
        router.preprocessor.transform = Mock(
            return_value=(
                np.array([]).reshape(0, 10),  # Empty X with proper shape
                np.array([])  # Empty y
            )
        )
        router.model.predict_proba = Mock(
            return_value=np.array([]).reshape(0, 2)  # Empty proba array
        )
        
        probabilities = router.predict_proba(empty_df)
        
        assert len(probabilities) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
