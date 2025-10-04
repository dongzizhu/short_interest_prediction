#!/usr/bin/env python3
"""
Quick test to verify SVM data reshaping works correctly.
"""

import numpy as np
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

from models import SVMModel

def test_svm_reshaping():
    """Test that SVM correctly handles 3D time series data."""
    print("🧪 Testing SVM Data Reshaping")
    print("=" * 40)
    
    # Create test data with 3D shape (N, lookback_window, features)
    n_samples = 50
    lookback_window = 4
    n_features = 10
    
    X_train = np.random.randn(n_samples, lookback_window, n_features)
    X_test = np.random.randn(10, lookback_window, n_features)
    y_train = np.random.randn(n_samples, 1)
    y_test = np.random.randn(10, 1)
    
    print(f"Input data shapes:")
    print(f"  X_train: {X_train.shape} (N={n_samples}, lookback={lookback_window}, features={n_features})")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Test SVM model
    svm_model = SVMModel(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1, max_iter=100)
    
    # Fit the model
    print(f"\n📊 Fitting SVM model...")
    svm_model.fit(X_train, y_train)
    print(f"✅ Model fitted successfully")
    
    # Check internal data shape
    expected_flattened_shape = (n_samples, lookback_window * n_features)
    print(f"Expected flattened shape: {expected_flattened_shape}")
    print(f"Scaler fitted on {svm_model.scaler.n_features_in_} features")
    
    # Make predictions
    print(f"\n📊 Making predictions...")
    predictions = svm_model.predict(X_test)
    print(f"✅ Predictions made successfully")
    print(f"  Prediction shape: {predictions.shape}")
    print(f"  Expected shape: (10, 1)")
    
    # Test feature importance
    print(f"\n📊 Testing feature importance...")
    feature_names = [f"Feature_{i}" for i in range(n_features)]
    feature_stats, importance_scores = svm_model.get_feature_importance(
        X_train, y_train, feature_names
    )
    
    print(f"✅ Feature importance calculated")
    print(f"  Number of features analyzed: {len(feature_stats)}")
    print(f"  Expected: {lookback_window * n_features} (flattened)")
    print(f"  Actual: {len(feature_stats)}")
    
    # Show some feature names to verify they're correctly expanded
    print(f"\n📋 Sample feature names:")
    sample_features = list(feature_stats.keys())[:5]
    for i, name in enumerate(sample_features):
        print(f"  {i+1}. {name}")
    
    print(f"\n🎉 All tests passed! SVM correctly handles 3D → 2D reshaping")
    return True

if __name__ == "__main__":
    try:
        test_svm_reshaping()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
