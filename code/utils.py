"""
Utility functions for the iterative agent-based feature selection system.

This module contains helper functions, data validation, and common utilities
used across the system.
"""

import numpy as np
import pandas as pd
import pickle
import warnings
from typing import Tuple, List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import json


def setup_logging(verbose: bool = True, log_level: str = 'INFO'):
    """Set up logging configuration."""
    import logging
    
    if verbose:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def validate_data_shape(data: np.ndarray, expected_shape: Tuple[int, ...], name: str = "data") -> bool:
    """Validate that data has the expected shape."""
    if data.shape != expected_shape:
        raise ValueError(f"{name} has shape {data.shape}, expected {expected_shape}")
    return True


def validate_feature_engineering_output(output: np.ndarray, input_shape: Tuple[int, int]) -> bool:
    """Validate that feature engineering output has correct format."""
    if not isinstance(output, np.ndarray):
        raise ValueError("Feature engineering function must return a numpy array")
    
    if output.ndim != 2:
        raise ValueError(f"Feature engineering output must be 2D, got {output.ndim}D")
    
    if output.shape[0] != input_shape[0]:
        raise ValueError(f"Output timesteps ({output.shape[0]}) must match input timesteps ({input_shape[0]})")
    
    return True


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Safely divide arrays, handling division by zero."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.divide(numerator, denominator, out=np.full_like(numerator, default, dtype=float), where=denominator!=0)
    return result


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate returns from price series."""
    if len(prices) < 2:
        return np.array([0.0])
    
    returns = np.diff(prices) / prices[:-1]
    return np.concatenate([[0.0], returns])  # First return is 0


def calculate_volatility(returns: np.ndarray, window: int = 5) -> np.ndarray:
    """Calculate rolling volatility of returns."""
    if len(returns) < window:
        return np.full(len(returns), np.std(returns))
    
    volatility = np.zeros_like(returns)
    for i in range(len(returns)):
        start_idx = max(0, i - window + 1)
        volatility[i] = np.std(returns[start_idx:i+1])
    
    return volatility


def calculate_momentum(prices: np.ndarray, periods: List[int]) -> np.ndarray:
    """Calculate momentum for different periods."""
    momentum_features = np.zeros((len(prices), len(periods)))
    
    for i, period in enumerate(periods):
        if len(prices) > period:
            momentum = (prices[-1] - prices[-period-1]) / prices[-period-1]
            momentum_features[:, i] = momentum
    
    return momentum_features


def calculate_technical_indicators(ohlc_data: np.ndarray) -> Dict[str, np.ndarray]:
    """Calculate common technical indicators from OHLC data."""
    if ohlc_data.shape[1] != 4:
        raise ValueError("OHLC data must have 4 columns: [open, high, low, close]")
    
    open_prices = ohlc_data[:, 0]
    high_prices = ohlc_data[:, 1]
    low_prices = ohlc_data[:, 2]
    close_prices = ohlc_data[:, 3]
    
    indicators = {}
    
    # Simple Moving Averages
    for period in [5, 10, 20]:
        if len(close_prices) >= period:
            sma = np.convolve(close_prices, np.ones(period)/period, mode='valid')
            # Pad with NaN or last value
            sma = np.concatenate([np.full(period-1, np.nan), sma])
            indicators[f'sma_{period}'] = sma
    
    # RSI (simplified)
    if len(close_prices) > 1:
        returns = np.diff(close_prices)
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        indicators['rsi'] = np.full(len(close_prices), rsi)
    
    # Bollinger Bands
    if len(close_prices) >= 20:
        sma_20 = np.mean(close_prices[-20:])
        std_20 = np.std(close_prices[-20:])
        indicators['bb_upper'] = sma_20 + 2 * std_20
        indicators['bb_lower'] = sma_20 - 2 * std_20
        indicators['bb_middle'] = sma_20
    else:
        indicators['bb_upper'] = close_prices[-1]
        indicators['bb_lower'] = close_prices[-1]
        indicators['bb_middle'] = close_prices[-1]
    
    # Volume indicators (if volume data available)
    if ohlc_data.shape[1] >= 5:  # Assuming volume is 5th column
        volume = ohlc_data[:, 4]
        indicators['volume_sma'] = np.mean(volume)
        indicators['volume_ratio'] = volume[-1] / np.mean(volume) if np.mean(volume) > 0 else 1.0
    
    return indicators


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """Save results to file (pickle or JSON)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if filepath.suffix == '.pkl':
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif filepath.suffix == '.json':
        # Convert numpy arrays to lists for JSON serialization
        json_results = convert_numpy_to_json(results)
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from file."""
    filepath = Path(filepath)
    
    if filepath.suffix == '.pkl':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def convert_numpy_to_json(obj: Any) -> Any:
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_json(item) for item in obj]
    else:
        return obj


def format_performance_summary(results: List[Dict[str, Any]]) -> str:
    """Format performance results into a readable summary."""
    if not results:
        return "No results to display"
    
    summary = []
    summary.append("=" * 80)
    summary.append("PERFORMANCE SUMMARY")
    summary.append("=" * 80)
    
    for i, result in enumerate(results):
        summary.append(f"\nIteration {i}: {result.get('model_name', 'Unknown')}")
        summary.append(f"  MAPE: {result.get('mape', 0):.2f}%")
        summary.append(f"  MAE: {result.get('mae', 0):.4f}")
        summary.append(f"  RMSE: {result.get('rmse', 0):.4f}")
        if 'improvement' in result:
            summary.append(f"  Improvement: {result['improvement']:+.2f}%")
    
    return "\n".join(summary)


def create_timestamp() -> str:
    """Create a timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """Print progress information."""
    percentage = (current / total) * 100
    print(f"{prefix}: {current}/{total} ({percentage:.1f}%)")


def validate_config(config: Any) -> bool:
    """Validate configuration object."""
    required_attrs = ['data', 'model', 'llm', 'evaluation', 'system']
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Config missing required attribute: {attr}")
    
    # Validate data config
    if config.data.train_split > config.data.val_split:
        raise ValueError("Train split index is greater than validation split index")
    
    # Validate model config
    if config.model.epochs <= 0:
        raise ValueError("Epochs must be positive")
    
    # Validate LLM config
    if config.llm.max_iterations <= 0:
        raise ValueError("Max iterations must be positive")
    
    return True


def get_feature_names(num_features: int, prefix: str = "Feature") -> List[str]:
    """Generate feature names for a given number of features."""
    return [f"{prefix}_{i}" for i in range(num_features)]


def calculate_improvement_metrics(baseline_mape: float, enhanced_mape: float) -> Dict[str, float]:
    """Calculate improvement metrics between baseline and enhanced models."""
    absolute_improvement = baseline_mape - enhanced_mape
    relative_improvement = (absolute_improvement / baseline_mape) * 100 if baseline_mape > 0 else 0
    
    return {
        'absolute_improvement': absolute_improvement,
        'relative_improvement': relative_improvement,
        'baseline_mape': baseline_mape,
        'enhanced_mape': enhanced_mape
    }


def create_summary_table(results: List[Dict[str, Any]], 
                        columns: List[str] = None) -> pd.DataFrame:
    """Create a summary table from results."""
    if not results:
        return pd.DataFrame()
    
    if columns is None:
        columns = ['iteration', 'model_name', 'mape', 'mae', 'rmse', 'improvement']
    
    # Filter columns that exist in results
    available_columns = [col for col in columns if any(col in result for result in results)]
    
    data = []
    for result in results:
        row = {col: result.get(col, None) for col in available_columns}
        data.append(row)
    
    return pd.DataFrame(data)


def log_execution_time(func):
    """Decorator to log execution time of functions."""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    
    return wrapper


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1):
        """Update progress."""
        self.current += increment
        percentage = (self.current / self.total) * 100
        elapsed = datetime.now() - self.start_time
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            print(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta}")
        else:
            print(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self):
        """Mark as finished."""
        elapsed = datetime.now() - self.start_time
        print(f"{self.description} completed in {elapsed}")
