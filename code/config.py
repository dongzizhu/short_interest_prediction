"""
Configuration management for the iterative agent-based feature selection system.

This module centralizes all configuration parameters, making it easy to modify
settings without changing the core logic.
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    parquet_path: str = '../data/price_data_multiindex_20250904_113138.parquet'
    ticker_timeseries_path: str = 'cache/ticker_timeseries.pkl'
    
    # Data splitting
    train_split: float = 0.6  # 60% for training
    val_split: float = 0.8    # 20% for validation (60-80%)
    # 20% for test (80-100%)
    
    # Time series parameters
    lookback_window: int = 4
    gap_days: int = 15  # Number of days to look back for price features
    
    # Feature dimensions
    total_features: int = 62  # SI(1) + Volume(1) + OHLC(4*15=60)
    
    # Data preprocessing
    eps: float = 1e-8  # Small value to avoid log(0)


@dataclass
class ModelConfig:
    """Configuration for model training and architecture."""
    
    # Model type and architecture
    model_type: str = 'lstm'  # 'lstm' or 'svm'
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.2
    output_size: int = 1
    
    # Training parameters
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Early stopping
    patience: int = 10
    
    # Feature importance
    importance_method: str = 'permutation'  # 'permutation' or 'gradient'
    
    # SVM-specific parameters
    svm_kernel: str = 'rbf'  # 'rbf', 'linear', 'poly', 'sigmoid'
    svm_C: float = 1.0
    svm_gamma: str = 'scale'  # 'scale', 'auto', or float
    svm_epsilon: float = 0.1
    svm_max_iter: int = 1000


@dataclass
class LLMConfig:
    """Configuration for LLM integration."""
    
    # API settings
    api_key: str = ''  # Will be set from environment or user input
    model: str = 'claude-3-7-sonnet-latest'
    max_tokens: int = 10000
    temperature: float = 0.2
    
    # Retry settings
    max_claude_retries: int = 3
    max_feature_retries: int = 5
    
    # Iteration settings
    max_iterations: int = 10
    min_improvement_threshold: float = 0.1
    patience_iterations: int = 5


@dataclass
class EvaluationConfig:
    """Configuration for evaluation and reporting."""
    
    # Output paths
    cache_dir: str = 'cache'
    output_dir: str = 'output'
    
    # Reporting
    save_individual_results: bool = True
    save_comprehensive_results: bool = True
    generate_performance_report: bool = True
    
    # Validation
    test_universal_on_all_tickers: bool = True
    max_validation_tickers: Optional[int] = None


@dataclass
class SystemConfig:
    """System-wide configuration."""
    
    # Logging
    verbose: bool = True
    log_level: str = 'INFO'
    
    # Parallel processing
    use_multiprocessing: bool = False
    max_workers: int = 4
    
    # Random seed for reproducibility
    random_seed: int = 42


class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self, 
                 data_config: Optional[DataConfig] = None,
                 model_config: Optional[ModelConfig] = None,
                 llm_config: Optional[LLMConfig] = None,
                 evaluation_config: Optional[EvaluationConfig] = None,
                 system_config: Optional[SystemConfig] = None):
        
        self.data = data_config or DataConfig()
        self.model = model_config or ModelConfig()
        self.llm = llm_config or LLMConfig()
        self.evaluation = evaluation_config or EvaluationConfig()
        self.system = system_config or SystemConfig()
        
        # Set up paths
        self._setup_paths()
    
    def _setup_paths(self):
        """Set up directory paths and create them if they don't exist."""
        Path(self.evaluation.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.evaluation.output_dir).mkdir(parents=True, exist_ok=True)
    
    def get_api_key(self) -> str:
        """Get API key from environment or config."""
        if self.llm.api_key:
            return self.llm.api_key
        
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if api_key:
            return api_key
        
        raise ValueError("API key not found. Set ANTHROPIC_API_KEY environment variable or set llm.api_key in config.")
    
    def get_available_tickers(self, max_tickers: Optional[int] = None) -> List[str]:
        """Get list of available tickers from the parquet file."""        
        try:
            df = pd.read_parquet(self.data.parquet_path)
            all_tickers = list(set([x[0] for x in df.columns]))
            
            if max_tickers:
                return all_tickers[:max_tickers]
            return all_tickers
            
        except Exception as e:
            print(f"Error loading tickers from parquet: {e}")
            # Fallback to default tickers
            return ['TSLA', 'PFE', 'AAPL']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'llm': self.llm.__dict__,
            'evaluation': self.evaluation.__dict__,
            'system': self.system.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        llm_config = LLMConfig(**config_dict.get('llm', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        
        return cls(data_config, model_config, llm_config, evaluation_config, system_config)


# Default configuration instance
default_config = Config()


# Predefined configurations for different use cases
def get_development_config() -> Config:
    """Configuration optimized for development and testing."""
    return Config(
        data_config=DataConfig(),
        model_config=ModelConfig(epochs=30, patience=5),  # Faster training
        llm_config=LLMConfig(max_iterations=5, min_improvement_threshold=0.05),
        evaluation_config=EvaluationConfig(max_validation_tickers=10),
        system_config=SystemConfig(verbose=True)
    )


def get_production_config() -> Config:
    """Configuration optimized for production runs."""
    return Config(
        data_config=DataConfig(),
        model_config=ModelConfig(epochs=150, patience=20),  # More thorough training
        llm_config=LLMConfig(max_iterations=10, min_improvement_threshold=0.1),
        evaluation_config=EvaluationConfig(max_validation_tickers=None),
        system_config=SystemConfig(verbose=False, use_multiprocessing=True)
    )


def get_quick_test_config() -> Config:
    """Configuration for quick testing and debugging."""
    return Config(
        data_config=DataConfig(),
        model_config=ModelConfig(epochs=10, patience=3),
        llm_config=LLMConfig(max_iterations=2, min_improvement_threshold=0.01),
        evaluation_config=EvaluationConfig(max_validation_tickers=3),
        system_config=SystemConfig(verbose=True)
    )
