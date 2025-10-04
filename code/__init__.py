"""
Iterative Agent-Based Feature Selection for Financial Time Series

A modular system for automated feature engineering using AI agents to improve
financial time series prediction models.

Modules:
- config: Configuration management
- data_loader: Data loading and preprocessing
- models: Model architecture and training
- feature_engineering: LLM-based feature engineering
- evaluation: Performance evaluation and reporting
- utils: Utility functions
- main: Main orchestration script
"""

from .config import Config, get_development_config, get_production_config, get_quick_test_config
from .data_loader import DataLoader
from .models import ModelTrainer, EnhancedLSTMTimeSeries
from .feature_engineering import IterativeLLMFeatureSelector, UniversalFeatureEngineering
from .evaluation import PerformanceEvaluator, ReportGenerator, ValidationTester
from .utils import setup_logging, ProgressTracker

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "Modular iterative agent-based feature selection for financial time series"

__all__ = [
    'Config',
    'get_development_config',
    'get_production_config', 
    'get_quick_test_config',
    'DataLoader',
    'ModelTrainer',
    'EnhancedLSTMTimeSeries',
    'IterativeLLMFeatureSelector',
    'UniversalFeatureEngineering',
    'PerformanceEvaluator',
    'ReportGenerator',
    'ValidationTester',
    'setup_logging',
    'ProgressTracker'
]
