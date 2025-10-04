# Modular Iterative Agent-Based Feature Selection System

This directory contains a modular, scalable implementation of the iterative agent-based feature selection system for financial time series prediction. The original monolithic script has been broken down into logical, reusable modules.

## 🏗️ Architecture Overview

The system is organized into the following modules:

### Core Modules

- **`config.py`** - Configuration management with preset configurations
- **`data_loader.py`** - Data loading, preprocessing, and feature construction
- **`models.py`** - LSTM model architecture and training logic
- **`feature_engineering.py`** - LLM integration and iterative feature engineering
- **`evaluation.py`** - Performance evaluation and report generation
- **`utils.py`** - Utility functions and helper classes
- **`main.py`** - Main orchestration script

## 🚀 Quick Start

### Basic Usage

```python
from code import Config, IterativeFeatureSelectionPipeline

# Use development configuration
config = Config()
pipeline = IterativeFeatureSelectionPipeline(config)

# Run for a single ticker
best_result, iteration_codes = pipeline.run_iterative_process_for_ticker('AAPL')
```

### Command Line Usage

```bash
# Development mode (faster, fewer iterations)
python code/main.py --config development --single-ticker AAPL

# Production mode (thorough, more iterations)
python code/main.py --config production --tickers AAPL TSLA PFE

# Quick test mode (minimal iterations)
python code/main.py --config quick_test --max-tickers 3
```

## 📋 Configuration

### Preset Configurations

- **Development**: Optimized for development and testing
- **Production**: Optimized for production runs
- **Quick Test**: Minimal configuration for quick testing

### Custom Configuration

```python
from code import Config, DataConfig, ModelConfig, LLMConfig

# Create custom configuration
config = Config(
    data_config=DataConfig(lookback_window=6, gap_days=20),
    model_config=ModelConfig(epochs=100, hidden_size=64),
    llm_config=LLMConfig(max_iterations=15, temperature=0.1)
)
```

## 🔧 Key Features

### 1. Modular Design
- **Separation of Concerns**: Each module has a specific responsibility
- **Easy to Extend**: Add new features without modifying existing code
- **Reusable Components**: Use individual modules in other projects

### 2. Configuration Management
- **Centralized Settings**: All configuration in one place
- **Preset Configurations**: Quick setup for common use cases
- **Environment Variables**: Support for environment-based configuration

### 3. Robust Error Handling
- **Retry Mechanisms**: Automatic retry for API calls and feature engineering
- **Fallback Functions**: Graceful degradation when errors occur
- **Comprehensive Logging**: Detailed logging for debugging

### 4. Performance Optimization
- **Early Stopping**: Stop training when no improvement is detected
- **Batch Processing**: Efficient data processing
- **Memory Management**: Proper cleanup of large objects

## 📊 Usage Examples

### Single Ticker Processing

```python
from code import get_development_config, IterativeFeatureSelectionPipeline

# Load configuration
config = get_development_config()
config.llm.api_key = "your-api-key-here"

# Initialize pipeline
pipeline = IterativeFeatureSelectionPipeline(config)

# Process single ticker
best_result, iteration_codes = pipeline.run_iterative_process_for_ticker('AAPL')

print(f"Best MAPE: {best_result['mape']:.2f}%")
print(f"Improvement: {best_result.get('improvement', 0):+.2f}%")
```

### Multi-Ticker Processing

```python
# Process multiple tickers
iterative_tickers = ['AAPL', 'TSLA', 'PFE']
validation_tickers = ['AAPL', 'TSLA', 'PFE', 'MSFT', 'GOOGL']

results = pipeline.run_multi_ticker_process(iterative_tickers, validation_tickers)

# Access results
ticker_results = results['ticker_results']
universal_function = results['universal_function']
validation_results = results['validation_results']
```

### Custom Feature Engineering

```python
from code import IterativeLLMFeatureSelector, LLMConfig

# Create custom feature selector
llm_config = LLMConfig(api_key="your-api-key", max_iterations=5)
feature_selector = IterativeLLMFeatureSelector(llm_config)

# Use with custom prompt
custom_prompt = "Create features for volatility prediction..."
response = feature_selector.call_claude_for_iterative_improvement(
    iteration=1, 
    previous_results=[], 
    feature_description=custom_prompt
)
```

## 🔍 Module Details

### DataLoader (`data_loader.py`)
- Loads data from parquet files and cached data
- Handles data preprocessing and feature construction
- Validates data integrity
- Supports multiple data sources

### ModelTrainer (`models.py`)
- LSTM model architecture with configurable parameters
- Training with early stopping and validation
- Feature importance calculation using DL-based methods
- Model comparison and evaluation

### Feature Engineering (`feature_engineering.py`)
- LLM integration for automated feature engineering
- Iterative improvement with error feedback
- Universal feature engineering code generation
- Robust error handling and fallback mechanisms

### Evaluation (`evaluation.py`)
- Performance metrics calculation and comparison
- Comprehensive report generation
- Validation testing on multiple tickers
- Statistical analysis of improvements

## 🛠️ Development

### Adding New Features

1. **New Model Architecture**: Extend `models.py`
2. **New Data Sources**: Extend `data_loader.py`
3. **New Evaluation Metrics**: Extend `evaluation.py`
4. **New LLM Providers**: Extend `feature_engineering.py`

### Testing

```python
# Test individual components
from code import DataLoader, ModelTrainer

# Test data loading
data_loader = DataLoader(config.data)
data = data_loader.load_data_for_ticker('AAPL')

# Test model training
model_trainer = ModelTrainer(config.model)
results = model_trainer.train_and_evaluate_model(
    data['X_train_raw'], data['X_test_raw'], 
    data['y_train'], data['y_test'], data['prev_log_test']
)
```

## 📈 Performance Monitoring

The system provides comprehensive performance monitoring:

- **Iteration Tracking**: Monitor improvement over iterations
- **Feature Importance**: Track which features are most important
- **Error Analysis**: Detailed error reporting and feedback
- **Validation Results**: Cross-ticker validation performance

## 🔒 Security and Best Practices

- **API Key Management**: Secure handling of API keys
- **Data Validation**: Comprehensive data integrity checks
- **Error Handling**: Graceful error handling and recovery
- **Logging**: Detailed logging for debugging and monitoring

## 📚 Dependencies

```bash
pip install torch numpy pandas scikit-learn anthropic matplotlib scipy
```

## 🤝 Contributing

The modular design makes it easy to contribute:

1. **Add New Modules**: Create new modules following the existing pattern
2. **Extend Existing Modules**: Add new methods to existing classes
3. **Improve Configuration**: Add new configuration options
4. **Enhance Evaluation**: Add new evaluation metrics and reports

## 📝 License

This project is part of the JP Morgan financial modeling research project.
