# Iterative Agent-Based Feature Selection for Financial Time Series

## Overview

This project implements an advanced iterative feature selection system that uses AI agents to automatically generate and improve feature engineering code for financial time series prediction. The system combines deep learning models with intelligent feature engineering to optimize short interest prediction.

## Key Features

### 🚀 Three Major Improvements

1. **Data Leakage Prevention**
   - Proper train/validation/test split (60%/20%/20%)
   - Validation set used for agent feedback
   - Test set reserved for final evaluation only

2. **Optimal Final Evaluation**
   - Uses train+validation data for final model training
   - Fair comparison between baseline and feature-engineered models
   - Iterate over a small set of stocks and then apply on a broader range of stocks.

3. **DL-Based Feature Importance**
   - Replaces problematic p-value calculations
   - Uses permutation importance and gradient-based methods
   - More meaningful for deep learning models

## Files

- `Agent_iterative_selecting.py` - Original implementation
- `Agent_iterative_selecting_all.py` - Enhanced version with all improvements
- `cache/` - Directory for storing results and reports

## Requirements

```bash
pip install torch numpy pandas scikit-learn anthropic matplotlib scipy
```

## Usage

### Basic Usage

```python
python Agent_iterative_selecting_all.py
```

### Configuration

Edit the following variables in the script:

```python
ANTHROPIC_API_KEY = 'your-api-key-here'  # Replace with your Anthropic API key
stock = 'TSLA'  # Change to your desired stock ticker
```

## How It Works

### 1. Data Preparation
- Loads financial time series data (short interest, volume, OHLC prices)
- Creates proper train/validation/test splits
- Constructs time series windows with lookback periods

### 2. Baseline Model
- Trains LSTM model on raw features
- Evaluates on validation set
- Establishes performance baseline

### 3. Iterative Improvement
- AI agent generates feature engineering code
- Tests new features on validation set
- Uses DL-based feature importance for feedback
- Continues until performance plateaus

### 4. Final Evaluation
- Trains best model on train+validation data
- Evaluates on test set (unseen data)
- Compares with baseline for fair assessment

## Output

### Console Output
- Real-time progress updates
- Performance trend table
- Feature importance analysis
- Final results summary

### Generated Files
- `cache/{stock}_iterative_summary.txt` - Complete summary with all codes
- `cache/{stock}_iterative_results_enhanced.pkl` - Pickle file with results
- `prompt_{iteration}.txt` - AI agent prompts for each iteration

## Feature Engineering

The AI agent generates Python functions that:
- Take raw time series data as input
- Apply financial domain knowledge
- Create meaningful features for LSTM models
- Maintain temporal structure
- Handle edge cases and NaN values

## DL-Based Feature Importance

### Permutation Importance
- Measures actual impact on model performance
- Permutes each feature and measures performance degradation
- Model-agnostic approach

### Gradient-Based Importance
- Uses neural network gradients
- Computationally efficient
- Feature-specific sensitivity analysis

## Configuration Options

### Model Parameters
```python
epochs = 50  # Training epochs
patience = 3  # Early stopping patience
max_iterations = 10  # Maximum iterations
min_improvement_threshold = 0.1  # Minimum improvement to continue
```

### Data Splits
```python
train_split = 0.6  # 60% for training
val_split = 0.8    # 20% for validation (60-80%)
# 20% for test (80-100%)
```
