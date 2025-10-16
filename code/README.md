# Iterative Agent-Based Feature Selection System

## Overview

This system implements an advanced **iterative agent-based feature selection** approach for financial time series prediction, specifically designed for **Short Interest prediction**. The system combines **Deep Learning (LSTM)** and **Support Vector Machine (SVM)** models with **Large Language Model (LLM)**-driven feature engineering to automatically discover optimal feature combinations.


## 🏗️ System Architecture

### Single Ticker Process:
```
1. Config → 2. DataLoader → 3. Baseline Training → 4. Iteration Loop────┐
                                                                        │
    ┌───────────────────────────────────────────────────────────────────┘
    │
    ▼
5. Prompt Generation → 6. Claude API Call → 7. Code Extraction → 8. Validation
    │
    ▼
9. Feature Application → 10. Model Training → 11. Performance Eval
    │
    └──→ Improvement? ──Yes──→ Continue to next iteration
            │
           No
            │
            ▼
12. Final Test Evaluation → 13. Save Results
```

### Multi-Ticker Process:
```
1. Run Single Ticker Process for each ticker (e.g., 5 tickers)
    │
    ▼
2. Collect best results from all tickers
    │
    ▼
3. Generate Universal Prompt (synthesize best practices)
    │
    ▼
4. Claude generates Universal Feature Engineering Code
    │
    ▼
5. Validate Universal Code on ALL tickers (e.g., 500+ tickers)
    │
    ▼
6. Generate Comprehensive Performance Report
```


## 📊 Data Structure

### Input Data Format
- **Shape**: `(samples, lookback_window=4, features=97)`
- **Features per timestamp** (total: 97 features):
  - **Short Interest** (1 feature): Current short interest value
  - **Volume** (1 feature): Average daily volume over past 15 days
  - **Days to Cover** (1 feature): Short interest / average volume
  - **OHLC Prices** (60 features): Open, High, Low, Close prices for past 15 days (15 days × 4 = 60)
  - **Options Put/Call Ratio** (1 feature): Volume ratio of put options to call options
  - **Synthetic Short Cost** (1 feature): Cost of creating synthetic short via options
  - **Implied Volatility** (1 feature): Average implied volatility of options
  - **Shares Outstanding** (1 feature): Total shares issued
  - **Short Volume** (15 features): Daily short volume for past 15 days
  - **Total Volume** (15 features): Daily total trading volume for past 15 days

### Time Series Structure
```
Timestamp T: [SI, Vol, DTC, OHLC_1...60, PutCall, ShortCost, IV, Shares, ShortVol_1...15, TotalVol_1...15]
Timestamp T-1: [SI, Vol, DTC, OHLC_1...60, PutCall, ShortCost, IV, Shares, ShortVol_1...15, TotalVol_1...15]
...
```

## 🤖 LLM-Driven Feature Engineering

### Iterative Process

The system uses **Claude (Anthropic)** to automatically generate feature engineering code through an iterative process:

#### 1. **Initial Feature Engineering**
```python
# LLM generates initial feature construction code
def construct_features(data):
    # Extract key components (data shape: lookback_window × 97)
    short_interest = data[:, 0]           # Feature 0
    volume = data[:, 1]                   # Feature 1
    days_to_cover = data[:, 2]            # Feature 2
    ohlc_data = data[:, 3:63].reshape(-1, 15, 4)  # Features 3-62 (60 total)
    put_call_ratio = data[:, 63]         # Feature 63
    short_cost = data[:, 64]             # Feature 64
    implied_vol = data[:, 65]            # Feature 65
    shares_out = data[:, 66]             # Feature 66
    short_volume = data[:, 67:82]        # Features 67-81 (15 total)
    total_volume = data[:, 82:97]        # Features 82-96 (15 total)

    # Create engineered features
    features = []
    # ... feature engineering logic
    return np.array(features)  # Shape: (lookback_window, num_engineered_features)
```

#### 2. **Performance-Based Iteration**
- **Baseline Performance**: Train model with original 97 features
- **Feature Importance Analysis**: Use DL-based methods to identify important features
- **LLM Feedback**: Provide performance metrics and feature importance to LLM
- **Code Generation**: LLM generates improved feature engineering code
- **Validation**: Test new features and compare performance

#### 3. **Error Handling & Retry Logic**
- **Validation**: Test generated code with mock data
- **Error Feedback**: If code fails, provide error details to LLM
- **Retry Mechanism**: LLM generates corrected code based on error feedback
- **Success Criteria**: Code must pass validation and improve performance

### Feature Engineering Capabilities

The LLM can generate sophisticated financial features:

- **Momentum Indicators**: Price momentum, volume momentum, short interest momentum
- **Volatility Measures**: Rolling volatility, ATR, price range analysis
- **Technical Indicators**: Moving averages, RSI, MACD-like indicators
- **Volume Analysis**: Volume-price relationships, volume patterns
- **Statistical Features**: Correlation, regression slopes, statistical moments
- **Domain-Specific Features**: Short interest ratios, market microstructure indicators


## 🔄 Iterative Process Flow

### Phase 1: Baseline Establishment
1. **Load Data**: Load financial time series data
2. **Split Data**: Train (60%), Validation (20%), Test (20%)
3. **Baseline Training**: Train LSTM or SVM with original 97 features
4. **Performance Metrics**: Calculate MAPE
5. **Feature Importance**: Analyze which features are most important

### Phase 2: LLM-Driven Iteration
1. **Prompt Generation**: Create detailed prompt with:
   - Performance history
   - Feature importance analysis
   - Error feedback from previous attempts
   - Domain knowledge about financial markets

2. **Code Generation**: LLM generates feature engineering code
3. **Code Validation**: Test generated code with mock data
4. **Feature Application**: Apply new features to training data
5. **Model Retraining**: Train models with new features
6. **Performance Comparison**: Compare with previous iteration

### Phase 3: Convergence & Selection
1. **Convergence Check**: Stop if no improvement for 5 iterations
2. **Best Code Selection**: Choose best performing feature engineering code
3. **Final Validation**: Train on train+val set, test on unseen test set

## 📈 Performance Evaluation
We also evaluate the generalizability of the feature engineering code by make one more call to the LLM to combine all the previous code to a unversal feature engineering code and test on a more general set of tickers.


## 📁 File Structure

```
code/
├── README.md                 # This file
├── config.py                 # Configuration management
├── data_loader.py            # Data loading and preprocessing
├── models.py                 # LSTM and SVM model definitions
├── feature_engineering.py    # LLM-driven feature engineering
├── evaluation.py             # Performance evaluation and reporting
├── main.py                   # Main orchestration pipeline
├── utils.py                  # Utility functions
├── prompt.py                 # LLM prompt templates
├── example_usage.py          # Usage examples
└── __init__.py              # Package initialization
```

## 🎯 Key Innovations

### **LLM-Driven Feature Engineering**
- **Automatic Code Generation**: LLM generates production-ready Python code
- **Domain Knowledge Integration**: Incorporates financial market expertise
- **Iterative Improvement**: Learns from previous attempts and errors
- **Error Handling**: Robust retry mechanism with error feedback


## 🚀 Usage
Please check out example_usage.py. Our final production version is the function example_svm_multi_ticker(). You can customize the iterative_tickers and validation_tickers there. Remember to include your anthropic API key in .env file in the root folder.
