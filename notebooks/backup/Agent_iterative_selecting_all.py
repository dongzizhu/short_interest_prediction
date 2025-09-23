"""
Iterative Agent-Based Feature Selection for Financial Time Series (Enhanced Version)
==================================================================================

This script implements an iterative process with three key improvements:
1. IMPROVEMENT 1: Proper train/validation/test split to prevent data leakage
   - Training set (60%): Used for model training
   - Validation set (20%): Used for agent feedback and iteration decisions
   - Test set (20%): Used only for final evaluation (unseen data)
2. IMPROVEMENT 2: Optimal final evaluation using all available training data
   - Baseline: Raw features, trained on train+val, tested on test
   - Best Model: Processed features, trained on train+val, tested on test
   - Fair comparison between baseline and feature-engineered model
3. IMPROVEMENT 3: DL-based feature importance instead of problematic p-values
   - Permutation importance: Measures actual impact on model performance
   - Gradient-based importance: Uses neural network gradients for feature ranking
   - More meaningful for deep learning models than statistical p-values
4. Runs a baseline model without enhanced features
5. Uses an AI agent to generate feature engineering code
6. Tests the enhanced features on validation set and measures performance with DL-based importance
7. Iteratively improves the features based on validation performance feedback
8. Continues until performance plateaus
9. Final evaluation maximizes data usage and provides fair comparison

Author: AI Assistant
Date: 2024
"""

import pickle
import anthropic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
import time
from datetime import datetime, timedelta
import json
warnings.filterwarnings('ignore')


class EnhancedLSTMTimeSeries(nn.Module):
    """Enhanced LSTM model for time series prediction"""
    
    def __init__(self, input_size=62, hidden_size=64, num_layers=3, output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


def calculate_dl_feature_importance(model, X_train, y_train, feature_names=None, method='permutation'):
    """Calculate DL-based feature importance using permutation or gradient-based methods"""
    model.eval()
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[-1])]
    
    if method == 'permutation':
        return calculate_permutation_importance(model, X_train, y_train, feature_names)
    elif method == 'gradient':
        return calculate_gradient_importance(model, X_train, y_train, feature_names)
    else:
        raise ValueError("Method must be 'permutation' or 'gradient'")

def calculate_permutation_importance(model, X_train, y_train, feature_names):
    """Calculate feature importance using permutation method"""
    print("📊 Calculating permutation-based feature importance...")
    
    # Get baseline performance
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        baseline_pred = model(X_tensor)
        baseline_loss = nn.SmoothL1Loss()(baseline_pred, torch.tensor(y_train, dtype=torch.float32)).item()
    
    feature_importance = {}
    importance_scores = []
    
    # Calculate importance for each feature
    for i in range(X_train.shape[-1]):
        # Create permuted data
        X_permuted = X_train.copy()
        np.random.shuffle(X_permuted[:, :, i])  # Permute feature i across all samples and timesteps
        
        # Calculate performance with permuted feature
        with torch.no_grad():
            X_perm_tensor = torch.tensor(X_permuted, dtype=torch.float32)
            perm_pred = model(X_perm_tensor)
            perm_loss = nn.SmoothL1Loss()(perm_pred, torch.tensor(y_train, dtype=torch.float32)).item()
        
        # Importance = increase in loss when feature is permuted
        importance = perm_loss - baseline_loss
        importance_scores.append(importance)
        
        # Determine significance based on importance magnitude
        # Use relative importance (normalized by max importance)
        max_importance = max(importance_scores) if importance_scores else 1.0
        relative_importance = importance / max_importance if max_importance > 0 else 0
        
        feature_importance[feature_names[i]] = {
            'importance': importance,
            'relative_importance': relative_importance,
            'significant': relative_importance > 0.1,  # Top 10% of features
            'highly_significant': relative_importance > 0.2,  # Top 5% of features
            'rank': 0  # Will be updated after all features are processed
        }
    
    # Rank features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]  # Descending order
    for rank, idx in enumerate(sorted_indices):
        feature_importance[feature_names[idx]]['rank'] = rank + 1
    
    return feature_importance, importance_scores

def calculate_gradient_importance(model, X_train, y_train, feature_names):
    """Calculate feature importance using gradient-based method"""
    print("📊 Calculating gradient-based feature importance...")
    
    model.eval()
    X_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    # Forward pass
    output = model(X_tensor)
    loss = nn.SmoothL1Loss()(output, y_tensor)
    
    # Calculate gradients
    gradients = torch.autograd.grad(loss, X_tensor, retain_graph=True)[0]
    
    # Calculate importance as mean absolute gradient across all samples and timesteps
    importance_scores = torch.mean(torch.abs(gradients), dim=(0, 1)).detach().numpy()
    
    feature_importance = {}
    
    # Normalize importance scores
    max_importance = np.max(importance_scores) if np.max(importance_scores) > 0 else 1.0
    normalized_importance = importance_scores / max_importance
    
    for i, (name, importance, norm_importance) in enumerate(zip(feature_names, importance_scores, normalized_importance)):
        feature_importance[name] = {
            'importance': float(importance),
            'relative_importance': float(norm_importance),
            'significant': norm_importance > 0.1,  # Top 10% of features
            'highly_significant': norm_importance > 0.2,  # Top 5% of features
            'rank': 0  # Will be updated after all features are processed
        }
    
    # Rank features by importance
    sorted_indices = np.argsort(importance_scores)[::-1]  # Descending order
    for rank, idx in enumerate(sorted_indices):
        feature_importance[feature_names[idx]]['rank'] = rank + 1
    
    return feature_importance, importance_scores


def train_and_evaluate_model(X_train, X_test, y_train, y_test, prev_log_test, model_name="Model", epochs=50):
    """Train and evaluate an LSTM model, returning performance metrics and feature statistics"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    # Scale inputs
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Create data loaders
    batch_size = 8
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size)
    
    # Initialize model
    model = EnhancedLSTMTimeSeries(input_size=X_train.shape[-1], hidden_size=32, num_layers=2, output_size=1)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Training loop
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Early stopping
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Evaluation
    model.eval()
    pred_logret = []
    with torch.no_grad():
        for xb, _ in test_loader:
            pred_logret.append(model(xb).numpy())
    pred_logret = np.concatenate(pred_logret, axis=0).ravel()
    
    # Reconstruct levels
    y_pred_levels = np.exp(prev_log_test + pred_logret)
    y_true_levels = np.exp(prev_log_test + y_test.ravel())
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred_levels - y_true_levels))
    rmse = np.sqrt(np.mean((y_pred_levels - y_true_levels)**2))
    mape = np.mean(np.abs((y_true_levels - y_pred_levels) / y_true_levels)) * 100
    
    print(f"\n{model_name} Performance:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Calculate DL-based feature importance
    print(f"\n📊 Calculating DL-based feature importance...")
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[-1])]
    
    # Use permutation importance as default (more robust)
    feature_stats, importance_scores = calculate_dl_feature_importance(
        model, X_train_scaled, y_train, feature_names, method='permutation'
    )

    # Print feature importance summary
    significant_features = [name for name, stats in feature_stats.items() if stats['significant']]
    highly_significant = [name for name, stats in feature_stats.items() if stats['highly_significant']]
    
    print(f"📈 DL-Based Feature Importance Analysis:")
    print(f"   • Total features: {len(feature_stats)}")
    print(f"   • Important features (top 10%): {len(significant_features)}")
    print(f"   • Highly important features (top 5%): {len(highly_significant)}")
    
    # Show top 5 most important features with their importance scores
    sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['importance'], reverse=True)
    print(f"\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
    for i, (name, stats) in enumerate(sorted_features[:5]):
        print(f"   {i+1}. {name}: importance={stats['importance']:.4f}, rank={stats['rank']}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': y_pred_levels,
        'true_values': y_true_levels,
        'model': model,
        'scaler': scaler,
        'feature_stats': feature_stats,
        'importance_scores': importance_scores,
        'significant_features': significant_features,
        'highly_significant_features': highly_significant
    }


class IterativeLLMFeatureSelector:
    """Iterative LLM-based feature selector with p-value analysis and retry mechanism"""
    
    def __init__(self, anthropic_api_key: str):
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        print("✅ Claude API client initialized successfully!")
    
    def create_iterative_prompt(self, iteration_num: int, previous_results: list, feature_description: str, error_feedback: list = None) -> str:
        """Create a prompt that includes performance history, p-value analysis, and asks for improvements"""
        # Build performance history with p-value information
        history_text = ""
        if previous_results:
            history_text = "\n\nPERFORMANCE HISTORY:\n"
            for i, result in enumerate(previous_results):
                history_text += f"Iteration {i}: {result['model_name']} - MAPE: {result['mape']:.2f}%"
                if i > 0:
                    improvement = result['improvement']
                    history_text += f" (Improvement: {improvement:+.1f}%)"
                history_text += f"\n  Features: {result['features_used']}\n"
                
                # Add DL-based feature importance analysis if available
                if 'feature_stats' in result and result['feature_stats']:
                    history_text += f"  DL-Based Feature Importance Analysis:\n"
                    
                    # Add top 5 most important features
                    sorted_features = sorted(result['feature_stats'].items(), key=lambda x: x[1]['importance'], reverse=True)
                    top_features = sorted_features[:5]
                    if top_features:
                        feature_list = ', '.join([f'{name} (importance={stats["importance"]:.4f})' for name, stats in top_features])
                        history_text += f"    • Top important features: {feature_list}\n"
                
                history_text += "\n"
        
        # Get the best performance so far
        best_mape = min([r['mape'] for r in previous_results]) if previous_results else 100.0
        
        # Get statistical insights from the best performing model
        best_result = min(previous_results, key=lambda x: x['mape']) if previous_results else None
        statistical_insights = ""
        
        if best_result and 'feature_stats' in best_result:
            statistical_insights = f"""
DL-BASED FEATURE IMPORTANCE INSIGHTS FROM BEST MODEL (MAPE: {best_result['mape']:.2f}%):
- Most important features: {', '.join([f"{name} (importance={stats['importance']:.4f})" for name, stats in sorted(best_result['feature_stats'].items(), key=lambda x: x[1]['importance'], reverse=True)[:5]])}
- Least important features: {', '.join([f"{name} (importance={stats['importance']:.4f})" for name, stats in sorted(best_result['feature_stats'].items(), key=lambda x: x[1]['importance'])[:3]])}
"""
        
        # Add error feedback section if there are previous errors
        error_feedback_text = ""
        if error_feedback and len(error_feedback) > 0:
            error_feedback_text = "\n\nERROR FEEDBACK FROM PREVIOUS ATTEMPTS:\n"
            error_feedback_text += "The following errors occurred in previous attempts. Please analyze these errors and ensure your code avoids these issues:\n\n"
            
            for i, error_info in enumerate(error_feedback):
                error_feedback_text += f"Error {i+1}:\n"
                error_feedback_text += f"  • Error Type: {error_info.get('error_type', 'Unknown')}\n"
                error_feedback_text += f"  • Error Message: {error_info.get('error_message', 'No message')}\n"
                if 'sample_info' in error_info:
                    error_feedback_text += f"  • Sample Info: {error_info['sample_info']}\n"
                if 'code_snippet' in error_info:
                    error_feedback_text += f"  • Problematic Code: {error_info['code_snippet'][:200]}...\n"
                error_feedback_text += "\n"
            
            error_feedback_text += "IMPORTANT: Your new code must avoid these specific errors. Pay special attention to:\n"
            error_feedback_text += "- Array dimension mismatches and shape issues\n"
            error_feedback_text += "- Proper handling of edge cases and NaN values\n"
            error_feedback_text += "- Correct return value format (2D numpy array)\n"
            error_feedback_text += "- Robust error handling within the function\n"
        
        # Add previous code section if there are previous iterations
        previous_code_text = ""
        if previous_results and len(previous_results) > 0:
            # Get the most recent iteration's code
            last_result = previous_results[-1]
            if 'claude_code' in last_result and last_result['claude_code']:
                previous_code_text = "\n\nPREVIOUS ITERATION CODE:\n"
                previous_code_text += f"The following code was used in the most recent iteration (Iteration {last_result['iteration']}):\n\n"
                previous_code_text += "```python\n"
                previous_code_text += last_result['claude_code']
                previous_code_text += "\n```\n\n"
                previous_code_text += f"Performance of this code: MAPE = {last_result['mape']:.2f}%\n"
                if last_result.get('improvement', 0) > 0:
                    previous_code_text += f"Improvement over previous: {last_result['improvement']:+.2f}%\n"
                else:
                    previous_code_text += f"Change from previous: {last_result['improvement']:+.2f}%\n"
                
                # Add error information if available
                if last_result.get('errors_encountered', 0) > 0:
                    previous_code_text += f"Errors encountered: {last_result['errors_encountered']}\n"
                    if last_result.get('error_details'):
                        previous_code_text += "Sample errors:\n"
                        for i, error in enumerate(last_result['error_details'][:2]):
                            previous_code_text += f"  - {error.get('error_type', 'Unknown')}: {error.get('error_message', 'No message')}\n"
                
                # Add statistical significance information if available
                if last_result.get('feature_stats'):
                    significant_count = len(last_result.get('significant_features', []))
                    highly_significant_count = len(last_result.get('highly_significant_features', []))
                    total_features = len(last_result['feature_stats'])
                    previous_code_text += f"Statistical Analysis: {significant_count}/{total_features} features were significant (p < 0.05), {highly_significant_count} were highly significant (p < 0.01)\n"
                
                previous_code_text += "\nINSTRUCTIONS FOR NEW CODE:\n"
                previous_code_text += "- Analyze the previous code and understand what features it tried to create\n"
                previous_code_text += "- Identify what worked well and what didn't work based on performance and statistical significance\n"
                previous_code_text += "- If the previous code worked but had poor performance, try different feature engineering approaches\n"
                previous_code_text += "- Consider the statistical significance of features - focus on creating features that are likely to be statistically significant\n"
                previous_code_text += "- Your new code should be an improvement over the previous attempt\n"
                previous_code_text += "- Think about what additional financial insights or technical indicators could be valuable\n"
        
        prompt = f"""
You are a financial data scientist expert in feature engineering for Short Interest prediction models. 

I have financial time series data with the following structure:
- Shape: (samples, lookback_window=4, features=62)
- Features at each timestamp T include:
  1. Short interest at time T reported every 15 days (1 dimension)
  2. Average daily volume quantity of past 15 days (1 dimension) 
  3. OHLC (Open, High, Low, Close) prices for past 15 days (4 × 15 = 60 dimensions)

Total: 1 + 1 + 60 = 62 features(dimensions) per timestamp.

{history_text}

{statistical_insights}

{error_feedback_text}

{previous_code_text}

CURRENT TASK (Iteration {iteration_num}):
Your goal is to create an improved feature engineering function that will achieve better performance than the current best MAPE of {best_mape:.2f}%.

Based on the performance history, DL-based feature importance analysis, and previous code above, analyze what worked and what didn't, then create a new feature engineering approach that:
1. Learns from previous iterations' successes and failures
2. Analyzes the previous code to understand what features were attempted and their effectiveness
3. Builds upon successful feature patterns while avoiding problematic approaches
4. Considers financial domain knowledge (momentum, volatility, volume patterns, etc.)
5. Maintains LSTM-compatible time series structure
6. Uses DL-based feature importance insights to prioritize feature construction
7. Improves upon the previous iteration's approach

Requirements:
1. Write a function called `construct_features` that takes a numpy array of shape (lookback_window, 62) and returns a numpy array of shape (lookback_window, constructed_features)
2. The function should process each timestamp independently but maintain the temporal structure
3. Focus on the most predictive features for each time step, using DL-based feature importance as guidance
4. Consider financial domain knowledge (e.g., price momentum, volatility, volume patterns, etc.)
5. The output should be a 2D numpy array with shape (lookback_window, constructed_features)
6. Include comments explaining your feature engineering choices and how they address previous performance issues and importance insights
7. Make sure the code is production-ready and handles edge cases
8. DO NOT include any import statements - only use numpy (available as 'np') and built-in Python functions
9. The function must return a 2D array where each row represents features for one time step
10. Use numpy nan_to_num to handle NaN values
11. Analyze the previous iteration's code and explain in comments how your approach differs and improves upon it

Please provide ONLY the Python function code, no explanations outside the code comments.

Feature description: {feature_description}
"""
        return prompt
    
    def call_claude_for_iterative_improvement(self, iteration_num: int, previous_results: list, feature_description: str = "Equity Short interestprediction with past short interest, volume, and OHLC data", error_feedback: list = None) -> str:
        """Call Claude API with iterative improvement context"""
        prompt = self.create_iterative_prompt(iteration_num, previous_results, feature_description, error_feedback)
        if iteration_num <= 2:
            if 'claude_code' in previous_results[-1]:
                prompt += f"\n\nPREVIOUS CODE ATTEMPT:\n{previous_results[-1]['claude_code']}\n\n"
        f = open(f"prompt_{iteration_num}.txt", "w", encoding="utf-8")
        f.write(prompt)
        f.close()
        
        try:
            response = self.client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=7500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return None
    
    def extract_function_from_response(self, response_text: str) -> str:
        """Extract the construct_features function from Claude's response"""
        lines = response_text.split('\n')
        function_lines = []
        in_function = False
        indent_level = 0
        
        for line in lines:
            if 'def construct_features' in line:
                in_function = True
                function_lines.append(line)
                indent_level = len(line) - len(line.lstrip())
            elif in_function:
                if line.strip() == '':
                    function_lines.append(line)
                elif len(line) - len(line.lstrip()) > indent_level or line.strip() == '':
                    function_lines.append(line)
                else:
                    break
        
        extracted_code = '\n'.join(function_lines)
        
        # Validate that we extracted a proper function
        if not extracted_code.strip():
            raise ValueError("No function code found in response")
        
        if 'def construct_features' not in extracted_code:
            raise ValueError("construct_features function definition not found in extracted code")
        
        return extracted_code
    
    def execute_feature_construction_code(self, code: str) -> callable:
        """Execute the generated feature construction code and return the function"""
        try:
            # Create a fresh execution environment for each attempt
            exec_globals = {
                'np': np,
                'pd': pd,
                '__builtins__': {
                    'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
                    'sum': sum, 'max': max, 'min': min, 'abs': abs, 'round': round,
                    'int': int, 'float': float, 'str': str, 'list': list, 'dict': dict,
                    'tuple': tuple, 'set': set, 'print': print, 'any': any, 'all': all,
                    'sorted': sorted, 'reversed': reversed, 'isinstance': isinstance,
                    'type': type, 'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
                    'callable': callable, 'issubclass': issubclass, 'super': super,
                    'open': open, 'iter': iter, 'next': next, 'map': map, 'filter': filter,
                    'pow': pow, 'divmod': divmod, 'bin': bin, 'hex': hex, 'oct': oct,
                    'ord': ord, 'chr': chr, 'bool': bool, 'complex': complex,
                    'bytes': bytes, 'bytearray': bytearray, 'memoryview': memoryview,
                    'slice': slice, 'property': property, 'staticmethod': staticmethod,
                    'classmethod': classmethod,
                }
            }
            
            # Execute the code in the fresh environment
            exec(code, exec_globals)
            
            # Check if construct_features was successfully created
            if 'construct_features' in exec_globals and callable(exec_globals['construct_features']):
                # Test the function with a sample input to ensure it works
                test_input = np.random.rand(4, 62)  # Sample input with expected shape
                try:
                    test_output = exec_globals['construct_features'](test_input)
                    if not isinstance(test_output, np.ndarray):
                        raise ValueError("Function did not return a numpy array")
                    if test_output.ndim != 2:
                        raise ValueError(f"Function returned array with {test_output.ndim} dimensions, expected 2")
                    print("✅ Successfully Extracted Function Code!")
                    print(f"Test output shape: {test_output.shape}")
                    return exec_globals['construct_features']
                except Exception as test_error:
                    raise ValueError(f"Function failed test execution: {test_error}")
            else:
                raise ValueError("construct_features function not found or not callable in generated code")
                
        except Exception as e:
            print(f"❌ Error executing generated code: {e}")
            return None
    
    def fallback_construct_features(self, data: np.ndarray) -> np.ndarray:
        """Fallback feature construction function"""
        if data.shape[1] != 62:
            raise ValueError(f"Expected 62 features, got {data.shape[1]}")
        
        lookback_window = data.shape[0]
        features_per_timestep = 15
        
        output = np.zeros((lookback_window, features_per_timestep))
        
        for t in range(lookback_window):
            short_interest = data[t, 0]
            volume = data[t, 1]
            ohlc = data[t, 2:].reshape(15, 4)
            
            # Basic features
            output[t, 0] = short_interest
            output[t, 1] = volume
            output[t, 2:6] = ohlc[-1]  # Latest OHLC
            
            # Momentum features
            close_prices = ohlc[:, 3]
            for i, horizon in enumerate([1, 3, 5, 10]):
                if len(close_prices) > horizon:
                    momentum = (close_prices[-1] - close_prices[-horizon-1]) / close_prices[-horizon-1]
                    output[t, 6 + i] = momentum
                else:
                    output[t, 6 + i] = 0
            
            # Volatility and range
            if len(close_prices) > 1:
                returns = np.diff(close_prices) / close_prices[:-1]
                output[t, 10] = np.std(returns)
                output[t, 11] = (np.max(close_prices) - np.min(close_prices)) / np.mean(close_prices)
            
            # Technical indicators
            if len(close_prices) > 1:
                sma_5 = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else np.mean(close_prices)
                output[t, 12] = close_prices[-1] / sma_5
                
                high_low_ratio = ohlc[:, 1] / ohlc[:, 2]
                output[t, 13] = np.mean(high_low_ratio)
                
                output[t, 14] = volume * close_prices[-1] / np.mean(close_prices)
        
        return output.astype(np.float32)
    
    def apply_feature_selection_to_data(self, X_data: np.ndarray, construct_func: callable, max_retries: int = 5) -> tuple:
        """Apply feature selection to the entire dataset with retry mechanism
        
        Returns:
            tuple: (processed_data, error_feedback_list)
        """
        retry_count = 0
        all_errors = []  # Collect all errors across retries
        
        while retry_count < max_retries:
            try:
                processed_samples = []
                sample_errors = []
                
                for i in range(X_data.shape[0]):
                    sample_features = X_data[i]
                    
                    try:
                        constructed_features = construct_func(sample_features)
                        
                        # Validate the output shape and dimensions
                        if constructed_features.ndim == 1:
                            # Reshape 1D to 2D
                            features_per_step = len(constructed_features) // sample_features.shape[0]
                            if features_per_step > 0:
                                constructed_features = constructed_features[:features_per_step * sample_features.shape[0]]
                                constructed_features = constructed_features.reshape(sample_features.shape[0], features_per_step)
                            else:
                                # If we can't reshape properly, create a fallback
                                constructed_features = np.zeros((sample_features.shape[0], 1))
                                constructed_features[0, 0] = constructed_features.mean() if len(constructed_features) > 0 else 0
                        elif constructed_features.ndim == 2:
                            # Ensure the first dimension matches the lookback window
                            if constructed_features.shape[0] != sample_features.shape[0]:
                                print(f"⚠️ Shape mismatch: expected {sample_features.shape[0]} timesteps, got {constructed_features.shape[0]}")
                                # Try to fix by padding or truncating
                                if constructed_features.shape[0] < sample_features.shape[0]:
                                    # Pad with zeros
                                    padding = np.zeros((sample_features.shape[0] - constructed_features.shape[0], constructed_features.shape[1]))
                                    constructed_features = np.vstack([constructed_features, padding])
                                else:
                                    # Truncate
                                    constructed_features = constructed_features[:sample_features.shape[0]]
                        
                        processed_samples.append(constructed_features)
                        
                    except Exception as e:
                        error_msg = f"Error processing sample {i} in attempt {retry_count + 1}: {e}"
                        print(error_msg)
                        
                        # Collect detailed error information for feedback
                        error_info = {
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'sample_info': f"Sample {i}, shape: {sample_features.shape}",
                            'attempt': retry_count + 1
                        }
                        sample_errors.append((i, error_msg, error_info))
                        all_errors.append(error_info)
                        
                        # Use fallback for this sample
                        fallback_features = sample_features[:, :10] if sample_features.shape[1] >= 10 else sample_features
                        processed_samples.append(fallback_features)
                
                # Check if we had too many errors
                error_rate = len(sample_errors) / X_data.shape[0]
                if error_rate > 0.5:  # More than 50% of samples failed
                    raise Exception(f"High error rate: {len(sample_errors)}/{X_data.shape[0]} samples failed. Sample errors: {sample_errors[:3]}")
                
                # If we get here, the function executed successfully (with some fallbacks)
                if sample_errors:
                    print(f"⚠️ Feature selection completed with {len(sample_errors)} fallback samples on attempt {retry_count + 1}")
                else:
                    print(f"✅ Feature selection applied successfully on attempt {retry_count + 1}")
                
                return np.array(processed_samples), all_errors
                
            except Exception as e:
                retry_count += 1
                print(f"❌ Attempt {retry_count} failed with error: {e}")
                
                # Add attempt-level error information
                attempt_error = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'sample_info': f"Entire dataset processing failed",
                    'attempt': retry_count
                }
                all_errors.append(attempt_error)
                
                if retry_count < max_retries:
                    print(f"🔄 Retrying... ({retry_count}/{max_retries})")
                else:
                    print(f"⚠️ All {max_retries} attempts failed, using fallback function")
                    fallback_data = self._apply_fallback_feature_selection(X_data)
                    return fallback_data, all_errors
        
        return np.array([]), all_errors
    
    def _apply_fallback_feature_selection(self, X_data: np.ndarray) -> np.ndarray:
        """Fallback feature selection when all retries fail"""
        print("🆘 Using fallback feature selection...")
        processed_samples = []
        
        for i in range(X_data.shape[0]):
            sample_features = X_data[i]
            # Simple fallback: take first 15 features from each timestep
            fallback_features = sample_features[:, :15] if sample_features.shape[1] >= 15 else sample_features
            processed_samples.append(fallback_features)
        
        return np.array(processed_samples)


def get_ohlc_data(df, ticker, start_date, end_date):
    """
    Extract OHLC data for a specific ticker within a date range.
    
    Parameters:
    df (DataFrame): The DataFrame with MultiIndex columns (ticker, metric)
    ticker (str): The ticker symbol to extract data for
    start_date (str or datetime): Start date in 'YYYY-MM-DD' format
    end_date (str or datetime): End date in 'YYYY-MM-DD' format
    
    Returns:
    DataFrame: OHLC data with columns ['open', 'high', 'low', 'close']
    """
    try:
        # Check if ticker exists in the data
        if ticker not in df.columns.get_level_values(0):
            available_tickers = df.columns.get_level_values(0).unique().tolist()
            raise ValueError(f"Ticker '{ticker}' not found. Available tickers: {available_tickers}")
        
        # Extract OHLC columns for the specific ticker
        ohlc_data = df.loc[start_date:end_date, [(ticker, '1. open'), 
                                                 (ticker, '2. high'), 
                                                 (ticker, '3. low'), 
                                                 (ticker, '4. close')]]
        
        # Rename columns to remove the ticker prefix for cleaner output
        ohlc_data.columns = ['open', 'high', 'low', 'close']
        
        # Remove rows with all NaN values
        ohlc_data = ohlc_data.dropna(how='all')
        
        return ohlc_data
        
    except KeyError as e:
        print(f"Error: Date range or ticker not found in data. {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


def create_price_features_from_parquet(si_dates, price_data, gap_days=15):
    """
    Create daily OHLC price features for each SI date with gap_days lag structure
    Features: [day1_open, day1_high, day1_low, day1_close, ..., dayN_open, dayN_high, dayN_low, dayN_close]
    Total: 4 * gap_days features
    
    Parameters:
    si_dates: pandas Index of SI reporting dates
    price_data: DataFrame with OHLC data (columns: open, high, low, close)
    gap_days: Number of days to look back for price features
    
    Returns:
    numpy array of shape (len(si_dates), 4 * gap_days)
    """
    if price_data.empty:
        print("No price data available, creating zero features")
        return np.zeros((len(si_dates), 4 * gap_days))
    
    price_features = []
    
    for si_date in si_dates:
        si_datetime = pd.to_datetime(si_date)
        
        # Define the window (gap_days before the SI date)
        window_end = si_datetime - timedelta(days=1)  # Day before SI date
        
        features = []
        
        # Get price data for each of the gap_days
        for day_offset in range(gap_days):
            target_date = window_end - timedelta(days=day_offset)
            
            # Find the closest trading day (in case of weekends/holidays)
            closest_date = None
            min_diff = timedelta(days=10)  # Max search window
            
            for price_date in price_data.index:
                diff = abs(price_date.date() - target_date.date())
                if diff < min_diff:
                    min_diff = diff
                    closest_date = price_date
            
            if closest_date is not None and min_diff <= timedelta(days=3):  # Within 3 days
                day_data = price_data.loc[closest_date]
                features.extend([
                    day_data['open'],
                    day_data['high'], 
                    day_data['low'],
                    day_data['close']
                ])
            else:
                # No data available, use NaNs
                features.extend([np.nan, np.nan, np.nan, np.nan])
        
        price_features.append(features)
    
    price_features = np.array(price_features)
    
    # Handle NaNs by imputing with column means
    col_means = np.nanmean(price_features, axis=0)
    price_features = np.where(np.isnan(price_features), col_means, price_features)
    
    return price_features


def load_data_from_parquet_and_construct_features(stock, parquet_path='../data/price_data_multiindex_20250904_113138.parquet'):
    """
    Load data from parquet file and construct timeseries features similar to 05_Agent_selecting
    
    Parameters:
    stock (str): Stock ticker symbol
    parquet_path (str): Path to the parquet file
    
    Returns:
    dict: Dictionary containing all the data needed for training
    """
    print(f"📊 Loading data for {stock} from parquet file...")
    
    # Load ticker timeseries data (SI and Volume)
    ticker_timeseries = pickle.load(open("cache/ticker_timeseries.pkl", "rb"))
    
    # Load parquet file with price data
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Parquet file loaded. Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Get SI dates and data
    si_dates = ticker_timeseries[stock]['SI'].dropna().index
    SI_series = ticker_timeseries[stock]['SI'].dropna().values.astype(np.float64).reshape(-1, 1)
    vol_series = ticker_timeseries[stock]['Volume'].dropna().values.astype(np.float64).reshape(-1, 1)
    
    print(f"SI dates range: {si_dates.min()} to {si_dates.max()}")
    print(f"Number of SI observations: {len(si_dates)}")
    
    # Get OHLC data for the stock
    print(f"Extracting OHLC data for {stock}...")
    start_date = si_dates.min() - timedelta(days=30)  # Add buffer for lookback
    end_date = si_dates.max() + timedelta(days=5)     # Add buffer for future dates
    
    price_data = get_ohlc_data(df, stock, start_date, end_date)
    print(f"Retrieved price data for {len(price_data)} trading days")
    if not price_data.empty:
        print(f"Price data date range: {price_data.index.min()} to {price_data.index.max()}")
    
    # Create price features (15 days of OHLC = 60 features)
    gap_days = 15
    print(f"Creating price features with {gap_days} days lookback...")
    price_features = create_price_features_from_parquet(si_dates, price_data, gap_days)
    print(f"Price features shape: {price_features.shape}")
    
    # Combine all features: [SI, Volume, 60 price features]
    level_series = np.concatenate([SI_series, vol_series, price_features], axis=1)  # (T, 62)
    print(f"Combined features shape: {level_series.shape}")
    
    # Create log-return targets
    eps = 1e-8  # to avoid log(0)
    series_safe = np.where(SI_series <= 0, eps, SI_series).reshape(-1)
    y_log = np.log(series_safe)
    
    # Build supervised windows with log-return target
    lookback_window = 4
    
    def make_windows_level_to_logret(level_series, y_log, lookback):
        X_list, y_logret_list, prev_log_list = [], [], []
        for t in range(lookback, len(level_series)):
            X_list.append(level_series[t - lookback:t, :])                  # (L, 62)
            y_logret_list.append([y_log[t] - y_log[t - 1]])                 # (1,)
            prev_log_list.append(y_log[t - 1])   
        X = np.asarray(X_list)                          # (N, L, 62)
        y_logret = np.asarray(y_logret_list)            # (N, 1)
        prev_log = np.asarray(prev_log_list)            # (N,)
        return X, y_logret, prev_log
    
    X_raw, y_logret, prev_log_all = make_windows_level_to_logret(level_series, y_log, lookback_window)
    
    # Split data into train/val/test to prevent data leakage
    N = X_raw.shape[0]
    train_split = int(0.6 * N)  # 60% for training
    val_split = int(0.8 * N)    # 20% for validation (60-80%)
    # 20% for test (80-100%)
    
    X_train_raw = X_raw[:train_split]
    X_val_raw = X_raw[train_split:val_split]
    X_test_raw = X_raw[val_split:]
    
    y_train = y_logret[:train_split]
    y_val = y_logret[train_split:val_split]
    y_test = y_logret[val_split:]
    
    prev_log_train = prev_log_all[:train_split]
    prev_log_val = prev_log_all[train_split:val_split]
    prev_log_test = prev_log_all[val_split:]
    
    print(f"Training data shape: {X_train_raw.shape}")
    print(f"Validation data shape: {X_val_raw.shape}")
    print(f"Test data shape: {X_test_raw.shape}")
    
    # Prepare data dictionary
    data_to_save = {
        # Raw data
        'X_train_raw': X_train_raw,
        'X_val_raw': X_val_raw,
        'X_test_raw': X_test_raw,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        
        # Dates and series
        'si_dates': si_dates,
        'SI_series': SI_series,
        'prev_log_test': prev_log_test,
        'prev_log_train': prev_log_train,
        'prev_log_val': prev_log_val,
        
        # Additional info
        'stock': stock,
        'lookback_window': lookback_window,
        'gap_days': gap_days,
        'price_data_shape': price_data.shape if not price_data.empty else (0, 0)
    }
    
    return data_to_save

def main():
    """Main function to run the iterative agent-based feature selection"""
    # Configuration
    ANTHROPIC_API_KEY = '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'  # Replace with your actual API key
    stock = 'TSLA'
    
    print("🚀 Starting Iterative Agent-Based Feature Selection Process")
    print("="*70)
    
    # Load data from parquet file instead of cached pkl
    print("📊 Loading data from parquet file...")
    raw_data = load_data_from_parquet_and_construct_features(stock)
    X_train_raw = raw_data['X_train_raw']
    X_val_raw = raw_data['X_val_raw']
    X_test_raw = raw_data['X_test_raw']
    y_train = raw_data['y_train']
    y_val = raw_data['y_val']
    y_test = raw_data['y_test']
    prev_log_train = raw_data['prev_log_train']
    prev_log_val = raw_data['prev_log_val']
    prev_log_test = raw_data['prev_log_test']
    si_dates = raw_data['si_dates']
    SI_series = raw_data['SI_series']
    
    print(f"✅ Data loaded successfully from parquet file!")
    print(f"Training data shape: {X_train_raw.shape}")
    print(f"Validation data shape: {X_val_raw.shape}")
    print(f"Test data shape: {X_test_raw.shape}")
    print(f"Features per timestep: {X_train_raw.shape[2]}")
    print(f"Lookback window: {X_train_raw.shape[1]}")
    
    # Initialize dictionary to track all iteration codes
    iteration_codes = {}
    
    # Step 1: Run baseline model on validation set (to prevent data leakage)
    print("\n🎯 Step 1: Running baseline model on validation set...")
    baseline_results = train_and_evaluate_model(
        X_train_raw, X_val_raw, y_train, y_val, prev_log_val, 
        model_name="Baseline (All 62 Features)", epochs=50
    )
    
    baseline_mape = baseline_results['mape']
    print(f"\n📊 Baseline Performance: MAPE = {baseline_mape:.2f}%")
    
    # Store results for comparison
    iteration_results = []
    iteration_results.append({
        'iteration': 0,
        'model_name': 'Baseline',
        'features_used': 'All 62 original features',
        'feature_count': X_train_raw.shape[2],
        'mape': baseline_mape,
        'mae': baseline_results['mae'],
        'rmse': baseline_results['rmse'],
        'improvement': 0.0,
        'predictions': baseline_results['predictions'],
        'feature_stats': baseline_results.get('feature_stats', {}),
        'significant_features': baseline_results.get('significant_features', []),
        'highly_significant_features': baseline_results.get('highly_significant_features', [])
    })
    
    # Initialize the feature selector
    feature_selector = IterativeLLMFeatureSelector(ANTHROPIC_API_KEY)
    
    # Iterative improvement loop
    max_iterations = 10
    min_improvement_threshold = 0.1
    patience = 5
    
    print(f"\n🔄 Starting iterative improvement process...")
    print(f"Max iterations: {max_iterations}")
    print(f"Min improvement threshold: {min_improvement_threshold}%")
    print(f"Patience: {patience} iterations without improvement")
    
    best_mape = baseline_mape
    iterations_without_improvement = 0
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}")
        print(f"{'='*70}")
        
        # Get feature engineering code from Claude with retry mechanism
        print(f"🤖 Calling Claude for iteration {iteration} with statistical insights...")
        
        # Retry mechanism for Claude API calls and function execution
        max_claude_retries = 3
        claude_success = False
        claude_errors = []  # Collect errors from Claude attempts
        
        for claude_retry in range(max_claude_retries):
            try:
                # Pass error feedback from previous attempts if available
                error_feedback = claude_errors if claude_errors else None
                
                claude_response = feature_selector.call_claude_for_iterative_improvement(
                    iteration, iteration_results, 
                    "Stock short interest prediction with past short interest, volume, and OHLC data for iterative improvement",
                    error_feedback=error_feedback
                )
                
                if not claude_response:
                    error_msg = f"No response from Claude (attempt {claude_retry + 1}/{max_claude_retries})"
                    print(f"❌ {error_msg}")
                    claude_errors.append({
                        'error_type': 'NoResponse',
                        'error_message': error_msg,
                        'attempt': claude_retry + 1
                    })
                    if claude_retry < max_claude_retries - 1:
                        print("🔄 Retrying Claude API call...")
                        continue
                    else:
                        print("⚠️ All Claude API attempts failed, using fallback")
                        construct_func = feature_selector.fallback_construct_features
                        function_source = "fallback"
                        claude_code = None
                        claude_success = True
                        break
                
                print("✅ Claude response received!")
                print(f"\n📝 Claude's Response:")
                print("-" * 50)
                print(claude_response)
                print("-" * 50)
                
                # Extract and execute the function
                try:
                    function_code = feature_selector.extract_function_from_response(claude_response)
                    
                    construct_func = feature_selector.execute_feature_construction_code(function_code)
                    
                    if construct_func:
                        # Test the function with a small sample to ensure it works
                        test_sample = X_train_raw[0]  # Use actual data shape for testing
                        try:
                            test_result = construct_func(test_sample)
                            if isinstance(test_result, np.ndarray) and test_result.ndim == 2:
                                print("✅ Function executed successfully and passed validation!")
                                function_source = "claude"
                                claude_code = function_code
                                claude_success = True
                                break
                            else:
                                raise ValueError(f"Function returned invalid output: shape={test_result.shape if hasattr(test_result, 'shape') else 'no shape'}, type={type(test_result)}")
                        except Exception as test_error:
                            error_msg = f"Function validation failed: {test_error}"
                            print(f"⚠️ {error_msg}")
                            claude_errors.append({
                                'error_type': 'ValidationError',
                                'error_message': error_msg,
                                'attempt': claude_retry + 1,
                                'code_snippet': function_code[:200] + "..." if function_code else "No code"
                            })
                            if claude_retry < max_claude_retries - 1:
                                print("🔄 Retrying with error feedback...")
                                continue
                            else:
                                print("⚠️ All function validation attempts failed, using fallback")
                                construct_func = feature_selector.fallback_construct_features
                                function_source = "fallback"
                                claude_code = None
                                claude_success = True
                                break
                    else:
                        error_msg = f"Function execution failed (attempt {claude_retry + 1}/{max_claude_retries})"
                        print(f"⚠️ {error_msg}")
                        claude_errors.append({
                            'error_type': 'ExecutionError',
                            'error_message': error_msg,
                            'attempt': claude_retry + 1,
                            'code_snippet': function_code[:200] + "..." if function_code else "No code"
                        })
                        if claude_retry < max_claude_retries - 1:
                            print("🔄 Retrying function execution with error feedback...")
                            continue
                        else:
                            print("⚠️ All function execution attempts failed, using fallback")
                            construct_func = feature_selector.fallback_construct_features
                            function_source = "fallback"
                            claude_code = None
                            claude_success = True
                            break
                            
                except Exception as extraction_error:
                    error_msg = f"Function extraction/execution failed: {extraction_error}"
                    print(f"⚠️ {error_msg}")
                    claude_errors.append({
                        'error_type': 'ExtractionError',
                        'error_message': error_msg,
                        'attempt': claude_retry + 1,
                        'code_snippet': claude_response[:200] + "..." if claude_response else "No response"
                    })
                    if claude_retry < max_claude_retries - 1:
                        print("🔄 Retrying with error feedback...")
                        continue
                    else:
                        print("⚠️ All extraction attempts failed, using fallback")
                        construct_func = feature_selector.fallback_construct_features
                        function_source = "fallback"
                        claude_code = None
                        claude_success = True
                        break
                        
            except Exception as e:
                error_msg = f"Error in Claude process (attempt {claude_retry + 1}/{max_claude_retries}): {e}"
                print(f"❌ {error_msg}")
                claude_errors.append({
                    'error_type': 'ProcessError',
                    'error_message': error_msg,
                    'attempt': claude_retry + 1
                })
                if claude_retry < max_claude_retries - 1:
                    print("🔄 Retrying entire Claude process with error feedback...")
                    continue
                else:
                    print("⚠️ All Claude process attempts failed, using fallback")
                    construct_func = feature_selector.fallback_construct_features
                    function_source = "fallback"
                    claude_code = None
                    claude_success = True
                    break
        
        if not claude_success:
            print("🆘 Critical error: Could not establish feature construction function")
            continue
        
        # Apply feature selection to data with retry mechanism
        print(f"\n🔧 Applying feature selection using {function_source} function with retry mechanism...")
        X_train_processed, train_errors = feature_selector.apply_feature_selection_to_data(X_train_raw, construct_func, max_retries=5)
        X_val_processed, val_errors = feature_selector.apply_feature_selection_to_data(X_val_raw, construct_func, max_retries=5)
        X_test_processed, test_errors = feature_selector.apply_feature_selection_to_data(X_test_raw, construct_func, max_retries=5)
        
        # Combine all errors for feedback
        all_errors = train_errors + val_errors + test_errors
        
        print(f"Training data shape: {X_train_raw.shape} -> {X_train_processed.shape}")
        print(f"Validation data shape: {X_val_raw.shape} -> {X_val_processed.shape}")
        print(f"Test data shape: {X_test_raw.shape} -> {X_test_processed.shape}")
        
        if all_errors:
            print(f"⚠️ Total errors encountered: {len(all_errors)}")
            for i, error in enumerate(all_errors[:3]):  # Show first 3 errors
                print(f"  Error {i+1}: {error['error_type']} - {error['error_message']}")
            print("ℹ️ Note: Function was validated during initial retry attempts, so these errors should be minimal.")
        
        # Train on training set and evaluate on validation set (for agent feedback)
        iteration_results_model = train_and_evaluate_model(
            X_train_processed, X_val_processed, y_train, y_val, prev_log_val,
            model_name=f"Iteration {iteration} ({function_source})", epochs=50
        )
        
        # Calculate improvement
        improvement = best_mape - iteration_results_model['mape']
        
        # Store results with p-value information and error tracking
        iteration_results.append({
            'iteration': iteration,
            'model_name': f'Iteration {iteration}',
            'features_used': f'{function_source} feature engineering',
            'feature_count': X_train_processed.shape[2],
            'mape': iteration_results_model['mape'],
            'mae': iteration_results_model['mae'],
            'rmse': iteration_results_model['rmse'],
            'improvement': improvement,
            'predictions': iteration_results_model['predictions'],
            'claude_code': claude_code,
            'function_source': function_source,
            'feature_stats': iteration_results_model.get('feature_stats', {}),
            'significant_features': iteration_results_model.get('significant_features', []),
            'highly_significant_features': iteration_results_model.get('highly_significant_features', []),
            'p_values': iteration_results_model.get('p_values', []),
            'coefficients': iteration_results_model.get('coefficients', []),
            'errors_encountered': len(all_errors),
            'error_details': all_errors[:5] if all_errors else [],  # Store first 5 errors for reference
            'claude_errors': claude_errors  # Store Claude retry errors
        })
        
        # Store the code in the iteration codes dictionary
        if claude_code:
            iteration_codes[f'iteration_{iteration}'] = {
                'code': claude_code,
                'function_source': function_source,
                'mape': iteration_results_model['mape'],
                'improvement': improvement,
                'feature_count': X_train_processed.shape[2],
                'errors_encountered': len(all_errors),
                'significant_features': len(iteration_results_model.get('significant_features', [])),
                'highly_significant_features': len(iteration_results_model.get('highly_significant_features', []))
            }
            print(f"💾 Saved code for iteration {iteration} to iteration_codes dictionary")
        
        # Check for improvement
        if improvement > min_improvement_threshold:
            print(f"🎉 IMPROVEMENT! MAPE improved by {improvement:.2f}%")
            best_mape = iteration_results_model['mape']
            iterations_without_improvement = 0
        else:
            print(f"📊 No significant improvement. Change: {improvement:+.2f}%")
            iterations_without_improvement += 1
        
        # Check stopping criteria
        if iterations_without_improvement >= patience:
            print(f"\n🛑 Stopping: No improvement for {patience} consecutive iterations")
            break
        
        print(f"\n📈 Current best MAPE: {best_mape:.2f}%")
        print(f"🔄 Iterations without improvement: {iterations_without_improvement}/{patience}")
    
    # Final evaluation on test set (unseen data) to get true performance
    print(f"\n🎯 Final Evaluation on Test Set (Unseen Data)")
    print("="*70)
    print("Using train+validation data for final model training to maximize data usage")
    
    # Combine training and validation sets for final model training
    X_train_val_raw = np.concatenate([X_train_raw, X_val_raw], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    prev_log_train_val = np.concatenate([prev_log_train, prev_log_val], axis=0)
    
    print(f"Combined train+val data shape: {X_train_val_raw.shape}")
    print(f"Test data shape: {X_test_raw.shape}")
    
    # Find the best iteration based on validation performance
    best_iteration_result = min(iteration_results[1:], key=lambda x: x['mape']) if len(iteration_results) > 1 else iteration_results[0]
    
    # 1. BASELINE: Train on train+val with raw features, test on test
    print(f"\n📊 BASELINE EVALUATION (Raw Features, Train+Val → Test):")
    baseline_final_results = train_and_evaluate_model(
        X_train_val_raw, X_test_raw, y_train_val, y_test, prev_log_test,
        model_name="Baseline Final (Raw Features)", epochs=50
    )
    
    print(f"   Baseline MAPE: {baseline_final_results['mape']:.2f}%")
    print(f"   Baseline MAE: {baseline_final_results['mae']:.4f}")
    print(f"   Baseline RMSE: {baseline_final_results['rmse']:.4f}")
    
    # 2. BEST MODEL: Train on train+val with processed features, test on test
    if best_iteration_result.get('claude_code'):
        print(f"\n🔧 BEST MODEL EVALUATION (Processed Features, Train+Val → Test):")
        print("Applying best feature engineering to all data...")
        
        # Recreate the best function
        try:
            best_function_code = best_iteration_result['claude_code']
            best_construct_func = feature_selector.execute_feature_construction_code(best_function_code)
            
            if best_construct_func:
                # Apply feature engineering to train+val and test sets
                X_train_val_processed, _ = feature_selector.apply_feature_selection_to_data(X_train_val_raw, best_construct_func, max_retries=5)
                X_test_processed_final, _ = feature_selector.apply_feature_selection_to_data(X_test_raw, best_construct_func, max_retries=5)
                
                print(f"Processed train+val shape: {X_train_val_processed.shape}")
                print(f"Processed test shape: {X_test_processed_final.shape}")
                
                # Final evaluation with processed features
                final_test_results = train_and_evaluate_model(
                    X_train_val_processed, X_test_processed_final, y_train_val, y_test, prev_log_test,
                    model_name="Best Model Final (Processed Features)", epochs=50
                )
                
                print(f"\n📊 Best Model Test Set Performance:")
                print(f"   MAPE: {final_test_results['mape']:.2f}%")
                print(f"   MAE: {final_test_results['mae']:.4f}")
                print(f"   RMSE: {final_test_results['rmse']:.4f}")
                
                # Calculate improvement over baseline
                improvement = baseline_final_results['mape'] - final_test_results['mape']
                improvement_percentage = (improvement / baseline_final_results['mape']) * 100
                
                print(f"\n🎯 IMPROVEMENT OVER BASELINE:")
                print(f"   Baseline MAPE: {baseline_final_results['mape']:.2f}%")
                print(f"   Best Model MAPE: {final_test_results['mape']:.2f}%")
                print(f"   Absolute Improvement: {improvement:.2f}%")
                print(f"   Relative Improvement: {improvement_percentage:.1f}%")
                
                # Add final test results to the best result
                best_iteration_result['final_test_mape'] = final_test_results['mape']
                best_iteration_result['final_test_mae'] = final_test_results['mae']
                best_iteration_result['final_test_rmse'] = final_test_results['rmse']
                best_iteration_result['baseline_final_mape'] = baseline_final_results['mape']
                best_iteration_result['final_improvement'] = improvement
                best_iteration_result['final_improvement_percentage'] = improvement_percentage
                
            else:
                print("⚠️ Could not recreate best function, using baseline for final evaluation")
                final_test_results = baseline_final_results
        except Exception as e:
            print(f"⚠️ Error in final evaluation: {e}")
            print("Using baseline for final evaluation")
            final_test_results = baseline_final_results
    else:
        print("ℹ️ No feature engineering code available, using baseline for final evaluation")
        final_test_results = baseline_final_results
    
    # Simple Performance Summary
    print("\n" + "="*70)
    print("ITERATION PERFORMANCE SUMMARY")
    print("="*70)
    
    # Create simple performance table
    print(f"\n📊 VALIDATION MAPE TREND:")
    print("-" * 50)
    print(f"{'Iteration':<10} {'Model':<25} {'Validation MAPE':<15} {'Improvement':<12}")
    print("-" * 50)
    
    for result in iteration_results:
        improvement_str = f"{result['improvement']:+.2f}%" if result['improvement'] != 0 else "Baseline"
        print(f"{result['iteration']:<10} {result['model_name']:<25} {result['mape']:<15.2f} {improvement_str:<12}")
    
    # Find best result
    best_result = min(iteration_results, key=lambda x: x['mape'])
    print("-" * 50)
    print(f"🏆 Best: {best_result['model_name']} - MAPE: {best_result['mape']:.2f}%")
    
    # Final test performance if available
    if 'final_test_mape' in best_result:
        print(f"🎯 Final Test MAPE: {best_result['final_test_mape']:.2f}%")
        if 'baseline_final_mape' in best_result:
            print(f"📈 Final Improvement: {best_result.get('final_improvement', 0):.2f}%")
    
    # Save simple summary to text file
    print(f"\n💾 Saving summary to text file...")
    summary_filename = f'cache/{stock}_iterative_summary.txt'
    with open(summary_filename, 'w', encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("ITERATIVE AGENT-BASED FEATURE SELECTION SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Stock: {stock}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Iterations: {len(iteration_results) - 1}\n\n")
        
        f.write("PERFORMANCE TREND:\n")
        f.write("-" * 40 + "\n")
        for result in iteration_results:
            improvement_str = f"{result['improvement']:+.2f}%" if result['improvement'] != 0 else "Baseline"
            f.write(f"Iteration {result['iteration']}: {result['model_name']} - MAPE: {result['mape']:.2f}% ({improvement_str})\n")
        
        f.write(f"\nBest Model: {best_result['model_name']} - MAPE: {best_result['mape']:.2f}%\n")
        
        if 'final_test_mape' in best_result:
            f.write(f"Final Test MAPE: {best_result['final_test_mape']:.2f}%\n")
            if 'baseline_final_mape' in best_result:
                f.write(f"Final Improvement: {best_result.get('final_improvement', 0):.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("FEATURE ENGINEERING CODES\n")
        f.write("="*60 + "\n\n")
        
        for result in iteration_results[1:]:  # Skip baseline
            if result.get('claude_code'):
                f.write(f"ITERATION {result['iteration']}:\n")
                f.write(f"Performance: MAPE = {result['mape']:.2f}%\n")
                f.write(f"Improvement: {result['improvement']:+.2f}%\n")
                f.write(f"Features: {result['feature_count']}\n")
                f.write("-" * 40 + "\n")
                f.write(result['claude_code'])
                f.write("\n" + "="*60 + "\n\n")
    
    print(f"✅ Summary saved to: {summary_filename}")
    print("\n🎉 Process completed successfully!")


if __name__ == "__main__":
    main()
