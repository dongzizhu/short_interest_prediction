"""
Iterative Agent-Based Feature Selection for Financial Time Series
===============================================================

This script implements an iterative process that:
1. Runs a baseline model without enhanced features
2. Uses an AI agent to generate feature engineering code
3. Tests the enhanced features and measures performance with p-value analysis
4. Iteratively improves the features based on performance feedback
5. Continues until performance plateaus

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
from datetime import datetime
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


def calculate_feature_pvalues(X_train, y_train, feature_names=None):
    """Calculate p-values for each feature using linear regression"""
    # Flatten the 3D data to 2D for statistical analysis
    X_train = X_train[:, -1, :]
    X_flat = X_train.reshape(X_train.shape[0], -1)
    y_flat = y_train.ravel()
    
    # Fit linear regression
    lr = LinearRegression()
    lr.fit(X_flat, y_flat)
    
    # Calculate residuals
    y_pred = lr.predict(X_flat)
    residuals = y_flat - y_pred
    mse = np.mean(residuals**2)
    
    # Calculate standard errors and t-statistics
    X_with_intercept = np.column_stack([np.ones(X_flat.shape[0]), X_flat])
    try:
        cov_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        standard_errors = np.sqrt(np.diag(cov_matrix))[1:]  # Exclude intercept
        t_statistics = lr.coef_ / standard_errors
        
        # Calculate p-values (two-tailed test)
        degrees_of_freedom = X_flat.shape[0] - X_flat.shape[1] - 1
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistics), degrees_of_freedom))
        
    except np.linalg.LinAlgError:
        print("⚠️ Singular matrix detected, using simplified p-value calculation")
        p_values = np.ones(X_flat.shape[1]) * 0.5  # Default to non-significant
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_flat.shape[1])]
    
    # Create results dictionary
    feature_stats = {}
    for i, (name, pval, coef) in enumerate(zip(feature_names, p_values, lr.coef_)):
        feature_stats[name] = {
            'p_value': pval,
            'coefficient': coef,
            'significant': pval < 0.05,
            'highly_significant': pval < 0.01
        }
    
    return feature_stats, lr.coef_, p_values


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
    
    # Calculate feature p-values
    print(f"\n📊 Calculating feature p-values...")
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[-1])]
    feature_stats, coefficients, p_values = calculate_feature_pvalues(X_train_scaled, y_train, feature_names)

    # Print feature significance summary
    significant_features = [name for name, stats in feature_stats.items() if stats['significant']]
    highly_significant = [name for name, stats in feature_stats.items() if stats['highly_significant']]
    
    print(f"📈 Feature Significance Analysis:")
    print(f"   • Total features: {len(feature_stats)}")
    print(f"   • Significant features (p < 0.05): {len(significant_features)}")
    print(f"   • Highly significant features (p < 0.01): {len(highly_significant)}")
    
    if significant_features:
        print(f"   • Significant features: {significant_features[:5]}{'...' if len(significant_features) > 5 else ''}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'predictions': y_pred_levels,
        'true_values': y_true_levels,
        'model': model,
        'scaler': scaler,
        'feature_stats': feature_stats,
        'coefficients': coefficients,
        'p_values': p_values,
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
                
                # Add p-value analysis if available
                if 'feature_stats' in result and result['feature_stats']:
                    significant_count = len(result.get('significant_features', []))
                    highly_significant_count = len(result.get('highly_significant_features', []))
                    total_features = len(result['feature_stats'])
                    
                    history_text += f"  Statistical Analysis:\n"
                    history_text += f"    • Total features: {total_features}\n"
                    history_text += f"    • Significant features (p < 0.05): {significant_count}\n"
                    history_text += f"    • Highly significant features (p < 0.01): {highly_significant_count}\n"
                    
                    # Add top significant features
                    if result.get('significant_features'):
                        top_significant = result['significant_features'][:3]
                        history_text += f"    • Top significant features: {', '.join(top_significant)}\n"
                    
                    # Add feature with lowest p-value
                    if result['feature_stats']:
                        min_pval_feature = min(result['feature_stats'].items(), key=lambda x: x[1]['p_value'])
                        history_text += f"    • Most significant feature: {min_pval_feature[0]} (p={min_pval_feature[1]['p_value']:.4f})\n"
                
                history_text += "\n"
        
        # Get the best performance so far
        best_mape = min([r['mape'] for r in previous_results]) if previous_results else 100.0
        
        # Get statistical insights from the best performing model
        best_result = min(previous_results, key=lambda x: x['mape']) if previous_results else None
        statistical_insights = ""
        
        if best_result and 'feature_stats' in best_result:
            statistical_insights = f"""
STATISTICAL INSIGHTS FROM BEST MODEL (MAPE: {best_result['mape']:.2f}%):
- Most predictive features (lowest p-values): {', '.join([f"{name} (p={stats['p_value']:.4f})" for name, stats in sorted(best_result['feature_stats'].items(), key=lambda x: x[1]['p_value'])[:5]])}
- Least predictive features (highest p-values): {', '.join([f"{name} (p={stats['p_value']:.4f})" for name, stats in sorted(best_result['feature_stats'].items(), key=lambda x: x[1]['p_value'], reverse=True)[:3]])}
- Feature significance ratio: {len(best_result.get('significant_features', []))}/{len(best_result['feature_stats'])} features are statistically significant
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

Based on the performance history, statistical analysis, and previous code above, analyze what worked and what didn't, then create a new feature engineering approach that:
1. Learns from previous iterations' successes and failures
2. Analyzes the previous code to understand what features were attempted and their effectiveness
3. Builds upon successful feature patterns while avoiding problematic approaches
4. Considers financial domain knowledge (momentum, volatility, volume patterns, etc.)
5. Maintains LSTM-compatible time series structure
6. Uses p-value insights to prioritize feature construction
7. Improves upon the previous iteration's approach

Requirements:
1. Write a function called `construct_features` that takes a numpy array of shape (lookback_window, 62) and returns a numpy array of shape (lookback_window, constructed_features)
2. The function should process each timestamp independently but maintain the temporal structure
3. Focus on the most predictive features for each time step, using statistical significance as guidance
4. Consider financial domain knowledge (e.g., price momentum, volatility, volume patterns, etc.)
5. The output should be a 2D numpy array with shape (lookback_window, constructed_features)
6. Include comments explaining your feature engineering choices and how they address previous performance issues and statistical insights
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
        f = open(f"prompt_{iteration_num}.txt", "w", encoding="utf-8")
        f.write(prompt)
        f.close()
        
        try:
            response = self.client.messages.create(
                model="claude-3-7-sonnet-latest",
                max_tokens=5000,
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


def main():
    """Main function to run the iterative agent-based feature selection"""
    # Configuration
    ANTHROPIC_API_KEY = ''  # Replace with your actual API key
    stock = 'TSLA'
    
    print("🚀 Starting Iterative Agent-Based Feature Selection Process")
    print("="*70)
    
    # Load data
    print("📊 Loading data...")
    raw_data = pickle.load(open(f'cache/{stock}_raw_data_with_ohlc.pkl', 'rb'))
    X_train_raw = raw_data['X_train_raw']
    X_test_raw = raw_data['X_test_raw']
    y_train = raw_data['y_train']
    y_test = raw_data['y_test']
    prev_log_train = raw_data['prev_log_train']
    prev_log_test = raw_data['prev_log_test']
    si_dates = raw_data['si_dates']
    SI_series = raw_data['SI_series']
    
    print(f"✅ Data loaded successfully!")
    print(f"Training data shape: {X_train_raw.shape}")
    print(f"Test data shape: {X_test_raw.shape}")
    
    # Initialize dictionary to track all iteration codes
    iteration_codes = {}
    
    # Step 1: Run baseline model
    print("\n🎯 Step 1: Running baseline model...")
    baseline_results = train_and_evaluate_model(
        X_train_raw, X_test_raw, y_train, y_test, prev_log_test, 
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
    patience = 3
    
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
        X_test_processed, test_errors = feature_selector.apply_feature_selection_to_data(X_test_raw, construct_func, max_retries=5)
        
        # Combine all errors for feedback
        all_errors = train_errors + test_errors
        
        print(f"Training data shape: {X_train_raw.shape} -> {X_train_processed.shape}")
        print(f"Test data shape: {X_test_raw.shape} -> {X_test_processed.shape}")
        
        if all_errors:
            print(f"⚠️ Total errors encountered: {len(all_errors)}")
            for i, error in enumerate(all_errors[:3]):  # Show first 3 errors
                print(f"  Error {i+1}: {error['error_type']} - {error['error_message']}")
            print("ℹ️ Note: Function was validated during initial retry attempts, so these errors should be minimal.")
        
        # Train and evaluate the model with p-value analysis
        iteration_results_model = train_and_evaluate_model(
            X_train_processed, X_test_processed, y_train, y_test, prev_log_test,
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
    
    # Performance Analysis and Summary
    print("\n" + "="*70)
    print("ENHANCED PERFORMANCE ANALYSIS AND SUMMARY")
    print("="*70)
    
    # Create enhanced performance summary DataFrame
    performance_data = []
    for result in iteration_results:
        significant_count = len(result.get('significant_features', []))
        highly_significant_count = len(result.get('highly_significant_features', []))
        
        performance_data.append({
            'Iteration': result['iteration'],
            'Model': result['model_name'],
            'Features': result['feature_count'],
            'MAPE (%)': result['mape'],
            'MAE': result['mae'],
            'RMSE': result['rmse'],
            'Improvement (%)': result['improvement'],
            'Source': result.get('function_source', 'baseline'),
            'Significant Features': significant_count,
            'Highly Significant': highly_significant_count
        })
    
    performance_df = pd.DataFrame(performance_data)
    print("\n📊 ENHANCED PERFORMANCE SUMMARY TABLE:")
    print("="*70)
    print(performance_df.round(4).to_string(index=False))
    
    # Find best performing model
    best_result = min(iteration_results, key=lambda x: x['mape'])
    print(f"\n🏆 BEST PERFORMING MODEL:")
    print(f"   Model: {best_result['model_name']}")
    print(f"   MAPE: {best_result['mape']:.2f}%")
    print(f"   MAE: {best_result['mae']:.4f}")
    print(f"   RMSE: {best_result['rmse']:.4f}")
    print(f"   Features: {best_result['feature_count']}")
    print(f"   Source: {best_result.get('function_source', 'baseline')}")
    
    # Statistical significance analysis
    if 'feature_stats' in best_result and best_result['feature_stats']:
        print(f"\n📈 STATISTICAL SIGNIFICANCE ANALYSIS:")
        print(f"   • Total features: {len(best_result['feature_stats'])}")
        print(f"   • Significant features (p < 0.05): {len(best_result.get('significant_features', []))}")
        print(f"   • Highly significant features (p < 0.01): {len(best_result.get('highly_significant_features', []))}")
        
        # Show top 5 most significant features
        if best_result['feature_stats']:
            sorted_features = sorted(best_result['feature_stats'].items(), key=lambda x: x[1]['p_value'])
            print(f"\n🔍 TOP 5 MOST SIGNIFICANT FEATURES:")
            for i, (name, stats) in enumerate(sorted_features[:5]):
                significance = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else ""
                print(f"   {i+1}. {name}: p={stats['p_value']:.4f} {significance}")
    
    # Calculate total improvement
    total_improvement = baseline_mape - best_result['mape']
    improvement_percentage = (total_improvement / baseline_mape) * 100
    
    print(f"\n📈 OVERALL IMPROVEMENT:")
    print(f"   Baseline MAPE: {baseline_mape:.2f}%")
    print(f"   Best MAPE: {best_result['mape']:.2f}%")
    print(f"   Total Improvement: {total_improvement:.2f}% ({improvement_percentage:.1f}% relative)")
    
    # Feature reduction analysis
    baseline_features = iteration_results[0]['feature_count']
    best_features = best_result['feature_count']
    feature_reduction = ((baseline_features - best_features) / baseline_features) * 100
    
    print(f"\n🔧 FEATURE REDUCTION:")
    print(f"   Baseline features: {baseline_features}")
    print(f"   Best model features: {best_features}")
    print(f"   Feature reduction: {feature_reduction:.1f}%")
    
    # Iteration efficiency
    successful_iterations = len([r for r in iteration_results[1:] if r['improvement'] > 0])
    total_iterations = len(iteration_results) - 1
    efficiency = (successful_iterations / total_iterations) * 100 if total_iterations > 0 else 0
    
    print(f"\n⚡ ITERATION EFFICIENCY:")
    print(f"   Total iterations: {total_iterations}")
    print(f"   Successful improvements: {successful_iterations}")
    print(f"   Success rate: {efficiency:.1f}%")
    
    # Save results and generate final report
    print("\n💾 Saving enhanced results and generating comprehensive report...")
    
    # Save iteration results and codes to pickle
    results_filename = f'cache/{stock}_iterative_results_enhanced.pkl'
    save_data = {
        'iteration_results': iteration_results,
        'iteration_codes': iteration_codes,
        'best_result': best_result,
        'baseline_mape': baseline_mape,
        'total_improvement': total_improvement
    }
    with open(results_filename, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✅ Enhanced results and iteration codes saved to: {results_filename}")
    
    # Generate detailed report with statistical analysis
    report_filename = f'cache/{stock}_iterative_report_enhanced.txt'
    with open(report_filename, 'w', encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("ENHANCED ITERATIVE AGENT-BASED FEATURE SELECTION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Stock: {stock}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Iterations: {len(iteration_results) - 1}\n")
        f.write("\n")
        
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Baseline MAPE: {baseline_mape:.2f}%\n")
        f.write(f"Best MAPE: {best_result['mape']:.2f}%\n")
        f.write(f"Total Improvement: {total_improvement:.2f}% ({improvement_percentage:.1f}% relative)\n")
        f.write(f"Feature Reduction: {feature_reduction:.1f}%\n")
        f.write(f"Success Rate: {efficiency:.1f}%\n")
        f.write("\n")
        
        f.write("DETAILED RESULTS WITH STATISTICAL ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        for result in iteration_results:
            f.write(f"Iteration {result['iteration']}: {result['model_name']}\n")
            f.write(f"  MAPE: {result['mape']:.2f}%\n")
            f.write(f"  Features: {result['feature_count']}\n")
            f.write(f"  Improvement: {result['improvement']:+.2f}%\n")
            f.write(f"  Source: {result.get('function_source', 'baseline')}\n")
            
            # Add statistical significance information
            if 'feature_stats' in result and result['feature_stats']:
                significant_count = len(result.get('significant_features', []))
                highly_significant_count = len(result.get('highly_significant_features', []))
                f.write(f"  Statistical Analysis:\n")
                f.write(f"    • Significant features (p < 0.05): {significant_count}\n")
                f.write(f"    • Highly significant features (p < 0.01): {highly_significant_count}\n")
                
                # Add most significant feature
                if result['feature_stats']:
                    min_pval_feature = min(result['feature_stats'].items(), key=lambda x: x[1]['p_value'])
                    f.write(f"    • Most significant feature: {min_pval_feature[0]} (p={min_pval_feature[1]['p_value']:.4f})\n")
            
            f.write("\n")
        
        f.write("BEST MODEL DETAILS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model: {best_result['model_name']}\n")
        f.write(f"MAPE: {best_result['mape']:.2f}%\n")
        f.write(f"MAE: {best_result['mae']:.4f}\n")
        f.write(f"RMSE: {best_result['rmse']:.4f}\n")
        f.write(f"Features: {best_result['feature_count']}\n")
        f.write(f"Source: {best_result.get('function_source', 'baseline')}\n")
        
        # Add detailed statistical analysis for best model
        if 'feature_stats' in best_result and best_result['feature_stats']:
            f.write(f"\nSTATISTICAL SIGNIFICANCE ANALYSIS:\n")
            f.write(f"Total features: {len(best_result['feature_stats'])}\n")
            f.write(f"Significant features (p < 0.05): {len(best_result.get('significant_features', []))}\n")
            f.write(f"Highly significant features (p < 0.01): {len(best_result.get('highly_significant_features', []))}\n")
            
            # Top 10 most significant features
            if best_result['feature_stats']:
                sorted_features = sorted(best_result['feature_stats'].items(), key=lambda x: x[1]['p_value'])
                f.write(f"\nTOP 10 MOST SIGNIFICANT FEATURES:\n")
                for i, (name, stats) in enumerate(sorted_features[:10]):
                    significance = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else ""
                    f.write(f"{i+1:2d}. {name}: p={stats['p_value']:.4f} {significance}\n")
        
        if best_result.get('claude_code'):
            f.write("\nBEST MODEL FEATURE ENGINEERING CODE:\n")
            f.write("-" * 40 + "\n")
            f.write(best_result['claude_code'])
        
        # Add comprehensive iteration codes summary
        f.write("\n\nALL ITERATION CODES SUMMARY:\n")
        f.write("=" * 50 + "\n")
        f.write("This section contains all the feature engineering codes generated during the iterative process.\n\n")
        
        for iteration_key, code_info in iteration_codes.items():
            f.write(f"{iteration_key.upper()}:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Performance: MAPE = {code_info['mape']:.2f}%\n")
            f.write(f"Improvement: {code_info['improvement']:+.2f}%\n")
            f.write(f"Features: {code_info['feature_count']}\n")
            f.write(f"Source: {code_info['function_source']}\n")
            f.write(f"Errors: {code_info['errors_encountered']}\n")
            f.write(f"Significant Features: {code_info['significant_features']}\n")
            f.write(f"Highly Significant Features: {code_info['highly_significant_features']}\n")
            f.write(f"\nCode:\n")
            f.write("```python\n")
            f.write(code_info['code'])
            f.write("\n```\n\n")
        
        # Add iteration codes dictionary to the saved results
        f.write("\nITERATION CODES DICTIONARY (for programmatic access):\n")
        f.write("-" * 50 + "\n")
        f.write("The iteration_codes dictionary contains all codes with metadata for easy programmatic access.\n")
        f.write("Keys: iteration_1, iteration_2, etc.\n")
        f.write("Each entry contains: code, function_source, mape, improvement, feature_count, errors_encountered, significant_features, highly_significant_features\n")
    
    print(f"✅ Enhanced detailed report saved to: {report_filename}")
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 ENHANCED ITERATIVE AGENT-BASED FEATURE SELECTION COMPLETED!")
    print("="*70)
    print(f"📊 Final Results:")
    print(f"   • Baseline MAPE: {baseline_mape:.2f}%")
    print(f"   • Best MAPE: {best_result['mape']:.2f}%")
    print(f"   • Total Improvement: {total_improvement:.2f}%")
    print(f"   • Feature Reduction: {feature_reduction:.1f}%")
    print(f"   • Iterations Completed: {len(iteration_results) - 1}")
    print(f"   • Success Rate: {efficiency:.1f}%")
    
    # Statistical insights
    if 'feature_stats' in best_result and best_result['feature_stats']:
        print(f"\n📈 Statistical Insights:")
        print(f"   • Significant features: {len(best_result.get('significant_features', []))}/{len(best_result['feature_stats'])}")
        print(f"   • Highly significant features: {len(best_result.get('highly_significant_features', []))}")
    
    # Iteration codes summary
    if iteration_codes:
        print(f"\n📝 Iteration Codes Summary:")
        print(f"   • Total iterations with code: {len(iteration_codes)}")
        for iteration_key, code_info in iteration_codes.items():
            print(f"   • {iteration_key}: MAPE={code_info['mape']:.2f}%, Features={code_info['feature_count']}, Source={code_info['function_source']}")
    
    print(f"\n💾 Files Saved:")
    print(f"   • Results: {results_filename}")
    print(f"   • Report: {report_filename}")
    print("\n✅ Enhanced process completed successfully!")


if __name__ == "__main__":
    main()
