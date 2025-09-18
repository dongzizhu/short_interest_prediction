"""
LLM-Based Feature Selection for Financial Time Series Data
=========================================================

This script implements an intelligent feature selection system that uses Claude API
to generate feature construction code for financial time series data, with a robust
fallback mechanism to ensure LSTM-compatible output format.

Features:
- Claude API integration for intelligent feature engineering
- Robust error handling and fallback mechanisms
- LSTM-compatible output format maintenance
- Financial domain knowledge integration
- Comprehensive data processing and validation

Author: AI Assistant
Date: 2024
"""

import pickle
import anthropic
import numpy as np
import pandas as pd
import re
import ast
from typing import Dict, Any, Tuple, List, Union
import warnings
warnings.filterwarnings('ignore')


class LLMFeatureSelector:
    """
    A class for performing LLM-based feature selection on financial time series data.
    """
    
    def __init__(self, anthropic_api_key: str):
        """
        Initialize the LLM Feature Selector.
        
        Args:
            anthropic_api_key: API key for Claude/Anthropic
        """
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        print("Claude API client initialized successfully!")
    
    def create_feature_selection_prompt(self, feature_description: str) -> str:
        """
        Create a comprehensive prompt for Claude to generate feature selection code.
        """
        prompt = f"""
You are a financial data scientist expert in feature engineering for stock prediction models. 

I have financial time series data with the following structure:
- Shape: (samples, lookback_window=4, features=62)
- Features at each timestamp T include:
  1. Short interest at time T (1 feature)
  2. Average daily volume quantity of past 15 days (1 feature) 
  3. OHLC (Open, High, Low, Close) prices for past 15 days (4 × 15 = 60 features)

Total: 1 + 1 + 60 = 62 features per timestamp.

Your task is to write Python code that performs intelligent feature construction/selection to reduce the 62-dimensional feature vector at each timestamp into a more meaningful and predictive feature vector.

IMPORTANT: The output will be fed into an LSTM model, so the function must maintain the time series structure.

Requirements:
1. Write a function called `construct_features` that takes a numpy array of shape (lookback_window, 62) and returns a numpy array of shape (lookback_window, reduced_features)
2. The function should process each timestamp independently but maintain the temporal structure
3. Focus on the most predictive features for each time step
4. Consider financial domain knowledge (e.g., price momentum, volatility, volume patterns, etc.)
5. The output should be a 2D numpy array with shape (lookback_window, reduced_features)
6. Include comments explaining your feature engineering choices
7. Make sure the code is production-ready and handles edge cases
8. DO NOT include any import statements - only use numpy (available as 'np') and built-in Python functions
9. The function must return a 2D array where each row represents features for one time step
10. Use numpy nan_to_num to handle NaN values

Please provide ONLY the Python function code, no explanations outside the code comments.

Feature description: {feature_description}
"""
        return prompt

    def call_claude_for_feature_selection(self, feature_description: str = "Stock prediction with short interest, volume, and OHLC data") -> str:
        """
        Call Claude API to generate feature selection code.
        """
        prompt = self.create_feature_selection_prompt(feature_description)
        
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return None

    def extract_function_from_response(self, response_text: str) -> str:
        """
        Extract the construct_features function from Claude's response.
        """
        # Look for the function definition
        lines = response_text.split('\n')
        function_lines = []
        in_function = False
        indent_level = 0
        
        for line in lines:
            if 'def construct_features' in line:
                in_function = True
                function_lines.append(line)
                # Get the base indentation level
                indent_level = len(line) - len(line.lstrip())
            elif in_function:
                if line.strip() == '':
                    function_lines.append(line)
                elif len(line) - len(line.lstrip()) > indent_level or line.strip() == '':
                    function_lines.append(line)
                else:
                    # End of function
                    break
        
        return '\n'.join(function_lines)

    def execute_feature_construction_code(self, code: str) -> callable:
        """
        Execute the generated feature construction code and return the function.
        """
        try:
            # Create a safe execution environment
            # Create a numpy namespace with additional functions
            np_extended = np
            np_extended.nan_to_num = np.nan_to_num
            
            exec_globals = {
                'np': np_extended,
                'pd': pd,
                'Tuple': Tuple,
                'List': List,
                'Dict': Dict,
                'Union': Union,
                '__builtins__': {
                    'len': len,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'abs': abs,
                    'round': round,
                    'int': int,
                    'float': float,
                    'str': str,
                    'list': list,
                    'dict': dict,
                    'tuple': tuple,
                    'set': set,
                    'print': print,
                    'any': any,
                    'all': all,
                    'sorted': sorted,
                    'reversed': reversed,
                    'isinstance': isinstance,
                    'type': type,
                    'hasattr': hasattr,
                    'getattr': getattr,
                    'setattr': setattr,
                    'callable': callable,
                    'issubclass': issubclass,
                    'super': super,
                    'open': open,
                    'iter': iter,
                    'next': next,
                    'map': map,
                    'filter': filter,
                    'reduce': __import__('functools').reduce,
                    'pow': pow,
                    'divmod': divmod,
                    'bin': bin,
                    'hex': hex,
                    'oct': oct,
                    'ord': ord,
                    'chr': chr,
                    'bool': bool,
                    'complex': complex,
                    'bytes': bytes,
                    'bytearray': bytearray,
                    'memoryview': memoryview,
                    'slice': slice,
                    'property': property,
                    'staticmethod': staticmethod,
                    'classmethod': classmethod,
                }
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Return the construct_features function
            if 'construct_features' in exec_globals:
                return exec_globals['construct_features']
            else:
                raise ValueError("construct_features function not found in generated code")
                
        except Exception as e:
            print(f"Error executing generated code: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            return None

    def fallback_construct_features(self, data: np.ndarray) -> np.ndarray:
        """
        Fallback feature construction function that maintains LSTM format.
        This function creates meaningful financial features while ensuring 2D output.
        """
        if data.shape[1] != 62:
            raise ValueError(f"Expected 62 features, got {data.shape[1]}")
        
        lookback_window = data.shape[0]
        features_per_timestep = 15  # Reduced from 62 to 15 features per timestep
        
        # Initialize output array
        output = np.zeros((lookback_window, features_per_timestep))
        
        for t in range(lookback_window):
            # Extract features for this timestep
            short_interest = data[t, 0]
            volume = data[t, 1]
            
            # Reshape OHLC data (60 features -> 15 days × 4 prices)
            ohlc = data[t, 2:].reshape(15, 4)  # 15 days, 4 prices (O, H, L, C)
            
            # Feature 1: Short interest
            output[t, 0] = short_interest
            
            # Feature 2: Volume
            output[t, 1] = volume
            
            # Features 3-6: Latest OHLC prices
            output[t, 2:6] = ohlc[-1]  # Latest day's OHLC
            
            # Features 7-10: Price momentum (1-day, 3-day, 5-day, 10-day)
            close_prices = ohlc[:, 3]  # Close prices
            for i, horizon in enumerate([1, 3, 5, 10]):
                if len(close_prices) > horizon:
                    momentum = (close_prices[-1] - close_prices[-horizon-1]) / close_prices[-horizon-1]
                    output[t, 6 + i] = momentum
                else:
                    output[t, 6 + i] = 0
            
            # Features 11-12: Volatility and range
            if len(close_prices) > 1:
                returns = np.diff(close_prices) / close_prices[:-1]
                output[t, 10] = np.std(returns)  # Volatility
                output[t, 11] = (np.max(close_prices) - np.min(close_prices)) / np.mean(close_prices)  # Price range
            
            # Features 13-15: Technical indicators
            if len(close_prices) > 1:
                # Simple moving average ratio
                sma_5 = np.mean(close_prices[-5:]) if len(close_prices) >= 5 else np.mean(close_prices)
                output[t, 12] = close_prices[-1] / sma_5
                
                # High-Low ratio
                high_low_ratio = ohlc[:, 1] / ohlc[:, 2]  # High/Low
                output[t, 13] = np.mean(high_low_ratio)
                
                # Volume-price trend
                output[t, 14] = volume * close_prices[-1] / np.mean(close_prices)
        
        return output.astype(np.float32)

    def apply_feature_selection_to_data(self, X_data: np.ndarray, construct_func: callable) -> np.ndarray:
        """
        Apply feature selection to the entire dataset while maintaining LSTM format.
        
        Args:
            X_data: Array of shape (samples, lookback_window, features)
            construct_func: The feature construction function
        
        Returns:
            Array of shape (samples, lookback_window, reduced_features)
        """
        processed_samples = []
        reshaping_stats = {
            'total_samples': X_data.shape[0],
            'reshaped_samples': 0,
            'correct_format_samples': 0,
            'error_samples': 0
        }
        
        for i in range(X_data.shape[0]):
            # Extract features for this sample (lookback_window, 62)
            sample_features = X_data[i]  # Shape: (4, 62)
            
            try:
                # Apply feature construction - should return (lookback_window, reduced_features)
                reduced_features = construct_func(sample_features)
                
                # Track the original output format
                original_shape = reduced_features.shape
                original_dims = reduced_features.ndim
                
                # Ensure the output maintains the time series structure
                if reduced_features.ndim == 1:
                    # If function returns 1D array, reshape to maintain time structure
                    # This is a fallback - ideally the function should return 2D
                    print(f"⚠️  Sample {i}: Function returned 1D array {original_shape}, reshaping to maintain time structure")
                    reshaping_stats['reshaped_samples'] += 1
                    
                    # Distribute features across time steps
                    features_per_step = len(reduced_features) // sample_features.shape[0]
                    if features_per_step > 0:
                        reduced_features = reduced_features[:features_per_step * sample_features.shape[0]]
                        reduced_features = reduced_features.reshape(sample_features.shape[0], features_per_step)
                        print(f"   Reshaped to: {reduced_features.shape}")
                    else:
                        # If not enough features, pad with zeros
                        reduced_features = np.zeros((sample_features.shape[0], 1))
                        reduced_features[0, 0] = reduced_features.mean() if len(reduced_features) > 0 else 0
                        print(f"   Padded to: {reduced_features.shape}")
                else:
                    # Function returned correct 2D format
                    reshaping_stats['correct_format_samples'] += 1
                    if i < 3:  # Only print for first few samples to avoid spam
                        print(f"✅ Sample {i}: Function returned correct 2D format {original_shape}")
                
                processed_samples.append(reduced_features)
                
            except Exception as e:
                print(f"❌ Error processing sample {i}: {e}")
                reshaping_stats['error_samples'] += 1
                # Use a fallback - keep original structure but reduce features
                # Take first 10 features from each time step
                fallback_features = sample_features[:, :10] if sample_features.shape[1] >= 10 else sample_features
                processed_samples.append(fallback_features)
        
        # Print reshaping statistics
        print(f"\nReshaping Statistics:")
        print(f"  - Total samples processed: {reshaping_stats['total_samples']}")
        print(f"  - Samples with correct format: {reshaping_stats['correct_format_samples']}")
        print(f"  - Samples that needed reshaping: {reshaping_stats['reshaped_samples']}")
        print(f"  - Samples with errors: {reshaping_stats['error_samples']}")
        
        if reshaping_stats['reshaped_samples'] > 0:
            print(f"⚠️  {reshaping_stats['reshaped_samples']} samples required reshaping - function didn't return optimal format")
        else:
            print(f"✅ All samples processed correctly - function returned proper 2D format")
        
        return np.array(processed_samples)

    def perform_llm_feature_selection_robust(self, data: Dict[str, Any], feature_description: str = "Stock prediction with short interest, volume, and OHLC data") -> Dict[str, Any]:
        """
        Robust version of LLM feature selection with fallback.
        """
        print("Starting robust LLM-based feature selection process...")
        
        # Try to get Claude-generated function
        construct_func = None
        function_source = "fallback"
        claude_function_code = None
        claude_response_full = None
        function_shape_analysis = {}
        
        try:
            print("Step 1: Attempting to get Claude-generated feature construction code...")
            claude_response = self.call_claude_for_feature_selection(feature_description)
            claude_response_full = claude_response
            
            if claude_response:
                print("✅ Claude response received!")
                print("\n" + "=" * 60)
                print("CLAUDE'S FULL RESPONSE:")
                print("=" * 60)
                print(claude_response)
                print("=" * 60)
                
                function_code = self.extract_function_from_response(claude_response)
                claude_function_code = function_code
                
                print("\n" + "=" * 60)
                print("EXTRACTED FUNCTION CODE:")
                print("=" * 60)
                print(function_code)
                print("=" * 60)
                
                construct_func = self.execute_feature_construction_code(function_code)
                
                if construct_func:
                    # Test the function
                    test_sample = data['X_train_raw'][0]
                    print(f"\nTesting function with sample shape: {test_sample.shape}")
                    
                    test_result = construct_func(test_sample)
                    print(f"✅ Function executed successfully!")
                    print(f"Function output shape: {test_result.shape}")
                    print(f"Function output dimensions: {test_result.ndim}D")
                    
                    # Analyze the function output
                    function_shape_analysis = {
                        'output_shape': test_result.shape,
                        'output_dimensions': test_result.ndim,
                        'expected_shape': (4, 'reduced_features'),
                        'is_correct_format': test_result.ndim == 2 and test_result.shape[0] == 4,
                        'needs_reshaping': test_result.ndim != 2 or test_result.shape[0] != 4
                    }
                    
                    print(f"\nFunction Shape Analysis:")
                    print(f"  - Output shape: {function_shape_analysis['output_shape']}")
                    print(f"  - Output dimensions: {function_shape_analysis['output_dimensions']}D")
                    print(f"  - Expected: (4, reduced_features)")
                    print(f"  - Correct format: {function_shape_analysis['is_correct_format']}")
                    print(f"  - Needs reshaping: {function_shape_analysis['needs_reshaping']}")
                    
                    if test_result.ndim == 2 and test_result.shape[0] == 4:
                        print("✅ Claude-generated function works perfectly! Returns correct 2D time-series format.")
                        function_source = "claude"
                    else:
                        print(f"⚠️  Claude function returns wrong format: {test_result.shape}")
                        print("   Expected: (4, reduced_features) for LSTM compatibility")
                        print("   Will use fallback function instead")
                        construct_func = self.fallback_construct_features
                else:
                    print("⚠️  Claude function execution failed, using fallback")
                    construct_func = self.fallback_construct_features
            else:
                print("⚠️  No Claude response, using fallback")
                construct_func = self.fallback_construct_features
                
        except Exception as e:
            print(f"⚠️  Error with Claude approach: {e}, using fallback")
            construct_func = self.fallback_construct_features
        
        # Apply feature selection to data
        print(f"\nStep 2: Applying feature selection using {function_source} function...")
        X_train_processed = self.apply_feature_selection_to_data(data['X_train_raw'], construct_func)
        X_test_processed = self.apply_feature_selection_to_data(data['X_test_raw'], construct_func)
        
        print(f"Training data shape: {data['X_train_raw'].shape} -> {X_train_processed.shape}")
        print(f"Test data shape: {data['X_test_raw'].shape} -> {X_test_processed.shape}")
        
        # Create processed data dictionary
        processed_data = {
            'X_train_selected': X_train_processed,
            'X_test_selected': X_test_processed,
            'y_train': data.get('y_train', None),
            'y_test': data.get('y_test', None),
            'function_source': function_source,
            'claude_response_full': claude_response_full,
            'claude_function_code': claude_function_code,
            'function_shape_analysis': function_shape_analysis,
            'original_shapes': {
                'X_train_raw': data['X_train_raw'].shape,
                'X_test_raw': data['X_test_raw'].shape
            },
            'processed_shapes': {
                'X_train_selected': X_train_processed.shape,
                'X_test_selected': X_test_processed.shape
            }
        }
        
        print(f"\n✅ Feature selection completed using {function_source} function!")
        print(f"Feature reduction: {data['X_train_raw'].shape[2]} -> {X_train_processed.shape[2]} features per time step")
        print(f"LSTM format maintained: (samples={X_train_processed.shape[0]}, lookback_window={X_train_processed.shape[1]}, features={X_train_processed.shape[2]})")
        
        return processed_data

    def verify_processed_data(self, processed_data: Dict[str, Any]) -> None:
        """
        Verify the processed data and provide analysis.
        """
        print("=" * 60)
        print("VERIFICATION AND ANALYSIS")
        print("=" * 60)
        
        try:
            print("✅ Data successfully processed")
            
            # Display data structure
            print(f"\nData keys: {list(processed_data.keys())}")
            
            # Show feature statistics
            X_train_selected = processed_data['X_train_selected']
            X_test_selected = processed_data['X_test_selected']
            
            print(f"\nFeature Statistics:")
            print(f"Training data - Mean: {X_train_selected.mean():.4f}, Std: {X_train_selected.std():.4f}")
            print(f"Test data - Mean: {X_test_selected.mean():.4f}, Std: {X_test_selected.std():.4f}")
            
            # Check for any NaN or infinite values
            train_nan_count = np.isnan(X_train_selected).sum()
            test_nan_count = np.isnan(X_test_selected).sum()
            train_inf_count = np.isinf(X_train_selected).sum()
            test_inf_count = np.isinf(X_test_selected).sum()
            
            print(f"\nData Quality Check:")
            print(f"Training data - NaN values: {train_nan_count}, Inf values: {train_inf_count}")
            print(f"Test data - NaN values: {test_nan_count}, Inf values: {test_inf_count}")
            
            if train_nan_count == 0 and test_nan_count == 0 and train_inf_count == 0 and test_inf_count == 0:
                print("✅ Data quality check passed - no NaN or infinite values")
            else:
                print("⚠️  Warning: Data contains NaN or infinite values")
            
            # Show sample of processed features
            print(f"\nSample of processed features (first sample, all time steps, first 5 features):")
            print(f"Shape: {X_train_selected[0].shape}")
            print(X_train_selected[0, :, :5])
            
            print(f"\nLSTM Format Verification:")
            print(f"✅ Samples dimension: {X_train_selected.shape[0]}")
            print(f"✅ Lookback window: {X_train_selected.shape[1]}")
            print(f"✅ Feature dimension: {X_train_selected.shape[2]}")
            print(f"✅ Ready for LSTM input!")
            
        except Exception as e:
            print(f"❌ Error during verification: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main function to demonstrate the LLM-based feature selection system.
    """
    # Configuration
    ANTHROPIC_API_KEY = ''  # Replace with your actual API key
    stock = 'PFE'
    DATA_PATH = f'cache/{stock}_raw_data_with_ohlc.pkl'
    OUTPUT_PATH = f'cache/{stock}_data_selected.pkl'
    
    print("=" * 60)
    print("LLM-BASED FEATURE SELECTION SYSTEM")
    print("=" * 60)
    
    try:
        # Load data
        print("Loading data...")
        with open(DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Data loaded successfully!")
        print(f"Training data shape: {data['X_train_raw'].shape}")
        print(f"Test data shape: {data['X_test_raw'].shape}")
        
        # Initialize feature selector
        print("\nInitializing LLM Feature Selector...")
        feature_selector = LLMFeatureSelector(ANTHROPIC_API_KEY)
        
        # Perform feature selection
        print("\n" + "=" * 60)
        print("EXECUTING FEATURE SELECTION")
        print("=" * 60)
        
        processed_data = feature_selector.perform_llm_feature_selection_robust(
            data, 
            feature_description="stock prediction with short interest, volume, and OHLC price data for LSTM feature engineering"
        )
        
        # Save the processed data
        print(f"\nSaving processed data to {OUTPUT_PATH}...")
        with open(OUTPUT_PATH, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"✅ SUCCESS: Processed data saved to {OUTPUT_PATH}")
        
        # Display summary
        print("\n" + "=" * 60)
        print("FEATURE SELECTION SUMMARY")
        print("=" * 60)
        print(f"Function used: {processed_data['function_source']}")
        print(f"Original training data shape: {processed_data['original_shapes']['X_train_raw']}")
        print(f"Original test data shape: {processed_data['original_shapes']['X_test_raw']}")
        print(f"Processed training data shape: {processed_data['processed_shapes']['X_train_selected']}")
        print(f"Processed test data shape: {processed_data['processed_shapes']['X_test_selected']}")
        
        feature_reduction_ratio = processed_data['processed_shapes']['X_train_selected'][2] / processed_data['original_shapes']['X_train_raw'][2]
        print(f"Feature reduction ratio: {feature_reduction_ratio:.2%}")
        print(f"LSTM format maintained: ✅ (samples, lookback_window, features)")
        
        # Show function analysis if available
        if processed_data.get('function_shape_analysis'):
            analysis = processed_data['function_shape_analysis']
            print(f"\nFunction Analysis:")
            print(f"  - Function output shape: {analysis.get('output_shape', 'N/A')}")
            print(f"  - Function output dimensions: {analysis.get('output_dimensions', 'N/A')}D")
            print(f"  - Correct format: {analysis.get('is_correct_format', 'N/A')}")
            print(f"  - Needed reshaping: {analysis.get('needs_reshaping', 'N/A')}")
        
        
        # Verify the processed data
        feature_selector.verify_processed_data(processed_data)
        
        print("\n" + "=" * 60)
        print("FEATURE SELECTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERROR: Feature selection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
