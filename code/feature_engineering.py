"""
Feature engineering and LLM integration module for the iterative agent-based feature selection system.

This module handles LLM-based feature engineering, prompt generation, and iterative improvement.
"""

import numpy as np
import pandas as pd
import anthropic
from typing import Dict, List, Tuple, Any, Optional, Callable
import time
from datetime import datetime
import math
import statistics
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    from .config import LLMConfig
    from .utils import validate_feature_engineering_output, get_feature_names
    from .prompt import create_iterative_prompt_template, create_universal_prompt_template
except ImportError:
    from config import LLMConfig
    from utils import validate_feature_engineering_output, get_feature_names
    from prompt import create_iterative_prompt_template, create_universal_prompt_template


class IterativeLLMFeatureSelector:
    """Iterative LLM-based feature selector with retry mechanism."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
        print("✅ Claude API client initialized successfully!")
    
    def create_iterative_prompt(self, iteration_num: int, previous_results: List[Dict[str, Any]], 
                               error_feedback: List[Dict[str, Any]] = None) -> str:
        """Create a prompt that includes performance history and asks for improvements."""
        return create_iterative_prompt_template(iteration_num, previous_results, error_feedback)
    
    def call_claude_for_iterative_improvement(self, iteration_num: int, previous_results: List[Dict[str, Any]], 
                                            error_feedback: List[Dict[str, Any]] = None) -> str:
        """Call Claude API with iterative improvement context."""
        prompt = self.create_iterative_prompt(iteration_num, previous_results, error_feedback)
        
        # Save prompt for debugging
        prompt_file = f"prompt_{iteration_num}.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)
        
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            return None
    
    def extract_function_from_response(self, response_text: str) -> str:
        """Extract the construct_features function from Claude's response."""
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
    
    def execute_feature_construction_code(self, code: str) -> Callable:
        """Extract and validate the feature construction code, return a callable function."""
        try:
            # Create a fresh execution environment for each attempt
            exec_globals = {
                # Core data science libraries
                'np': np,
                'pd': pd,
                'math': math,
                'statistics': statistics,
                'stats': stats,
                'StandardScaler': StandardScaler,
                'MinMaxScaler': MinMaxScaler,
                'mean_squared_error': mean_squared_error,
                'mean_absolute_error': mean_absolute_error,
                
                # Common functions and modules
                'datetime': datetime,
                'time': time,
                
                # Built-in functions
                '__builtins__': __builtins__
            }
            
            # Execute the code in the fresh environment
            exec(code, exec_globals)
            func = exec_globals['construct_features']
            func.__globals__.update(exec_globals)
            
            # Check if construct_features was successfully created
            if 'construct_features' in exec_globals and callable(exec_globals['construct_features']):
                # Test the function with a sample input to ensure it works
                test_input = np.random.rand(4, 97)  # Sample input with expected shape
                try:
                    test_output = func(test_input)
                    if not isinstance(test_output, np.ndarray):
                        raise ValueError("Function did not return a numpy array")
                    if test_output.ndim != 2:
                        raise ValueError(f"Function returned array with {test_output.ndim} dimensions, expected 2")
                    print("✅ Successfully Extracted Function Code!")
                    print(f"Test output shape: {test_output.shape}")
                    
                    # Return a wrapper function that uses our execution method
                    # def wrapped_construct_features(data):
                    #     return self._execute_feature_function(code, data)
                    
                    
                    return func
                except Exception as test_error:
                    raise ValueError(f"Function failed test execution: {test_error}")
            else:
                raise ValueError("construct_features function not found or not callable in generated code")
                
        except Exception as e:
            print(f"❌ Error executing generated code: {e}")
            return None
    
    def fallback_construct_features(self, data: np.ndarray) -> np.ndarray:
        """Fallback feature construction function."""
        if data.shape[1] != 97:
            raise ValueError(f"Expected 97 features, got {data.shape[1]}")
        
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
    
    def _execute_feature_function(self, func_code: str, data: np.ndarray) -> np.ndarray:
        """Execute the feature construction function with proper imports."""
        try:
            # Create execution environment with all necessary imports
            exec_globals = {
                'np': np,
                'pd': pd,
                'math': math,
                'statistics': statistics,
                'stats': stats,
                'StandardScaler': StandardScaler,
                'MinMaxScaler': MinMaxScaler,
                'mean_squared_error': mean_squared_error,
                'mean_absolute_error': mean_absolute_error,
                'datetime': datetime,
                'time': time,
                '__builtins__': __builtins__
            }
            
            # Execute the function code
            exec(func_code, exec_globals)
            
            # Get the function and update its globals to include our imports
            construct_func = exec_globals['construct_features']
            construct_func.__globals__.update(exec_globals)
            
            # Call the function
            return construct_func(data)
            
        except Exception as e:
            print(f"❌ Error executing feature construction function: {e}")
            raise

    def apply_feature_selection_to_data(self, X_data: np.ndarray, construct_func: Callable, 
                                      max_retries: int = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Apply feature selection to the entire dataset with retry mechanism."""
        if max_retries is None:
            max_retries = self.config.max_feature_retries
            
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
        """Fallback feature selection when all retries fail."""
        print("🆘 Using fallback feature selection...")
        processed_samples = []
        
        for i in range(X_data.shape[0]):
            sample_features = X_data[i]
            # Simple fallback: take first 15 features from each timestep
            fallback_features = sample_features[:, :15] if sample_features.shape[1] >= 15 else sample_features
            processed_samples.append(fallback_features)
        
        return np.array(processed_samples)


class UniversalFeatureEngineering:
    """Universal feature engineering code generation from multiple ticker results."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
    
    def extract_function_from_response(self, response_text: str) -> str:
        """Extract the construct_features function from Claude's response."""
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
    
    def execute_feature_construction_code(self, code: str) -> Callable:
        """
        Execute the feature construction code and return the function.
        
        Args:
            code (str): The feature construction code as a string
            
        Returns:
            Callable: The construct_features function if successful, None otherwise
        """
        try:
            # Create a safe execution environment
            exec_globals = {
                # Core data science libraries
                'np': np,
                'pd': pd,
                'math': math,
                'statistics': statistics,
                'stats': stats,
                'StandardScaler': StandardScaler,
                'MinMaxScaler': MinMaxScaler,
                'mean_squared_error': mean_squared_error,
                'mean_absolute_error': mean_absolute_error,
                
                # Common functions and modules
                'datetime': datetime,
                'time': time,
                
                # Built-in functions
                '__builtins__': __builtins__
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Get the function
            if 'construct_features' in exec_globals:
                return exec_globals['construct_features']
            else:
                print("❌ construct_features function not found in executed code")
                return None
                
        except Exception as e:
            print(f"❌ Error executing feature construction code: {e}")
            return None
    
    def create_universal_prompt(self, ticker_results: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
        """Create a prompt to generate universal feature engineering code from multiple ticker results."""
        return create_universal_prompt_template(ticker_results)
    
    def call_claude_for_universal_code(self, ticker_results: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
        """Call Claude API to generate universal feature engineering code with retry logic and validation."""
        prompt = self.create_universal_prompt(ticker_results)
        
        # Save the prompt for reference
        with open("prompt_universal.txt", "w", encoding="utf-8") as f:
            f.write(prompt)
        
        max_retries = 3
        base_delay = 1  # seconds
        validation_errors = []
        
        for attempt in range(max_retries + 1):
            try:
                # Add error feedback to prompt if we have previous validation errors
                current_prompt = prompt
                if validation_errors:
                    error_feedback = self._create_validation_error_feedback(validation_errors)
                    current_prompt = f"{prompt}\n\n{error_feedback}"
                
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=8000,
                    temperature=0.1,  # Lower temperature for more consistent synthesis
                    messages=[{"role": "user", "content": current_prompt}]
                )
                
                universal_response = response.content[0].text
                
                # Validate the response by extracting and testing the function
                if self._validate_universal_code(universal_response):
                    print("✅ Universal code generated and validated successfully!")
                    return universal_response
                else:
                    # If validation fails, add to errors and retry
                    validation_errors.append({
                        'attempt': attempt + 1,
                        'error_type': 'ValidationError',
                        'error_message': 'Universal function failed validation with mock data',
                        'code_snippet': universal_response[:200] + "..." if universal_response else "No code"
                    })
                    
                    if attempt < max_retries:
                        print(f"⚠️ Universal code validation failed on attempt {attempt + 1}. Retrying...")
                        import time
                        time.sleep(base_delay * (2 ** attempt))
                        continue
                    else:
                        print(f"❌ Universal code validation failed after {max_retries + 1} attempts")
                        return None
                        
            except Exception as e:
                if attempt == max_retries:
                    print(f"Error calling Claude API for universal code after {max_retries + 1} attempts: {e}")
                    return None
                else:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                    import time
                    time.sleep(delay)
    
    def _validate_universal_code(self, universal_response: str) -> bool:
        """Validate the universal code by extracting and testing it with mock data."""
        try:
            # Extract the function from the response
            function_code = self.extract_function_from_response(universal_response)
            
            # Execute the feature construction code
            construct_func = self.execute_feature_construction_code(function_code)
            
            if not construct_func:
                print("❌ Failed to extract or execute universal function")
                return False
            
            # Test with mock data similar to single ticker validation
            import numpy as np
            np.random.seed(42)  # For reproducible testing
            test_input = np.random.rand(4, 97).astype(np.float32)  # Sample input with expected shape (lookback_window, features)
            
            try:
                test_result = construct_func(test_input)
                
                # Validate output format
                if not isinstance(test_result, np.ndarray):
                    print(f"❌ Universal function returned non-array: {type(test_result)}")
                    return False
                
                if test_result.ndim != 2:
                    print(f"❌ Universal function returned wrong dimensions: {test_result.ndim}D, expected 2D")
                    return False
                
                if test_result.shape[0] != test_input.shape[0]:
                    print(f"❌ Universal function returned wrong number of rows: {test_result.shape[0]}, expected {test_input.shape[0]}")
                    return False
                
                # Check for NaN or infinite values
                if np.any(np.isnan(test_result)) or np.any(np.isinf(test_result)):
                    print("❌ Universal function returned NaN or infinite values")
                    return False
                
                print(f"✅ Universal function validation passed! Output shape: {test_result.shape}")
                return True
                
            except Exception as test_error:
                print(f"❌ Universal function execution failed: {test_error}")
                return False
                
        except Exception as e:
            print(f"❌ Universal code validation failed: {e}")
            return False
    
    def _create_validation_error_feedback(self, validation_errors: List[Dict[str, Any]]) -> str:
        """Create error feedback message for Claude based on validation failures."""
        error_summary = "VALIDATION ERROR FEEDBACK:\n"
        error_summary += "The previous universal code failed validation. Here are the issues:\n\n"
        
        for i, error in enumerate(validation_errors, 1):
            error_summary += f"Attempt {error['attempt']} Error:\n"
            error_summary += f"- Type: {error['error_type']}\n"
            error_summary += f"- Message: {error['error_message']}\n"
            error_summary += f"- Code snippet: {error['code_snippet']}\n\n"
        
        error_summary += """
Please fix the issues in your universal feature engineering function:

The function must be production-ready and handle the mock data validation successfully.
"""
        return error_summary
