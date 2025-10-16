"""
Main orchestration script for the iterative agent-based feature selection system.

This script coordinates all modules to run the complete iterative feature selection process.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pickle
from datetime import datetime
import pandas as pd

from pydantic import constr

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from .config import Config, get_development_config, get_production_config, get_quick_test_config
    from .data_loader import DataLoader
    from .models import ModelTrainer
    from .feature_engineering import IterativeLLMFeatureSelector, UniversalFeatureEngineering
    from .evaluation import PerformanceEvaluator, ReportGenerator, ValidationTester
    from .utils import setup_logging, ProgressTracker, validate_config
except ImportError:
    from config import Config, get_development_config, get_production_config, get_quick_test_config
    from data_loader import DataLoader
    from models import ModelTrainer
    from feature_engineering import IterativeLLMFeatureSelector, UniversalFeatureEngineering
    from evaluation import PerformanceEvaluator, ReportGenerator, ValidationTester
    from utils import setup_logging, ProgressTracker, validate_config


class IterativeFeatureSelectionPipeline:
    """Main pipeline for iterative feature selection."""
    
    def __init__(self, config: Config):
        self.config = config
        validate_config(config)
        
        # Initialize components
        self.data_loader = DataLoader(config.data)
        self.model_trainer = ModelTrainer(config.model)
        self.feature_selector = IterativeLLMFeatureSelector(config.llm)
        self.universal_engineer = UniversalFeatureEngineering(config.llm)
        self.performance_evaluator = PerformanceEvaluator(config.evaluation)
        self.report_generator = ReportGenerator(config.evaluation)
        self.validation_tester = ValidationTester(config.evaluation)
        
        # Set up logging
        setup_logging(config.system.verbose, config.system.log_level)
    
    def run_iterative_process_for_ticker(self, ticker: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the iterative process for a single ticker and return the best code."""
        print(f"🚀 Starting Iterative Agent-Based Feature Selection Process for {ticker}")
        print("="*70)
        
        # Load data
        print("📊 Loading data from parquet file...")
        raw_data = self.data_loader.load_data_for_ticker(ticker)
        
        if not self.data_loader.validate_data_integrity(raw_data):
            raise ValueError(f"Data integrity validation failed for {ticker}")
        
        X_train_raw = raw_data['X_train_raw']
        X_val_raw = raw_data['X_val_raw']
        X_test_raw = raw_data['X_test_raw']
        y_train = raw_data['y_train']
        y_val = raw_data['y_val']
        y_test = raw_data['y_test']
        prev_log_train = raw_data['prev_log_train']
        prev_log_val = raw_data['prev_log_val']
        prev_log_test = raw_data['prev_log_test']
        
        print(f"✅ Data loaded successfully!")
        print(f"Training data shape: {X_train_raw.shape}")
        print(f"Validation data shape: {X_val_raw.shape}")
        print(f"Test data shape: {X_test_raw.shape}")
        print(f"Features per timestep: {X_train_raw.shape[2]}")
        print(f"Lookback window: {X_train_raw.shape[1]}")
        
        # Initialize tracking
        iteration_codes = {}
        iteration_results = []
        
        # Step 1: Run baseline model on validation set
        print("\n🎯 Step 1: Running baseline model on validation set...")
        baseline_results = self.model_trainer.train_and_evaluate_model(
            X_train_raw, X_val_raw, y_train, y_val, prev_log_val, 
            model_name="Baseline (All 97 Features)", epochs=self.config.model.epochs,
            model_type=self.config.model.model_type
        )
        
        baseline_mape = baseline_results['mape']
        print(f"\n📊 Baseline Performance: MAPE = {baseline_mape:.2f}%")
        
        # Store baseline results
        iteration_results.append({
            'iteration': 0,
            'model_name': 'Baseline',
            'features_used': 'All 97 original features',
            'feature_count': X_train_raw.shape[2],
            'mape': baseline_mape,
            'mae': baseline_results['mae'],
            'rmse': baseline_results['rmse'],
            'improvement': 0.0,
            'improvement_over_last': 0.0,
            'predictions': baseline_results['predictions'],
            'feature_stats': baseline_results.get('feature_stats', {}),
            'significant_features': baseline_results.get('significant_features', []),
            'highly_significant_features': baseline_results.get('highly_significant_features', [])
        })
        
        # Iterative improvement loop
        best_mape = baseline_mape
        iterations_without_improvement = 0
        
        print(f"\n🔄 Starting iterative improvement process...")
        print(f"Max iterations: {self.config.llm.max_iterations}")
        print(f"Min improvement threshold: {self.config.llm.min_improvement_threshold}%")
        print(f"Patience: {self.config.llm.patience_iterations} iterations without improvement")
        
        for iteration in range(1, self.config.llm.max_iterations + 1):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}")
            print(f"{'='*70}")
            
            # Get feature engineering code from Claude with retry mechanism
            print(f"🤖 Calling Claude for iteration {iteration}...")
            
            # Retry mechanism for Claude API calls and function execution
            claude_success = False
            claude_errors = []
            
            for claude_retry in range(self.config.llm.max_claude_retries):
                try:
                    # Pass error feedback from previous attempts if available
                    error_feedback = claude_errors if claude_errors else None
                    
                    claude_response = self.feature_selector.call_claude_for_iterative_improvement(
                        iteration, iteration_results, 
                        error_feedback=error_feedback
                    )
                    
                    if not claude_response:
                        error_msg = f"No response from Claude (attempt {claude_retry + 1}/{self.config.llm.max_claude_retries})"
                        print(f"❌ {error_msg}")
                        claude_errors.append({
                            'error_type': 'NoResponse',
                            'error_message': error_msg,
                            'attempt': claude_retry + 1
                        })
                        if claude_retry < self.config.llm.max_claude_retries - 1:
                            print("🔄 Retrying Claude API call...")
                            continue
                        else:
                            print("⚠️ All Claude API attempts failed, using fallback")
                            construct_func = self.feature_selector.fallback_construct_features
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
                        function_code = self.feature_selector.extract_function_from_response(claude_response)
                        construct_func = self.feature_selector.execute_feature_construction_code(function_code)
                        
                        if construct_func:
                            # Test the function with a small sample to ensure it works
                            test_sample = X_train_raw[0]
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
                                if claude_retry < self.config.llm.max_claude_retries - 1:
                                    print("🔄 Retrying with error feedback...")
                                    continue
                                else:
                                    print("⚠️ All function validation attempts failed, using fallback")
                                    construct_func = self.feature_selector.fallback_construct_features
                                    function_source = "fallback"
                                    claude_code = None
                                    claude_success = True
                                    break
                        else:
                            error_msg = f"Function execution failed (attempt {claude_retry + 1}/{self.config.llm.max_claude_retries})"
                            print(f"⚠️ {error_msg}")
                            claude_errors.append({
                                'error_type': 'ExecutionError',
                                'error_message': error_msg,
                                'attempt': claude_retry + 1,
                                'code_snippet': function_code[:200] + "..." if function_code else "No code"
                            })
                            if claude_retry < self.config.llm.max_claude_retries - 1:
                                print("🔄 Retrying function execution with error feedback...")
                                continue
                            else:
                                print("⚠️ All function execution attempts failed, using fallback")
                                construct_func = self.feature_selector.fallback_construct_features
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
                        if claude_retry < self.config.llm.max_claude_retries - 1:
                            print("🔄 Retrying with error feedback...")
                            continue
                        else:
                            print("⚠️ All extraction attempts failed, using fallback")
                            construct_func = self.feature_selector.fallback_construct_features
                            function_source = "fallback"
                            claude_code = None
                            claude_success = True
                            break
                            
                except Exception as e:
                    error_msg = f"Error in Claude process (attempt {claude_retry + 1}/{self.config.llm.max_claude_retries}): {e}"
                    print(f"❌ {error_msg}")
                    claude_errors.append({
                        'error_type': 'ProcessError',
                        'error_message': error_msg,
                        'attempt': claude_retry + 1
                    })
                    if claude_retry < self.config.llm.max_claude_retries - 1:
                        print("🔄 Retrying entire Claude process with error feedback...")
                        continue
                    else:
                        print("⚠️ All Claude process attempts failed, using fallback")
                        construct_func = self.feature_selector.fallback_construct_features
                        function_source = "fallback"
                        claude_code = None
                        claude_success = True
                        break
            
            if not claude_success:
                print("🆘 Critical error: Could not establish feature construction function")
                continue
            
            # Apply feature selection to data with retry mechanism
            print(f"\n🔧 Applying feature selection using {function_source} function...")
            X_train_processed, train_errors = self.feature_selector.apply_feature_selection_to_data(
                X_train_raw, construct_func, max_retries=self.config.llm.max_feature_retries
            )
            X_val_processed, val_errors = self.feature_selector.apply_feature_selection_to_data(
                X_val_raw, construct_func, max_retries=self.config.llm.max_feature_retries
            )
            
            # Combine all errors for feedback
            all_errors = train_errors + val_errors
            
            print(f"Training data shape: {X_train_raw.shape} -> {X_train_processed.shape}")
            print(f"Validation data shape: {X_val_raw.shape} -> {X_val_processed.shape}")
            
            if all_errors:
                print(f"⚠️ Total errors encountered: {len(all_errors)}")
                for i, error in enumerate(all_errors[:3]):  # Show first 3 errors
                    print(f"  Error {i+1}: {error['error_type']} - {error['error_message']}")
            
            # Train on training set and evaluate on validation set
            iteration_results_model = self.model_trainer.train_and_evaluate_model(
                X_train_processed, X_val_processed, y_train, y_val, prev_log_val,
                model_name=f"Iteration {iteration} ({function_source})", epochs=self.config.model.epochs,
                model_type=self.config.model.model_type
            )
            
            # Calculate improvement
            improvement = best_mape - iteration_results_model['mape']
            improvement_over_last = iteration_results[-1]['mape'] - iteration_results_model['mape']
            
            # Store results
            iteration_results.append({
                'iteration': iteration,
                'model_name': f'Iteration {iteration}',
                'features_used': f'{function_source} feature engineering',
                'feature_count': X_train_processed.shape[2],
                'mape': iteration_results_model['mape'],
                'mae': iteration_results_model['mae'],
                'rmse': iteration_results_model['rmse'],
                'improvement': improvement,
                'improvement_over_last': improvement_over_last,
                'predictions': iteration_results_model['predictions'],
                'claude_code': claude_code,
                'function_source': function_source,
                'feature_stats': iteration_results_model.get('feature_stats', {}),
                'significant_features': iteration_results_model.get('significant_features', []),
                'highly_significant_features': iteration_results_model.get('highly_significant_features', []),
                'errors_encountered': len(all_errors),
                'error_details': all_errors[:5] if all_errors else [],
                'claude_errors': claude_errors
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
            if improvement > self.config.llm.min_improvement_threshold:
                print(f"🎉 IMPROVEMENT! MAPE improved by {improvement:.2f}%")
                best_mape = iteration_results_model['mape']
                iterations_without_improvement = 0
            else:
                print(f"📊 No significant improvement. Change: {improvement:+.2f}%")
                iterations_without_improvement += 1
            
            # Check stopping criteria
            if iterations_without_improvement >= self.config.llm.patience_iterations:
                print(f"\n🛑 Stopping: No improvement for {self.config.llm.patience_iterations} consecutive iterations")
                break
            
            print(f"\n📈 Current best MAPE: {best_mape:.2f}%")
            print(f"🔄 Iterations without improvement: {iterations_without_improvement}/{self.config.llm.patience_iterations}")
        
        # Final evaluation on test set
        print(f"\n🎯 Final Evaluation on Test Set (Unseen Data)")
        print("="*70)
        print("Using train+validation data for final model training to maximize data usage")
        
        # Combine training and validation sets for final model training
        X_train_val_raw = np.concatenate([X_train_raw, X_val_raw], axis=0)
        y_train_val = np.concatenate([y_train, y_val], axis=0)
        
        print(f"Combined train+val data shape: {X_train_val_raw.shape}")
        print(f"Test data shape: {X_test_raw.shape}")
        
        # Find the best iteration based on validation performance
        best_iteration_result = min(iteration_results[1:], key=lambda x: x['mape']) if len(iteration_results) > 1 else iteration_results[0]
        
        # 1. BASELINE: Train on train+val with raw features, test on test
        print(f"\n📊 BASELINE EVALUATION (Raw Features, Train+Val → Test):")
        baseline_final_results = self.model_trainer.train_and_evaluate_model(
            X_train_val_raw, X_test_raw, y_train_val, y_test, prev_log_test,
            model_name="Baseline Final (Raw Features)", epochs=self.config.model.epochs,
            model_type=self.config.model.model_type
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
                # Validate the function code and get a callable function
                best_construct_func = self.feature_selector.execute_feature_construction_code(best_function_code)
                
                if best_construct_func:
                    # Apply feature engineering to train+val and test sets
                    X_train_val_processed, _ = self.feature_selector.apply_feature_selection_to_data(
                        X_train_val_raw, best_construct_func, max_retries=self.config.llm.max_feature_retries
                    )
                    X_test_processed_final, _ = self.feature_selector.apply_feature_selection_to_data(
                        X_test_raw, best_construct_func, max_retries=self.config.llm.max_feature_retries
                    )
                    
                    print(f"Processed train+val shape: {X_train_val_processed.shape}")
                    print(f"Processed test shape: {X_test_processed_final.shape}")
                    
                    # Final evaluation with processed features
                    final_test_results = self.model_trainer.train_and_evaluate_model(
                        X_train_val_processed, X_test_processed_final, y_train_val, y_test, prev_log_test,
                        model_name="Best Model Final (Processed Features)", epochs=self.config.model.epochs,
                        model_type=self.config.model.model_type
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
        
        # Generate performance summary
        print("\n" + "="*70)
        print("ITERATION PERFORMANCE SUMMARY")
        print("="*70)
        
        # Create simple performance table
        print(f"\n📊 VALIDATION MAPE TREND:")
        print("-" * 80)
        print(f"{'Iteration':<10} {'Model':<25} {'Validation MAPE':<15} {'Improvement from Last':<20}")
        print("-" * 80)
        
        for result in iteration_results:
            improvement_str = f"{result['improvement']:+.2f}%" if result['improvement'] != 0 else "N/A"
            print(f"{result['iteration']:<10} {result['model_name']:<25} {result['mape']:<15.2f} {improvement_str:<20}")
        
        # Find best result
        best_result = min(iteration_results[1:], key=lambda x: x['mape'])
        print("-" * 80)
        print(f"🏆 Best: {best_result['model_name']} - MAPE: {best_result['mape']:.2f}%")

        # Save results
        if self.config.evaluation.save_individual_results:
            self._save_ticker_results(ticker, best_result, iteration_codes, iteration_results)
        
        print(f"\n🎉 Process completed successfully for {ticker}!")

        return iteration_results[0], best_result, iteration_codes

    def _save_ticker_results(self, ticker: str, best_result: Dict[str, Any], 
                           iteration_codes: Dict[str, Any], iteration_results: List[Dict[str, Any]]) -> None:
        """Save results for a single ticker."""
        # Save individual ticker results
        ticker_summary_file = f'{self.config.evaluation.cache_dir}/{ticker}_iterative_results_enhanced.pkl'
        with open(ticker_summary_file, 'wb') as f:
            pickle.dump({
                'best_result': best_result,
                'iteration_codes': iteration_codes,
                'ticker': ticker
            }, f)
        print(f"✅ Saved {ticker} results to {ticker_summary_file}")
        
        # Generate and save summary report
        summary_content = self.report_generator.generate_iteration_summary(
            ticker, iteration_results, 
            f'{self.config.evaluation.cache_dir}/{ticker}_iterative_summary.txt'
        )
        print(f"✅ Summary report saved for {ticker}")
    
    def run_multi_ticker_process(self, iterative_tickers: List[str], 
                               validation_tickers: List[str]) -> Dict[str, Any]:
        """Run the complete multi-ticker iterative process."""
        print("🚀 Starting Multi-Ticker Iterative Agent-Based Feature Selection Process")
        print("="*80)
        print(f"Processing iterative tickers: {', '.join(iterative_tickers)}")
        print(f"Available for validation: {', '.join(validation_tickers)}")
        print("="*80)
        
        # Dictionary to store results for each ticker
        ticker_results = {}
        ticker_baseline_results = {}
        
        # Process each ticker for iterative feature engineering
        for i, ticker in enumerate(iterative_tickers, 1):
            print(f"\n{'='*80}")
            print(f"PROCESSING TICKER {i}/{len(iterative_tickers)}: {ticker}")
            print(f"{'='*80}")
            
            try:
                baseline_result, best_result, iteration_codes = self.run_iterative_process_for_ticker(ticker)
                ticker_results[ticker] = (best_result, iteration_codes)
                ticker_baseline_results[ticker] = baseline_result
                
            except Exception as e:
                print(f"❌ Error processing {ticker}: {e}")
                print(f"⚠️ Skipping {ticker} and continuing with next ticker...")
                continue
        
        # Check if we have results from at least one ticker
        if not ticker_results:
            print("❌ No tickers were processed successfully. Exiting.")
            return {}
        
        print(f"\n{'='*80}")
        print("GENERATING UNIVERSAL FEATURE ENGINEERING CODE")
        print(f"{'='*80}")
        print(f"Successfully processed {len(ticker_results)} tickers: {', '.join(ticker_results.keys())}")
        
        # Generate universal feature engineering code
        print("\n🤖 Calling Claude to generate universal feature engineering code...")
        universal_response = self.universal_engineer.call_claude_for_universal_code(ticker_results)
        
        universal_function = None
        if universal_response:
            print("✅ Universal code response received!")
            print(f"\n📝 Claude's Universal Feature Engineering Code:")
            print("-" * 60)
            print(universal_response)
            print("-" * 60)
            
            # Extract the universal function
            try:
                universal_function_code = self.feature_selector.extract_function_from_response(universal_response)
                universal_function = self.feature_selector.execute_feature_construction_code(universal_function_code)
                
                if universal_function:
                    print("✅ Universal function extracted and validated successfully!")
                    
                    # Save the universal code
                    if self.config.evaluation.save_comprehensive_results:
                        self._save_universal_code(universal_function_code, universal_response, ticker_results)
                    
                else:
                    print("❌ Failed to extract or validate universal function")
                    
            except Exception as e:
                print(f"❌ Error processing universal code: {e}")
        else:
            print("❌ Failed to get universal code from Claude")
        
        # Final summary of iterative process
        print(f"\n{'='*80}")
        print("ITERATIVE PROCESS SUMMARY")
        print(f"{'='*80}")
        
        for ticker, (best_result, iteration_codes) in ticker_results.items():
            print(f"\n{ticker}:")
            print(f"  Best MAPE: {best_result['mape']:.2f}%")
            print(f"  Improvement: {ticker_baseline_results[ticker]['mape'] - best_result['mape']:.2f}%")
            print(f"  Feature count: {best_result.get('feature_count', 'Unknown')}")
            print(f"  Iterations: {len(iteration_codes)}")
        
        # VALIDATION PHASE: Test universal feature engineering on all available tickers
        validation_results = {}
        if universal_function and self.config.evaluation.test_universal_on_all_tickers:
            print(f"\n{'='*80}")
            print("STARTING VALIDATION PHASE")
            print(f"{'='*80}")
            print("Testing universal feature engineering on all available tickers...")
            
            # Test on all available tickers
            validation_results, successful_tickers, failed_tickers = self.validation_tester.test_universal_feature_engineering(
                universal_function, validation_tickers, self.config.model.model_type, self.data_loader, self.model_trainer
            )
            
            # Generate comprehensive performance report
            performance_report = self.report_generator.generate_performance_report(
                validation_results, successful_tickers, failed_tickers
            )
            
            # Save validation results
            if self.config.evaluation.save_comprehensive_results:
                self._save_validation_results(validation_results, successful_tickers, failed_tickers, 
                                            performance_report, universal_function_code if universal_function else None)
            
            # Final validation summary
            if performance_report:
                stats = performance_report['summary_stats']
                print(f"\n🎯 VALIDATION SUMMARY:")
                print(f"  Successfully tested: {stats['successful']}/{stats['total_tested']} tickers")
                print(f"  Average MAPE improvement: {stats['avg_mape_improvement']:.2f}%")
                print(f"  Tickers with positive improvement: {stats['positive_improvements']}/{stats['successful']} ({stats['positive_improvements']/stats['successful']*100:.1f}%)")
                print(f"  Tickers with >0.5% improvement: {stats['significant_improvements']}/{stats['successful']} ({stats['significant_improvements']/stats['successful']*100:.1f}%)")
        
        print(f"\n🎉 Complete multi-ticker process completed successfully!")
        print(f"Processed {len(ticker_results)} tickers for iterative feature engineering")
        if universal_function:
            print(f"Generated and validated universal feature engineering code on {len(validation_tickers)} tickers")
        
        return {
            'ticker_results': ticker_results,
            'universal_function': universal_function,
            'validation_results': validation_results
        }
    
    def _save_universal_code(self, universal_function_code: str, universal_response: str, 
                           ticker_results: Dict[str, Any]) -> None:
        """Save universal feature engineering code."""
        universal_code_file = f'{self.config.evaluation.cache_dir}/universal_feature_engineering_code.py'
        with open(universal_code_file, 'w', encoding='utf-8') as f:
            f.write('"""\n')
            f.write('Universal Feature Engineering Code for Short Interest Prediction\n')
            f.write('Generated from best practices across multiple tickers\n')
            f.write(f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Source tickers: {", ".join(ticker_results.keys())}\n')
            f.write('"""\n\n')
            f.write('import numpy as np\n\n')
            f.write(universal_function_code)
        
        print(f"💾 Universal code saved to: {universal_code_file}")
        
        # Save comprehensive results
        comprehensive_results = {
            'ticker_results': ticker_results,
            'universal_code': universal_function_code,
            'universal_response': universal_response,
            'generation_timestamp': datetime.now().isoformat(),
            'source_tickers': list(ticker_results.keys())
        }
        
        comprehensive_file = f'{self.config.evaluation.cache_dir}/comprehensive_multi_ticker_results.pkl'
        with open(comprehensive_file, 'wb') as f:
            pickle.dump(comprehensive_results, f)
        print(f"💾 Comprehensive results saved to: {comprehensive_file}")
    
    def _save_validation_results(self, validation_results: List[Dict[str, Any]], 
                               successful_tickers: List[str], failed_tickers: List[str],
                               performance_report: Dict[str, Any], universal_function_code: str) -> None:
        """Save validation results."""
        validation_file = f'{self.config.evaluation.cache_dir}/universal_feature_engineering_validation_results.pkl'
        with open(validation_file, 'wb') as f:
            pickle.dump({
                'validation_results': validation_results,
                'successful_tickers': successful_tickers,
                'failed_tickers': failed_tickers,
                'performance_report': performance_report,
                'universal_function_code': universal_function_code,
                'validation_timestamp': datetime.now().isoformat()
            }, f)
        print(f"💾 Validation results saved to: {validation_file}")


def main():
    """Main function to run the iterative agent-based feature selection process."""
    parser = argparse.ArgumentParser(description='Iterative Agent-Based Feature Selection for Financial Time Series')
    parser.add_argument('--config', type=str, choices=['development', 'production', 'quick_test'], 
                       default='development', help='Configuration preset to use')
    parser.add_argument('--api-key', type=str, help='Anthropic API key (overrides config)')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to process')
    parser.add_argument('--max-tickers', type=int, help='Maximum number of tickers to process')
    parser.add_argument('--single-ticker', type=str, help='Process only a single ticker')
    parser.add_argument('--skip-validation', action='store_true', help='Skip universal validation phase')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config == 'development':
        config = get_development_config()
    elif args.config == 'production':
        config = get_production_config()
    elif args.config == 'quick_test':
        config = get_quick_test_config()
    else:
        config = Config()
    
    # Override API key if provided
    if args.api_key:
        config.llm.api_key = args.api_key
    
    # Set API key from environment if not provided
    if not config.llm.api_key:
        config.llm.api_key = os.getenv('ANTHROPIC_API_KEY', '')
        if not config.llm.api_key:
            print("❌ No API key provided. Set ANTHROPIC_API_KEY environment variable or use --api-key")
            return
    
    # Initialize pipeline
    pipeline = IterativeFeatureSelectionPipeline(config)
    
    # Get available tickers
    if args.single_ticker:
        iterative_tickers = [args.single_ticker]
        validation_tickers = [args.single_ticker]
    elif args.tickers:
        iterative_tickers = args.tickers
        validation_tickers = args.tickers
    else:
        # Get available tickers from data
        all_tickers = pipeline.data_loader.get_available_tickers(args.max_tickers)
        
        # Split into iterative and validation tickers
        if args.max_tickers:
            iterative_tickers = all_tickers[:min(5, args.max_tickers)]  # First 5 for iterative
            validation_tickers = all_tickers[:args.max_tickers]  # All for validation
        else:
            iterative_tickers = all_tickers[:5]  # First 5 for iterative
            validation_tickers = all_tickers  # All for validation
    
    # Skip validation if requested
    if args.skip_validation:
        config.evaluation.test_universal_on_all_tickers = False
    
    # Run the pipeline
    try:
        results = pipeline.run_multi_ticker_process(iterative_tickers, validation_tickers)
        print("\n🎉 Pipeline completed successfully!")
        return results
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
