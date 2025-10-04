"""
Evaluation and reporting module for the iterative agent-based feature selection system.

This module handles performance evaluation, report generation, and validation testing.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from .config import EvaluationConfig
    from .utils import save_results, load_results, format_performance_summary, create_timestamp
except ImportError:
    from config import EvaluationConfig
    from utils import save_results, load_results, format_performance_summary, create_timestamp


class PerformanceEvaluator:
    """Performance evaluation and comparison utilities."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def evaluate_iteration_results(self, iteration_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate and summarize iteration results."""
        if not iteration_results:
            return {}
        
        # Extract metrics
        mape_values = [r.get('mape', 0) for r in iteration_results]
        mae_values = [r.get('mae', 0) for r in iteration_results]
        rmse_values = [r.get('rmse', 0) for r in iteration_results]
        improvements = [r.get('improvement', 0) for r in iteration_results[1:]]  # Skip baseline
        
        # Find best result
        best_result = min(iteration_results, key=lambda x: x.get('mape', float('inf')))
        
        # Calculate statistics
        evaluation = {
            'total_iterations': len(iteration_results),
            'best_iteration': best_result.get('iteration', 0),
            'best_mape': best_result.get('mape', 0),
            'best_mae': best_result.get('mae', 0),
            'best_rmse': best_result.get('rmse', 0),
            'best_improvement': best_result.get('improvement', 0),
            'metrics_summary': {
                'mape': {
                    'mean': np.mean(mape_values),
                    'std': np.std(mape_values),
                    'min': np.min(mape_values),
                    'max': np.max(mape_values),
                    'median': np.median(mape_values)
                },
                'mae': {
                    'mean': np.mean(mae_values),
                    'std': np.std(mae_values),
                    'min': np.min(mae_values),
                    'max': np.max(mae_values),
                    'median': np.median(mae_values)
                },
                'rmse': {
                    'mean': np.mean(rmse_values),
                    'std': np.std(rmse_values),
                    'min': np.min(rmse_values),
                    'max': np.max(rmse_values),
                    'median': np.median(rmse_values)
                }
            },
            'improvement_summary': {
                'total_improvements': len([imp for imp in improvements if imp > 0]),
                'total_degradations': len([imp for imp in improvements if imp < 0]),
                'mean_improvement': np.mean(improvements) if improvements else 0,
                'max_improvement': np.max(improvements) if improvements else 0,
                'min_improvement': np.min(improvements) if improvements else 0
            },
            'best_result': best_result
        }
        
        return evaluation
    
    def compare_baseline_vs_enhanced(self, baseline_results: Dict[str, Any], 
                                   enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare baseline and enhanced model results."""
        baseline_mape = baseline_results.get('mape', 0)
        enhanced_mape = enhanced_results.get('mape', 0)
        
        absolute_improvement = baseline_mape - enhanced_mape
        relative_improvement = (absolute_improvement / baseline_mape) * 100 if baseline_mape > 0 else 0
        
        comparison = {
            'baseline': {
                'mape': baseline_mape,
                'mae': baseline_results.get('mae', 0),
                'rmse': baseline_results.get('rmse', 0),
                'feature_count': baseline_results.get('feature_count', 0)
            },
            'enhanced': {
                'mape': enhanced_mape,
                'mae': enhanced_results.get('mae', 0),
                'rmse': enhanced_results.get('rmse', 0),
                'feature_count': enhanced_results.get('feature_count', 0)
            },
            'improvement': {
                'absolute_mape_improvement': absolute_improvement,
                'relative_mape_improvement': relative_improvement,
                'absolute_mae_improvement': baseline_results.get('mae', 0) - enhanced_results.get('mae', 0),
                'absolute_rmse_improvement': baseline_results.get('rmse', 0) - enhanced_results.get('rmse', 0)
            },
            'feature_analysis': {
                'baseline_significant_features': len(baseline_results.get('significant_features', [])),
                'enhanced_significant_features': len(enhanced_results.get('significant_features', [])),
                'baseline_highly_significant_features': len(baseline_results.get('highly_significant_features', [])),
                'enhanced_highly_significant_features': len(enhanced_results.get('highly_significant_features', []))
            }
        }
        
        return comparison


class ReportGenerator:
    """Generate comprehensive reports and summaries."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def generate_iteration_summary(self, ticker: str, iteration_results: List[Dict[str, Any]], 
                                 output_file: Optional[str] = None) -> str:
        """Generate a summary report for iteration results."""
        if output_file is None:
            output_file = f"{self.config.cache_dir}/{ticker}_iterative_summary.txt"
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("ITERATIVE AGENT-BASED FEATURE SELECTION SUMMARY")
        report_lines.append("=" * 60)
        report_lines.append(f"Stock: {ticker}")
        report_lines.append(f"Date: {timestamp}")
        report_lines.append(f"Total Iterations: {len(iteration_results) - 1}")
        report_lines.append("")
        
        # Performance trend
        report_lines.append("PERFORMANCE TREND:")
        report_lines.append("-" * 40)
        for result in iteration_results:
            improvement_str = f"{result['improvement']:+.2f}%" if result['improvement'] != 0 else "Baseline"
            report_lines.append(f"Iteration {result['iteration']}: {result['model_name']} - MAPE: {result['mape']:.2f}% ({improvement_str})")
        
        # Find best result
        best_result = min(iteration_results, key=lambda x: x.get('mape', float('inf')))
        report_lines.append(f"\nBest Model: {best_result['model_name']} - MAPE: {best_result['mape']:.2f}%")
        
        if 'final_test_mape' in best_result:
            report_lines.append(f"Final Test MAPE: {best_result['final_test_mape']:.2f}%")
            if 'baseline_final_mape' in best_result:
                report_lines.append(f"Final Improvement: {best_result.get('final_improvement', 0):.2f}%")
        
        # Feature engineering codes
        report_lines.append("\n" + "=" * 60)
        report_lines.append("FEATURE ENGINEERING CODES")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        for result in iteration_results[1:]:  # Skip baseline
            if result.get('claude_code'):
                report_lines.append(f"ITERATION {result['iteration']}:")
                report_lines.append(f"Performance: MAPE = {result['mape']:.2f}%")
                report_lines.append(f"Improvement: {result['improvement']:+.2f}%")
                report_lines.append(f"Features: {result['feature_count']}")
                report_lines.append("-" * 40)
                report_lines.append(result['claude_code'])
                report_lines.append("=" * 60)
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save to file
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding="utf-8") as f:
            f.write(report_content)
        
        return report_content
    
    def generate_performance_report(self, results: List[Dict[str, Any]], 
                                  successful_tickers: List[str], 
                                  failed_tickers: List[str]) -> Dict[str, Any]:
        """Generate comprehensive performance report for universal feature engineering validation."""
        if not results:
            print("❌ No successful results to report")
            return {}
        
        print(f"\n{'='*80}")
        print("UNIVERSAL FEATURE ENGINEERING PERFORMANCE REPORT")
        print(f"{'='*80}")
        
        # Calculate statistics
        mape_improvements = [r['mape_improvement'] for r in results]
        relative_improvements = [r['relative_mape_improvement'] for r in results]
        baseline_mapes = [r['baseline_mape'] for r in results]
        enhanced_mapes = [r['enhanced_mape'] for r in results]
        
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Total tickers tested: {len(successful_tickers) + len(failed_tickers)}")
        print(f"  Successful tests: {len(successful_tickers)}")
        print(f"  Failed tests: {len(failed_tickers)}")
        if failed_tickers:
            print(f"  Failed tickers: {', '.join(failed_tickers)}")
        
        print(f"\n🎯 MAPE IMPROVEMENT STATISTICS:")
        print(f"  Average MAPE improvement: {np.mean(mape_improvements):.2f}%")
        print(f"  Median MAPE improvement: {np.median(mape_improvements):.2f}%")
        print(f"  Std deviation: {np.std(mape_improvements):.2f}%")
        print(f"  Min improvement: {np.min(mape_improvements):.2f}%")
        print(f"  Max improvement: {np.max(mape_improvements):.2f}%")
        
        print(f"\n📈 RELATIVE IMPROVEMENT STATISTICS:")
        print(f"  Average relative improvement: {np.mean(relative_improvements):.1f}%")
        print(f"  Median relative improvement: {np.median(relative_improvements):.1f}%")
        print(f"  Std deviation: {np.std(relative_improvements):.1f}%")
        
        # Count improvements
        positive_improvements = sum(1 for imp in mape_improvements if imp > 0)
        significant_improvements = sum(1 for imp in mape_improvements if imp > 0.5)  # >0.5% improvement
        
        print(f"\n🏆 IMPROVEMENT DISTRIBUTION:")
        print(f"  Tickers with positive improvement: {positive_improvements}/{len(results)} ({positive_improvements/len(results)*100:.1f}%)")
        print(f"  Tickers with >0.5% improvement: {significant_improvements}/{len(results)} ({significant_improvements/len(results)*100:.1f}%)")
        
        # Detailed results table
        print(f"\n📋 DETAILED RESULTS:")
        print("-" * 100)
        print(f"{'Ticker':<8} {'Baseline MAPE':<12} {'Enhanced MAPE':<13} {'Improvement':<12} {'Rel. Imp.':<10} {'Features':<12}")
        print("-" * 100)
        
        for result in results:
            print(f"{result['ticker']:<8} {result['baseline_mape']:<12.2f} {result['enhanced_mape']:<13.2f} "
                  f"{result['mape_improvement']:<12.2f} {result['relative_mape_improvement']:<10.1f} "
                  f"{result['feature_count_baseline']}->{result['feature_count_enhanced']}")
        
        print("-" * 100)
        
        # Save detailed report
        report_filename = f'{self.config.cache_dir}/universal_feature_engineering_validation_report.txt'
        self._save_detailed_report(report_filename, results, successful_tickers, failed_tickers, 
                                 mape_improvements, relative_improvements, positive_improvements, 
                                 significant_improvements)
        
        print(f"\n💾 Detailed report saved to: {report_filename}")
        
        return {
            'summary_stats': {
                'total_tested': len(successful_tickers) + len(failed_tickers),
                'successful': len(successful_tickers),
                'failed': len(failed_tickers),
                'avg_mape_improvement': np.mean(mape_improvements),
                'median_mape_improvement': np.median(mape_improvements),
                'std_mape_improvement': np.std(mape_improvements),
                'avg_relative_improvement': np.mean(relative_improvements),
                'positive_improvements': positive_improvements,
                'significant_improvements': significant_improvements
            },
            'detailed_results': results
        }
    
    def _save_detailed_report(self, filename: str, results: List[Dict[str, Any]], 
                            successful_tickers: List[str], failed_tickers: List[str],
                            mape_improvements: List[float], relative_improvements: List[float],
                            positive_improvements: int, significant_improvements: int) -> None:
        """Save detailed report to file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("UNIVERSAL FEATURE ENGINEERING VALIDATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total tickers tested: {len(successful_tickers) + len(failed_tickers)}\n")
            f.write(f"Successful tests: {len(successful_tickers)}\n")
            f.write(f"Failed tests: {len(failed_tickers)}\n\n")
            
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"Average MAPE improvement: {np.mean(mape_improvements):.2f}%\n")
            f.write(f"Median MAPE improvement: {np.median(mape_improvements):.2f}%\n")
            f.write(f"Std deviation: {np.std(mape_improvements):.2f}%\n")
            f.write(f"Min improvement: {np.min(mape_improvements):.2f}%\n")
            f.write(f"Max improvement: {np.max(mape_improvements):.2f}%\n\n")
            
            f.write("RELATIVE IMPROVEMENT STATISTICS:\n")
            f.write(f"Average relative improvement: {np.mean(relative_improvements):.1f}%\n")
            f.write(f"Median relative improvement: {np.median(relative_improvements):.1f}%\n")
            f.write(f"Std deviation: {np.std(relative_improvements):.1f}%\n\n")
            
            f.write("IMPROVEMENT DISTRIBUTION:\n")
            f.write(f"Tickers with positive improvement: {positive_improvements}/{len(results)} ({positive_improvements/len(results)*100:.1f}%)\n")
            f.write(f"Tickers with >0.5% improvement: {significant_improvements}/{len(results)} ({significant_improvements/len(results)*100:.1f}%)\n\n")
            
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Ticker':<8} {'Baseline MAPE':<12} {'Enhanced MAPE':<13} {'Improvement':<12} {'Rel. Imp.':<10} {'Features':<12}\n")
            f.write("-" * 100 + "\n")
            
            for result in results:
                f.write(f"{result['ticker']:<8} {result['baseline_mape']:<12.2f} {result['enhanced_mape']:<13.2f} "
                       f"{result['mape_improvement']:<12.2f} {result['relative_mape_improvement']:<10.1f} "
                       f"{result['feature_count_baseline']}->{result['feature_count_enhanced']}\n")
            
            f.write("-" * 100 + "\n")


class ValidationTester:
    """Test universal feature engineering on multiple tickers."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def test_universal_feature_engineering(self, universal_function: callable, 
                                        tickers: List[str], 
                                        data_loader, model_trainer) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
        """Test the universal feature engineering code on multiple tickers and measure performance."""
        print(f"\n{'='*80}")
        print("TESTING UNIVERSAL FEATURE ENGINEERING ON MULTIPLE TICKERS")
        print(f"{'='*80}")
        print(f"Testing on {len(tickers)} tickers: {', '.join(tickers)}")
        
        results = []
        successful_tickers = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n{'='*60}")
            print(f"TESTING TICKER {i}/{len(tickers)}: {ticker}")
            print(f"{'='*60}")
            
            try:
                # Load data for this ticker
                print(f"📊 Loading data for {ticker}...")
                raw_data = data_loader.load_data_for_ticker(ticker)
                
                if raw_data is None:
                    print(f"⚠️ No data available for {ticker}, skipping...")
                    failed_tickers.append(ticker)
                    continue
                    
                X_train_raw = raw_data['X_train_raw']
                X_test_raw = raw_data['X_test_raw']
                y_train = raw_data['y_train']
                y_test = raw_data['y_test']
                prev_log_test = raw_data['prev_log_test']
                
                print(f"✅ Data loaded: Train={X_train_raw.shape}, Test={X_test_raw.shape}")
                
                # 1. Train baseline model (raw features)
                print(f"\n🎯 Training baseline model for {ticker}...")
                baseline_results = model_trainer.train_and_evaluate_model(
                    X_train_raw, X_test_raw, y_train, y_test, prev_log_test,
                    model_name=f"Baseline {ticker}", epochs=30  # Reduced epochs for faster testing
                )
                
                # 2. Apply universal feature engineering
                print(f"\n🔧 Applying universal feature engineering for {ticker}...")
                try:
                    from .feature_engineering import IterativeLLMFeatureSelector
                    from .config import LLMConfig
                except ImportError:
                    from feature_engineering import IterativeLLMFeatureSelector
                    from config import LLMConfig
                
                # Create a dummy LLM config for the feature selector
                dummy_llm_config = LLMConfig(api_key="dummy")
                feature_selector = IterativeLLMFeatureSelector(dummy_llm_config)
                
                X_train_processed, train_errors = feature_selector.apply_feature_selection_to_data(
                    X_train_raw, universal_function, max_retries=3
                )
                X_test_processed, test_errors = feature_selector.apply_feature_selection_to_data(
                    X_test_raw, universal_function, max_retries=3
                )
                
                if len(train_errors) > 0 or len(test_errors) > 0:
                    print(f"⚠️ Some errors in feature engineering: {len(train_errors)} train, {len(test_errors)} test")
                
                print(f"Feature engineering: {X_train_raw.shape} -> {X_train_processed.shape}")
                
                # 3. Train enhanced model (processed features)
                print(f"\n🚀 Training enhanced model for {ticker}...")
                enhanced_results = model_trainer.train_and_evaluate_model(
                    X_train_processed, X_test_processed, y_train, y_test, prev_log_test,
                    model_name=f"Enhanced {ticker}", epochs=30
                )
                
                # 4. Calculate improvements
                mape_improvement = baseline_results['mape'] - enhanced_results['mape']
                mae_improvement = baseline_results['mae'] - enhanced_results['mae']
                rmse_improvement = baseline_results['rmse'] - enhanced_results['rmse']
                
                relative_mape_improvement = (mape_improvement / baseline_results['mape']) * 100 if baseline_results['mape'] > 0 else 0
                
                ticker_result = {
                    'ticker': ticker,
                    'baseline_mape': baseline_results['mape'],
                    'enhanced_mape': enhanced_results['mape'],
                    'mape_improvement': mape_improvement,
                    'relative_mape_improvement': relative_mape_improvement,
                    'baseline_mae': baseline_results['mae'],
                    'enhanced_mae': enhanced_results['mae'],
                    'mae_improvement': mae_improvement,
                    'baseline_rmse': baseline_results['rmse'],
                    'enhanced_rmse': enhanced_results['rmse'],
                    'rmse_improvement': rmse_improvement,
                    'feature_count_baseline': X_train_raw.shape[2],
                    'feature_count_enhanced': X_train_processed.shape[2],
                    'train_errors': len(train_errors),
                    'test_errors': len(test_errors)
                }
                
                results.append(ticker_result)
                successful_tickers.append(ticker)
                
                print(f"\n📊 {ticker} Results:")
                print(f"  Baseline MAPE: {baseline_results['mape']:.2f}%")
                print(f"  Enhanced MAPE: {enhanced_results['mape']:.2f}%")
                print(f"  MAPE Improvement: {mape_improvement:+.2f}% ({relative_mape_improvement:+.1f}%)")
                print(f"  Features: {X_train_raw.shape[2]} -> {X_train_processed.shape[2]}")
                
            except Exception as e:
                print(f"❌ Error testing {ticker}: {e}")
                failed_tickers.append(ticker)
                continue
        
        return results, successful_tickers, failed_tickers
