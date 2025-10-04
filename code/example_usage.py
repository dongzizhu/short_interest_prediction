"""
Example usage of the modular iterative agent-based feature selection system.

This script demonstrates how to use the new modular architecture for different scenarios.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(str(Path(__file__).parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from the project root (parent directory of code/)
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")
    print("   Environment variables will be loaded from system environment only.")

try:
    from .config import get_development_config, get_production_config, get_quick_test_config
    from .main import IterativeFeatureSelectionPipeline
except ImportError:
    from config import get_development_config, get_production_config, get_quick_test_config
    from main import IterativeFeatureSelectionPipeline


def example_single_ticker():
    """Example: Process a single ticker with development configuration."""
    print("🔍 Example: Single Ticker Processing")
    print("="*50)
    
    # Load development configuration
    config = get_development_config()
    
    # Set API key (replace with your actual key)
    config.llm.api_key = os.getenv('ANTHROPIC_API_KEY', 'your-api-key-here')
    
    if config.llm.api_key == 'your-api-key-here':
        print("⚠️ Please set ANTHROPIC_API_KEY environment variable or update the code")
        return
    
    # Initialize pipeline
    pipeline = IterativeFeatureSelectionPipeline(config)
    
    # Process single ticker
    try:
        best_result, iteration_codes = pipeline.run_iterative_process_for_ticker('AAPL')
        
        print(f"\n✅ Results for AAPL:")
        print(f"  Best MAPE: {best_result['mape']:.2f}%")
        print(f"  Improvement: {best_result.get('improvement', 0):+.2f}%")
        print(f"  Feature count: {best_result.get('feature_count', 'Unknown')}")
        print(f"  Iterations completed: {len(iteration_codes)}")
        
    except Exception as e:
        print(f"❌ Error processing AAPL: {e}")


def example_multi_ticker():
    """Example: Process multiple tickers with production configuration."""
    print("\n🔍 Example: Multi-Ticker Processing")
    print("="*50)
    
    # Load production configuration
    config = get_production_config()
    config.llm.api_key = os.getenv('ANTHROPIC_API_KEY', 'your-api-key-here')
    
    
    if config.llm.api_key == 'your-api-key-here':
        print("⚠️ Please set ANTHROPIC_API_KEY environment variable or update the code")
        return
    
    # Initialize pipeline
    pipeline = IterativeFeatureSelectionPipeline(config)
    
    # Define tickers to process
    iterative_tickers = ['AAPL', 'TSLA']  # Tickers for iterative feature engineering
    validation_tickers = ['PFE', 'MSFT']  # Tickers for validation
    
    try:
        results = pipeline.run_multi_ticker_process(iterative_tickers, validation_tickers)
        
        print(f"\n✅ Multi-ticker results:")
        print(f"  Processed {len(results['ticker_results'])} tickers for iterative engineering")
        
        if results['universal_function']:
            print(f"  Generated universal feature engineering function")
        
        if results['validation_results']:
            print(f"  Validated on {len(results['validation_results'])} tickers")
        
    except Exception as e:
        print(f"❌ Error in multi-ticker processing: {e}")



def main():
    """Run all examples."""
    print("🚀 Modular Iterative Agent-Based Feature Selection Examples")
    print("="*70)
    
    # Check if API key is available from .env file or environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key or api_key == 'your-anthropic-api-key-here':
        print("⚠️ ANTHROPIC_API_KEY not found or not set properly")
        print("   Please set your API key in the .env file:")
        print("   ANTHROPIC_API_KEY=your-actual-api-key-here")
        print("   Or set it as an environment variable: export ANTHROPIC_API_KEY='your-key-here'")
        print("\n⚠️ Skipping examples that require API key")
    else:
        print("✅ API key loaded successfully")
        # example_single_ticker()
        example_multi_ticker()  # Commented out as it takes longer
    
    print("\n🎉 Examples completed!")
    print("\nTo run the full pipeline:")
    print("  python code/main.py --config development --single-ticker AAPL")
    print("  python code/main.py --config production --tickers AAPL TSLA")
    print("  python code/main.py --config quick_test --max-tickers 3")


if __name__ == "__main__":
    main()
