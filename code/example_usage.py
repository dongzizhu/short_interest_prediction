"""
Example usage of the modular iterative agent-based feature selection system.

This script demonstrates how to use the new modular architecture for different scenarios.
"""

import os
import sys
from pathlib import Path
import pandas as pd

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
    from .config import get_development_config, get_production_config, get_quick_test_config, get_production_SVM_config, get_quick_test_SVM_config
    from .main import IterativeFeatureSelectionPipeline
except ImportError:
    from config import get_development_config, get_production_config, get_quick_test_config, get_production_SVM_config, get_quick_test_SVM_config
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


def example_svm_multi_ticker():
    """Example: Process multiple tickers with SVM model."""
    print("\n🔍 Example: Multi-Ticker Processing with SVM")
    print("="*50)
    
    # Load development configuration for faster testing
    config = get_production_SVM_config()
    # config = get_quick_test_SVM_config()
    config.llm.api_key = os.getenv('ANTHROPIC_API_KEY', 'your-api-key-here')
    
    # Configure SVM model
    config.model.model_type = 'svm'
    config.model.svm_kernel = 'rbf'  # 'rbf', 'linear', 'poly', 'sigmoid'
    config.model.svm_C = 1.0
    config.model.svm_gamma = 'scale'
    config.model.svm_epsilon = 0.1
    config.model.svm_max_iter = 1000
    
    print(f"🤖 Using SVM model with kernel: {config.model.svm_kernel}")
    print(f"   C: {config.model.svm_C}, Gamma: {config.model.svm_gamma}")
    print(f"   Epsilon: {config.model.svm_epsilon}, Max iterations: {config.model.svm_max_iter}")
    
    if config.llm.api_key == 'your-api-key-here':
        print("⚠️ Please set ANTHROPIC_API_KEY environment variable or update the code")
        return
    
    # Initialize pipeline
    pipeline = IterativeFeatureSelectionPipeline(config)
    
    # Define tickers to process (smaller set for faster testing)
    # iterative_tickers = ['SLG', 'ABM']  # Single ticker for iterative feature engineering
    # validation_tickers = ['TSLA', 'PFE']  # Tickers for validation
    # iterative_tickers = ['CYRX', 'ZEUS', 'DXLG', 'SMBK', 'FCEL']  
    # validation_tickers = ['BBW', 'UNFI', 'CMPR', 'VNDA', 'LWAY', 'MEI', 'SMBK', 'HLIT', 'INBK', 'FDUS', 'MCRI', 'GNE', 'CYRX', 'BBSI', 'INVA', 'OPK', 'OCUL', 'DXLG', 'DXPE', 'AMRK', 'AXGN', 'HCKT', 'ZEUS', 'ANIP', 'IMMR', 'CLFD', 'AEHR', 'NAT', 'EXTR', 'CHEF', 'PLYM', 'PEB', 'PLCE', 'XHR', 'FBK', 'GSBC', 'GOGO', 'SENEA', 'GNK', 'QNST', 'KRO', 'MITK', 'GSBD', 'URGN', 'AVNW', 'HTLD', 'XOMA', 'UFPT', 'FCEL', 'NVAX', 'GERN', 'CSV', 'FOR']  

    iterative_tickers = [
    "ABCB",  # Ameris Bancorp (Financials - Regional Banks)
    "EIG",   # Employers Holdings, Inc. (Financials - P&C Insurance)
    # "EYE",   # National Vision Holdings (Consumer Discretionary - Specialty Stores)
    # "AAP",   # Advance Auto Parts, Inc. (Consumer Discretionary - Automotive Retail)
    "FSS",   # Federal Signal Corporation (Industrials - Machinery & Transportation Equipment)
    "ABM",   # ABM Industries, Inc. (Industrials - Environmental Services)
    "IART",  # Integra Lifesciences Holdings (Health Care - Equipment)
    "SRPT",  # Sarepta Therapeutics (Health Care - Biotechnology)
    "EXTR",  # Extreme Networks, Inc. (IT - Communications Equipment)
    "SCSC",  # ScanSource, Inc. (IT - Technology Distributors)
    "SLG",   # SL Green Realty (Real Estate - Office REITs)
    "HL",    # Hecla Mining (Materials - Silver)
    "ANDE",  # The Andersons, Inc. (Consumer Staples - Food Distributors)
    "AROC"   # Archrock, Inc. (Energy - Oil & Gas Equipment & Services)]
    ]

    ticker_names = pd.read_csv('../data/sp600_matched_with_peers.csv')
    validation_tickers = ticker_names['Symbol'].tolist()
    
    try:
        results = pipeline.run_multi_ticker_process(iterative_tickers, validation_tickers)
        
        print(f"\n✅ SVM Multi-ticker results:")
        print(f"  Processed {len(results['ticker_results'])} tickers for iterative engineering")
        
        if results['universal_function']:
            print(f"  Generated universal feature engineering function")
        
        if results['validation_results']:
            print(f"  Validated on {len(results['validation_results'])} tickers")
        
    except Exception as e:
        print(f"❌ Error in SVM multi-ticker processing: {e}")



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
        # example_multi_ticker()  # Commented out as it takes longer
        example_svm_multi_ticker()  # Run SVM example
    
    print("\n🎉 Examples completed!")
    print("\nTo run the full pipeline:")
    print("  python code/main.py --config development --single-ticker AAPL")
    print("  python code/main.py --config production --tickers AAPL TSLA")
    print("  python code/main.py --config quick_test --max-tickers 3")


if __name__ == "__main__":
    main()
