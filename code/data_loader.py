"""
Data loading and preprocessing module for the iterative agent-based feature selection system.

This module handles all data loading, preprocessing, and feature construction
from various data sources including parquet files and cached data.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, timedelta
from pathlib import Path

try:
    from .config import DataConfig
    from .utils import validate_data_shape, safe_divide, calculate_returns, calculate_volatility
except ImportError:
    from config import DataConfig
    from utils import validate_data_shape, safe_divide, calculate_returns, calculate_volatility


class DataLoader:
    """Main data loading and preprocessing class."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self._ticker_timeseries = None
        self._price_data = None
        self._extra_features = None
    
    def load_ticker_timeseries(self) -> Dict[str, Any]:
        """Load ticker timeseries data from cache."""
        if self._ticker_timeseries is None:
            try:
                with open(self.config.ticker_timeseries_path, 'rb') as f:
                    self._ticker_timeseries = pickle.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Ticker timeseries file not found: {self.config.ticker_timeseries_path}")
        return self._ticker_timeseries

    def load_extra_features(self) -> Dict[str, Any]:
        """Load extra features from cache."""
        if self._extra_features is None:
            try:
                self._extra_features = pd.read_parquet(self.config.extra_features_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Extra features file not found: {self.config.extra_features_path}")
        return self._extra_features

    def load_short_volume_data(self) -> Dict[str, Any]:
        """Load short volume data from parquet file."""
        try:
            short_volume_data = pd.read_parquet(self.config.short_volume_path)
            return short_volume_data
        except FileNotFoundError:
            raise FileNotFoundError(f"Short volume data file not found: {self.config.short_volume_path}")

    def load_price_data(self) -> pd.DataFrame:
        """Load price data from parquet file."""
        if self._price_data is None:
            try:
                self._price_data = pd.read_parquet(self.config.parquet_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Price data file not found: {self.config.parquet_path}")
        return self._price_data
    
    def get_ohlc_data(self, df: pd.DataFrame, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
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
    
    def get_short_volume_data(self, df: pd.DataFrame, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Extract short volume data for a specific ticker within a date range.
        
        Parameters:
        df (DataFrame): The DataFrame with MultiIndex columns (ticker, metric)
        ticker (str): The ticker symbol to extract data for
        start_date (str or datetime): Start date in 'YYYY-MM-DD' format
        end_date (str or datetime): End date in 'YYYY-MM-DD' format
        
        Returns:
        DataFrame: Short volume data with columns ['ShortVolume', 'TotalVolume']
        """
        try:
            # Check if ticker exists in the data
            if ('ShortVolume', ticker) not in df.columns:
                raise ValueError(f"Ticker '{ticker}' not found in short volume data.")
            
            # Extract short volume columns for the specific ticker
            short_vol_data = df.loc[start_date:end_date, [('ShortVolume', ticker), 
                                                          ('TotalVolume', ticker)]]
            
            # Rename columns to remove the ticker prefix for cleaner output
            short_vol_data.columns = ['ShortVolume', 'TotalVolume']
            
            # Remove rows with all NaN values
            short_vol_data = short_vol_data.dropna(how='all')
            
            return short_vol_data
            
        except KeyError as e:
            print(f"Error: Date range or ticker not found in data. {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
    
    def create_price_features_from_parquet(self, si_dates: pd.Index, price_data: pd.DataFrame, 
                                         gap_days: int = None) -> np.ndarray:
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
        if gap_days is None:
            gap_days = self.config.gap_days
            
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

    def create_short_volume_features(self, si_dates: pd.Index, short_vol_data: pd.DataFrame, gap_days: int) -> np.ndarray:
        """
        Create short volume features for each SI date.
        Features: [day1_short_volume, day1_total_volume, ..., dayN_short_volume, dayN_total_volume]
        Total: 2 * gap_days features

        Parameters:
        si_dates: pandas Index of SI reporting dates
        short_vol_data: DataFrame with short volume data (columns: ShortVolume, TotalVolume)

        Returns:
        numpy array of shape (len(si_dates), 2 * gap_days)
        """
        if short_vol_data.empty:
            print("No short volume data available, creating zero features")
            return np.zeros((len(si_dates), 2 * gap_days))

        short_volume_features = []

        for si_date in si_dates:
            si_datetime = pd.to_datetime(si_date)

            # Define the window (gap_days before the SI date)
            window_end = si_datetime - timedelta(days=1)  # Day before SI date

            features = []

            # Get short volume data for each of the gap_days
            for day_offset in range(gap_days):
                target_date = window_end - timedelta(days=day_offset)

                # Find the closest trading day (in case of weekends/holidays)
                closest_date = None
                min_diff = timedelta(days=10)  # Max search window

                for vol_date in short_vol_data.index:
                    diff = abs(vol_date.date() - target_date.date())
                    if diff < min_diff:
                        min_diff = diff
                        closest_date = vol_date

                if closest_date is not None and min_diff <= timedelta(days=3):  # Within 3 days
                    day_data = short_vol_data.loc[closest_date]
                    features.extend([
                        day_data['ShortVolume'],
                        day_data['TotalVolume']
                    ])
                else:
                    # No data available, use NaNs
                    features.extend([np.nan, np.nan])

            short_volume_features.append(features)

        short_volume_features = np.array(short_volume_features)

        # Handle NaNs by imputing with column means
        col_means = np.nanmean(short_volume_features, axis=0)
        short_volume_features = np.where(np.isnan(short_volume_features), col_means, short_volume_features)

        return short_volume_features

    def load_data_for_ticker(self, stock: str, ifFake: bool = False) -> Dict[str, Any]:
        """
        Load and preprocess data for a specific ticker.
        
        Parameters:
        stock (str): Stock ticker symbol
        
        Returns:
        dict: Dictionary containing all the data needed for training
        """
        print(f"📊 Loading data for {stock} from parquet file...")
        
        # Generate fake data for demonstration/testing
        if ifFake:
            print("Using fake data for testing purposes")
            np.random.seed(42)
            X_train_raw = np.random.randn(106, 4, 97)
            X_val_raw = np.random.randn(36, 4, 97)
            X_test_raw = np.random.randn(36, 4, 97)
            y_train = np.random.randn(106, 1)
            y_val = np.random.randn(36, 1)
            y_test = np.random.randn(36, 1)
            prev_log_train = np.random.randn(106)
            prev_log_val = np.random.randn(36)
            prev_log_test = np.random.randn(36)
            si_dates = pd.date_range(start='2020-01-01', periods=106+4, freq='D')[:106]
            SI_series = np.random.rand(106, 1)
            vol_series = np.random.rand(106, 1)
            price_data = pd.DataFrame(np.random.rand(106, 4), index=si_dates, columns=['open', 'high', 'low', 'close'])
        else:
            # Load ticker timeseries data (SI and Volume)
            ticker_timeseries = self.load_ticker_timeseries()
            ticker_extra_features = self.load_extra_features()
            ticker_short_volume = self.load_short_volume_data()
            
            # Load parquet file with price data
            print(f"Loading parquet file: {self.config.parquet_path}")
            df = self.load_price_data()
            print(f"Parquet file loaded. Shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Get SI dates and data
            si_dates = ticker_timeseries[stock]['SI'].dropna().index
            SI_series = ticker_timeseries[stock]['SI'].dropna().values.astype(np.float64).reshape(-1, 1)
            vol_series = ticker_timeseries[stock]['Volume'].dropna().values.astype(np.float64).reshape(-1, 1)
            
            print(f"SI dates range: {si_dates.min()} to {si_dates.max()}")
            print(f"Number of SI observations: {len(si_dates)}")

            # Get extra features for the stock
            options_put_call_volume_ratio = ticker_extra_features[stock]['options_put_call_volume_ratio'].reindex(si_dates).fillna(method='ffill').fillna(0).values.reshape(-1, 1)
            options_synthetic_short_cost = ticker_extra_features[stock]['options_synthetic_short_cost'].reindex(si_dates).fillna(method='ffill').fillna(0).values.reshape(-1, 1)
            options_avg_implied_volatility = ticker_extra_features[stock]['options_avg_implied_volatility'].reindex(si_dates).fillna(method='ffill').fillna(0).values.reshape(-1, 1)
            shares_outstanding = ticker_extra_features[stock]['shares_outstanding'].reindex(si_dates).fillna(method='ffill').fillna(0).values.reshape(-1, 1)
            # volume = ticker_extra_features[stock]['volume'].reindex(si_dates).fillna(method='ffill').fillna(0).values.reshape(-1, 1)
            days_to_cover = ticker_extra_features[stock]['daysToCoverQuantity'].reindex(si_dates).fillna(method='ffill').fillna(0).values.reshape(-1, 1)

            # Get OHLC data for the stock
            print(f"Extracting OHLC data for {stock}...")
            start_date = si_dates.min() - timedelta(days=30)  # Add buffer for lookback
            end_date = si_dates.max() + timedelta(days=5)     # Add buffer for future dates
            
            price_data = self.get_ohlc_data(df, stock, start_date, end_date)
            print(f"Retrieved price data for {len(price_data)} trading days")
            if not price_data.empty:
                print(f"Price data date range: {price_data.index.min()} to {price_data.index.max()}")
            
            # Create price features (15 days of OHLC = 60 features)
            print(f"Creating price features with {self.config.gap_days} days lookback...")
            price_features = self.create_price_features_from_parquet(si_dates, price_data, self.config.gap_days)
            print(f"Price features shape: {price_features.shape}")

            short_volume_data = self.get_short_volume_data(ticker_short_volume, stock, start_date, end_date)
            print(f"Retrieved short volume data for {len(short_volume_data)} days")
            if not short_volume_data.empty:
                print(f"Short volume data date range: {short_volume_data.index.min()} to {short_volume_data.index.max()}")
            
            print(f"Creating short volume features with {self.config.gap_days} days lookback...")
            short_volume_features = self.create_short_volume_features(si_dates, short_volume_data, self.config.gap_days)
            print(f"Short volume features shape: {short_volume_features.shape}")
            
            # Combine all features: [SI, Volume, 60 price features]
            level_series = np.concatenate([SI_series, vol_series, days_to_cover, price_features, 
                                           options_put_call_volume_ratio, options_synthetic_short_cost, options_avg_implied_volatility, 
                                           shares_outstanding, short_volume_features], axis=1)  # (T, 97)
            print(f"Combined features shape without short volume: {level_series.shape}")
            
            # Create log-return targets
            eps = self.config.eps  # to avoid log(0)
            series_safe = np.where(SI_series <= 0, eps, SI_series).reshape(-1)
            y_log = np.log(series_safe)
            
            # Build supervised windows with log-return target
            X_raw, y_logret, prev_log_all = self._make_windows_level_to_logret(
                level_series, y_log, self.config.lookback_window
            )
            
            # Split data into train/val/test to prevent data leakage
            X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, prev_log_train, prev_log_val, prev_log_test = self._split_data(
                X_raw, y_logret, prev_log_all
            )
            
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
            'lookback_window': self.config.lookback_window,
            'gap_days': self.config.gap_days,
            'price_data_shape': price_data.shape if not price_data.empty else (0, 0)
        }

        # save data_to_save for testing
        # save_path = Path(f"processed_data_{stock}.pkl")
        # pickle.dump(data_to_save, open(save_path, 'wb'))
        # print(f"Processed data saved to {save_path}")
        # exit(0)

        return data_to_save
    
    def _make_windows_level_to_logret(self, level_series: np.ndarray, y_log: np.ndarray, 
                                    lookback: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create supervised windows with log-return targets."""
        X_list, y_logret_list, prev_log_list = [], [], []
        for t in range(lookback, len(level_series)):
            X_list.append(level_series[t - lookback:t, :])                  # (L, 97)
            y_logret_list.append([y_log[t] - y_log[t - 1]])                 # (1,)
            prev_log_list.append(y_log[t - 1])   
        X = np.asarray(X_list)                          # (N, L, 97)
        y_logret = np.asarray(y_logret_list)            # (N, 1)
        prev_log = np.asarray(prev_log_list)            # (N,)
        return X, y_logret, prev_log
    
    def _split_data(self, X_raw: np.ndarray, y_logret: np.ndarray, 
                   prev_log_all: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Split data into train/validation/test sets."""
        N = X_raw.shape[0]
        train_split = int(self.config.train_split * N)  # 60% for training
        val_split = int(self.config.val_split * N)      # 20% for validation (60-80%)
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
        
        return X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, prev_log_train, prev_log_val, prev_log_test
    
    def get_available_tickers(self, max_tickers: Optional[int] = None) -> List[str]:
        """Get list of available tickers from the parquet file."""
        try:
            df = self.load_price_data()
            all_tickers = list(set([x[0] for x in df.columns]))
            
            if max_tickers:
                return all_tickers[:max_tickers]
            return all_tickers
            
        except Exception as e:
            print(f"Error loading tickers from parquet: {e}")
            # Fallback to default tickers
            return ['TSLA', 'PFE', 'AAPL']
    
    def validate_data_integrity(self, data: Dict[str, Any]) -> bool:
        """Validate that loaded data has correct structure and no obvious issues."""
        try:
            # Check required keys
            required_keys = ['X_train_raw', 'X_val_raw', 'X_test_raw', 'y_train', 'y_val', 'y_test']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required data key: {key}")
            
            # Check shapes
            X_train = data['X_train_raw']
            X_val = data['X_val_raw']
            X_test = data['X_test_raw']
            
            if X_train.shape[2] != self.config.total_features:
                raise ValueError(f"Expected {self.config.total_features} features, got {X_train.shape[2]}")
            
            if X_train.shape[1] != self.config.lookback_window:
                raise ValueError(f"Expected lookback window {self.config.lookback_window}, got {X_train.shape[1]}")
            
            # Check for NaN values
            for key in ['X_train_raw', 'X_val_raw', 'X_test_raw']:
                if np.isnan(data[key]).any():
                    print(f"Warning: NaN values found in {key}")
            
            # Check target variables
            y_train = data['y_train']
            y_val = data['y_val']
            y_test = data['y_test']
            
            if np.isnan(y_train).any() or np.isnan(y_val).any() or np.isnan(y_test).any():
                print("Warning: NaN values found in target variables")
            
            print("✅ Data integrity validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Data integrity validation failed: {e}")
            return False
    
    def save_processed_data(self, data: Dict[str, Any], filepath: str) -> None:
        """Save processed data to file."""
        from .utils import save_results
        save_results(data, filepath)
    
    def load_processed_data(self, filepath: str) -> Dict[str, Any]:
        """Load processed data from file."""
        from .utils import load_results
        return load_results(filepath)


class DataPreprocessor:
    """Data preprocessing utilities."""
    
    @staticmethod
    def normalize_features(X: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Any]:
        """Normalize features using specified method."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Reshape for scaling
        original_shape = X.shape
        X_reshaped = X.reshape(-1, X.shape[-1])
        
        # Fit and transform
        X_scaled = scaler.fit_transform(X_reshaped)
        
        # Reshape back
        X_scaled = X_scaled.reshape(original_shape)
        
        return X_scaled, scaler
    
    @staticmethod
    def handle_missing_values(X: np.ndarray, method: str = 'mean') -> np.ndarray:
        """Handle missing values in the data."""
        if method == 'mean':
            # Fill with column means
            for i in range(X.shape[-1]):
                col_data = X[:, :, i]
                mean_val = np.nanmean(col_data)
                X[:, :, i] = np.where(np.isnan(col_data), mean_val, col_data)
        elif method == 'forward_fill':
            # Forward fill missing values
            for i in range(X.shape[0]):  # For each sample
                for j in range(X.shape[2]):  # For each feature
                    series = X[i, :, j]
                    mask = ~np.isnan(series)
                    if mask.any():
                        X[i, :, j] = np.interp(np.arange(len(series)), 
                                             np.arange(len(series))[mask], 
                                             series[mask])
        elif method == 'zero':
            # Fill with zeros
            X = np.where(np.isnan(X), 0, X)
        
        return X
    
    @staticmethod
    def create_technical_features(ohlc_data: np.ndarray) -> np.ndarray:
        """Create technical indicators from OHLC data."""
        from .utils import calculate_technical_indicators
        
        features = []
        
        for i in range(ohlc_data.shape[0]):  # For each timestep
            timestep_features = []
            
            # Basic OHLC features
            timestep_features.extend(ohlc_data[i])  # [open, high, low, close]
            
            # Technical indicators
            indicators = calculate_technical_indicators(ohlc_data[i:i+1])
            
            for indicator_name, indicator_values in indicators.items():
                if isinstance(indicator_values, np.ndarray):
                    timestep_features.append(indicator_values[0])
                else:
                    timestep_features.append(indicator_values)
            
            features.append(timestep_features)
        
        return np.array(features)
