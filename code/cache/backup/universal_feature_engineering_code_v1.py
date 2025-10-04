"""
Universal Feature Engineering Code for Short Interest Prediction
Generated from best practices across multiple tickers
Generated on: 2025-09-26 02:06:57
Source tickers: YORW, PBHC, HOFT, CULP, KIRK
"""

import numpy as np

def construct_features(data):
    """
    Universal feature engineering function for short interest prediction.
    
    Synthesizes best practices from multiple ticker-specific models to create
    a robust, generalizable solution for any stock.
    
    Args:
        data: numpy array of shape (lookback_window, 62)
            - Feature 0: Short interest
            - Feature 1: Average daily volume
            - Features 2-61: OHLC prices for past 15 days (4 × 15 = 60 dimensions)
    
    Returns:
        numpy array of shape (lookback_window, constructed_features)
    """
    # Validate input shape and handle edge cases
    if len(data.shape) != 2 or data.shape[1] != 62:
        return np.zeros((0, 50))  # Return empty array with expected feature count
    
    lookback_window = data.shape[0]
    if lookback_window == 0:
        return np.zeros((0, 50))
    
    # Handle NaN values in input data
    data = np.nan_to_num(data, nan=0.0)
    
    # Initialize output array
    engineered_features = []
    
    # Process each timestamp independently
    for t in range(lookback_window):
        features_t = []
        
        # Extract key components
        short_interest = data[t, 0]  # Feature_0 (highest importance across all tickers)
        avg_volume = data[t, 1]      # Feature_1 (second highest importance)
        
        # Extract OHLC data and reshape for easier manipulation
        ohlc_data = data[t, 2:62].reshape(15, 4)
        open_prices = ohlc_data[:, 0]
        high_prices = ohlc_data[:, 1]
        low_prices = ohlc_data[:, 2]
        close_prices = ohlc_data[:, 3]
        
        # ---- 1. CORE ORIGINAL FEATURES ----
        # Analysis across all tickers shows original features are critical
        # Keep the most important original features intact
        
        # 1.1 Short interest (Feature_0) - highest importance in all tickers
        features_t.append(short_interest)
        
        # 1.2 Average volume (Feature_1) - consistently important
        features_t.append(avg_volume)
        
        # 1.3 Days to cover - key metric for short squeeze potential
        # Consistently important across YORW, PBHC, HOFT, CULP
        days_to_cover = short_interest / (avg_volume + 1e-8)
        features_t.append(days_to_cover)
        
        # 1.4 Most recent price data (last day's OHLC)
        # Important across multiple tickers (KIRK, YORW, CULP)
        if len(close_prices) > 0:
            features_t.extend([
                open_prices[0],   # Most recent open
                high_prices[0],   # Most recent high
                low_prices[0],    # Most recent low
                close_prices[0]   # Most recent close
            ])
        else:
            features_t.extend([0, 0, 0, 0])
        
        # ---- 2. SHORT INTEREST DYNAMICS ----
        # Short interest change metrics were important in all tickers
        
        # 2.1 Short interest change (absolute and percentage)
        if t > 0:
            prev_si = data[t-1, 0]
            si_change = short_interest - prev_si
            si_pct_change = si_change / (prev_si + 1e-8)
        else:
            si_change = 0
            si_pct_change = 0
        features_t.extend([si_change, si_pct_change])
        
        # 2.2 Short interest acceleration (second derivative)
        # Important in PBHC, CULP, KIRK
        if t > 1:
            prev_si_change = data[t-1, 0] - data[t-2, 0]
            si_accel = si_change - prev_si_change
            si_accel_norm = si_accel / (abs(prev_si_change) + 1e-8)
        else:
            si_accel = 0
            si_accel_norm = 0
        features_t.extend([si_accel, si_accel_norm])
        
        # 2.3 Short interest relative to recent high
        # Important in PBHC for detecting potential reversals
        if t > 0:
            max_si = np.max(data[max(0, t-5):t+1, 0])
            si_rel_to_high = short_interest / (max_si + 1e-8)
        else:
            si_rel_to_high = 1.0
        features_t.append(si_rel_to_high)
        
        # 2.4 Lagged short interest (t-1)
        # Important for temporal patterns in PBHC, KIRK
        if t > 0:
            lagged_si = data[t-1, 0]
        else:
            lagged_si = short_interest
        features_t.append(lagged_si)
        
        # ---- 3. VOLUME DYNAMICS ----
        # Volume patterns important across all tickers
        
        # 3.1 Volume change (absolute and percentage)
        if t > 0:
            prev_volume = data[t-1, 1]
            vol_change = avg_volume - prev_volume
            vol_pct_change = vol_change / (prev_volume + 1e-8)
        else:
            vol_change = 0
            vol_pct_change = 0
        features_t.extend([vol_change, vol_pct_change])
        
        # 3.2 Volume volatility
        # Important in CULP, KIRK
        if t >= 3:
            recent_vol = data[max(0, t-5):t+1, 1]
            vol_volatility = np.std(recent_vol) / (np.mean(recent_vol) + 1e-8)
        else:
            vol_volatility = 0
        features_t.append(vol_volatility)
        
        # ---- 4. PRICE DYNAMICS ----
        # Price trends and patterns important across all tickers
        
        # 4.1 Price momentum at different timeframes
        # Important in all tickers
        for days in [1, 3, 5]:
            if len(close_prices) > days:
                momentum = close_prices[0] / close_prices[days] - 1
            else:
                momentum = 0
            features_t.append(momentum)
        
        # 4.2 Price volatility
        # Important in YORW, PBHC, HOFT, CULP, KIRK
        if len(close_prices) >= 5:
            price_std = np.std(close_prices[:5])
            price_mean = np.mean(close_prices[:5])
            price_volatility = price_std / (price_mean + 1e-8)
        else:
            price_volatility = 0
        features_t.append(price_volatility)
        
        # 4.3 Price range volatility
        # Important in CULP, KIRK
        if len(high_prices) >= 5 and len(low_prices) >= 5:
            price_ranges = (high_prices[:5] - low_prices[:5]) / (open_prices[:5] + 1e-8)
            avg_range = np.mean(price_ranges)
        else:
            avg_range = 0
        features_t.append(avg_range)
        
        # ---- 5. TECHNICAL INDICATORS ----
        # Select technical indicators that performed well across tickers
        
        # 5.1 Moving Averages
        # Important in CULP, KIRK
        ma5 = np.mean(close_prices[:5]) if len(close_prices) >= 5 else (close_prices[0] if len(close_prices) > 0 else 0)
        ma10 = np.mean(close_prices[:10]) if len(close_prices) >= 10 else ma5
        
        # Price relative to moving averages
        if len(close_prices) > 0:
            price_to_ma5 = close_prices[0] / (ma5 + 1e-8) - 1
            price_to_ma10 = close_prices[0] / (ma10 + 1e-8) - 1
        else:
            price_to_ma5 = 0
            price_to_ma10 = 0
        features_t.extend([price_to_ma5, price_to_ma10])
        
        # 5.2 Bollinger Bands
        # Important in YORW, CULP, KIRK
        if len(close_prices) >= 5:
            std5 = np.std(close_prices[:5])
            upper_band = ma5 + (2 * std5)
            lower_band = ma5 - (2 * std5)
            
            # Bollinger Band width
            bb_width = (upper_band - lower_band) / (ma5 + 1e-8)
            
            # Position within Bollinger Bands
            if upper_band > lower_band and len(close_prices) > 0:
                bb_position = (close_prices[0] - lower_band) / (upper_band - lower_band + 1e-8)
            else:
                bb_position = 0.5
        else:
            bb_width = 0
            bb_position = 0.5
        features_t.extend([bb_width, bb_position])
        
        # 5.3 RSI (Relative Strength Index)
        # Important in PBHC, CULP, KIRK
        if len(close_prices) >= 5:
            diff = np.diff(np.concatenate([[close_prices[min(4, len(close_prices)-1)]], close_prices[:4]]))
            gains = np.sum(np.clip(diff, 0, None))
            losses = np.sum(np.abs(np.clip(diff, None, 0)))
            
            if losses > 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100 if gains > 0 else 50
            
            # Normalize RSI to [-1, 1] range
            rsi_norm = (rsi - 50) / 50
        else:
            rsi = 50
            rsi_norm = 0
        features_t.extend([rsi / 100, rsi_norm])  # Normalize to [0,1] and [-1,1]
        
        # ---- 6. RELATIONSHIP METRICS ----
        # Relationships between different metrics important in all tickers
        
        # 6.1 Short interest to price ratio
        # Important in YORW, PBHC, CULP
        if len(close_prices) > 0:
            si_price_ratio = short_interest / (close_prices[0] + 1e-8)
        else:
            si_price_ratio = 0
        features_t.append(si_price_ratio)
        
        # 6.2 Short interest relative to volatility
        # Important in CULP, KIRK
        si_vol_interaction = short_interest * price_volatility
        features_t.append(si_vol_interaction)
        
        # 6.3 Short interest change relative to price momentum
        # Important in CULP, KIRK
        if t > 0 and len(close_prices) >= 5:
            si_momentum_interaction = si_pct_change * features_t[-10]  # Interaction with 5-day momentum
        else:
            si_momentum_interaction = 0
        features_t.append(si_momentum_interaction)
        
        # ---- 7. PATTERN RECOGNITION ----
        # Pattern detection important in CULP, KIRK
        
        # 7.1 Potential short squeeze indicator
        # High short interest + price increase + volume increase
        if len(close_prices) >= 5 and t > 0:
            price_up = close_prices[0] > close_prices[4]
            si_high = days_to_cover > 5  # Arbitrary threshold
            vol_increase = avg_volume > data[t-1, 1] * 1.1  # 10% volume increase
            
            short_squeeze_indicator = 1.0 if (price_up and si_high and vol_increase) else 0.0
        else:
            short_squeeze_indicator = 0.0
        features_t.append(short_squeeze_indicator)
        
        # 7.2 Price-SI divergence indicator
        # Price up but short interest also up, or price down but short interest down
        if t > 0 and len(close_prices) > 1:
            price_up = close_prices[0] > close_prices[1]
            si_up = short_interest > data[t-1, 0]
            
            divergence_indicator = 1.0 if (price_up and si_up) or (not price_up and not si_up) else 0.0
        else:
            divergence_indicator = 0.0
        features_t.append(divergence_indicator)
        
        # ---- 8. NORMALIZED FEATURES ----
        # Z-scores and other normalizations important in PBHC, KIRK
        
        # 8.1 Z-score of short interest
        if t >= 3:
            recent_si = data[max(0, t-5):t+1, 0]
            si_mean = np.mean(recent_si)
            si_std = np.std(recent_si)
            si_zscore = (short_interest - si_mean) / (si_std + 1e-8)
        else:
            si_zscore = 0
        features_t.append(si_zscore)
        
        # 8.2 Z-score of volume
        if t >= 3:
            recent_vol = data[max(0, t-5):t+1, 1]
            vol_mean = np.mean(recent_vol)
            vol_std = np.std(recent_vol)
            vol_zscore = (avg_volume - vol_mean) / (vol_std + 1e-8)
        else:
            vol_zscore = 0
        features_t.append(vol_zscore)
        
        # ---- 9. TREND FEATURES ----
        # Linear trends important in YORW, KIRK
        
        # 9.1 Price trend (linear regression slope)
        if len(close_prices) >= 5:
            x = np.arange(5)
            y = close_prices[:5]
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            
            numerator = np.sum((x - mean_x) * (y - mean_y))
            denominator = np.sum((x - mean_x) ** 2)
            
            slope = numerator / (denominator + 1e-8)
            # Normalize by average price
            norm_slope = slope / (mean_y + 1e-8)
        else:
            norm_slope = 0
        features_t.append(norm_slope)
        
        # 9.2 Volume trend
        if t >= 4:
            recent_volumes = np.array([data[t-i, 1] for i in range(5)])
            x = np.arange(5)
            mean_x = np.mean(x)
            mean_vol = np.mean(recent_volumes)
            
            numerator = np.sum((x - mean_x) * (recent_volumes - mean_vol))
            denominator = np.sum((x - mean_x) ** 2)
            
            vol_slope = numerator / (denominator + 1e-8)
            # Normalize by average volume
            norm_vol_slope = vol_slope / (mean_vol + 1e-8)
        else:
            norm_vol_slope = 0
        features_t.append(norm_vol_slope)
        
        # ---- 10. TRANSFORMATIONS OF KEY FEATURES ----
        # Non-linear transformations important in HOFT, KIRK
        
        # 10.1 Log transform of short interest
        log_si = np.log1p(short_interest)  # log(1+x) to handle zeros
        features_t.append(log_si)
        
        # 10.2 Square root transform of volume
        sqrt_volume = np.sqrt(avg_volume)
        features_t.append(sqrt_volume)
        
        # 10.3 Square of short interest (polynomial feature)
        # Important in KIRK
        features_t.append(short_interest ** 2)
        
        # ---- 11. INTERACTION TERMS ----
        # Interactions between important features
        
        # 11.1 Short interest × Volume interaction
        si_vol_interaction = short_interest * avg_volume
        features_t.append(si_vol_interaction)
        
        # 11.2 Volume × Volatility interaction
        vol_volatility_interaction = avg_volume * price_volatility
        features_t.append(vol_volatility_interaction)
        
        # 11.3 Short interest × Price interaction
        if len(close_prices) > 0:
            si_price_interaction = short_interest * close_prices[0]
        else:
            si_price_interaction = 0
        features_t.append(si_price_interaction)
        
        # Add features for current timestamp to output
        engineered_features.append(features_t)
    
    # Convert to numpy array
    result = np.array(engineered_features)
    
    # Handle any NaN or inf values
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    
    return result