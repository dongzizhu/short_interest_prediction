"""
Universal Feature Engineering Code for Short Interest Prediction
Generated from best practices across multiple tickers
Generated on: 2025-10-09 02:36:43
Source tickers: ABCB, EIG, FSS, ABM, IART, SRPT, EXTR, SCSC, SLG, HL, ANDE, AROC
"""

import numpy as np

def construct_features(data):
    """
    Universal feature construction function for short interest prediction.
    
    Args:
        data: numpy array of shape (lookback_window, 97)
        
    Returns:
        numpy array of engineered features
    """
    # Determine if we're using 15 or 25 features based on input data characteristics
    lookback_window = data.shape[0]
    
    # Analyze the data to determine optimal feature count
    # Higher volatility stocks benefit from more features
    if lookback_window > 1:
        price_volatility = np.std(data[:, 3:63].reshape(-1, 15, 4)[:, :, 3]) / np.mean(data[:, 3:63].reshape(-1, 15, 4)[:, :, 3])
        si_volatility = np.std(data[:, 0]) / max(np.mean(data[:, 0]), 1e-8)
        combined_volatility = (price_volatility + si_volatility) / 2
        MAX_TOTAL = 25 if combined_volatility > 0.05 else 15
    else:
        MAX_TOTAL = 25  # Default to higher feature count when we can't determine volatility
    
    features = np.zeros((lookback_window, MAX_TOTAL), dtype=np.float32)
    
    for t in range(lookback_window):
        # Initialize lists for raw and engineered features
        raw_keep = []
        eng = []
        
        # Extract key raw features
        short_interest = data[t, 0]
        avg_volume = data[t, 1]
        days_to_cover = data[t, 2]
        
        # Extract OHLC data for the past 15 days
        ohlc = data[t, 3:63].reshape(15, 4)
        open_prices, high_prices, low_prices, close_prices = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        
        # Extract options data
        put_call_ratio = data[t, 64]
        synthetic_short_cost = data[t, 65]
        implied_volatility = data[t, 66]
        shares_outstanding = data[t, 67]
        
        # Extract short volume and total volume data
        short_volume = data[t, 68:83]
        total_volume = data[t, 83:98]
        
        # Keep critical raw features (consistent across all best models)
        raw_keep.append(short_interest)  # Short interest (always keep)
        raw_keep.append(avg_volume)      # Average daily volume (always keep)
        raw_keep.append(days_to_cover)   # Days to cover (high importance)
        raw_keep.append(close_prices[-1])  # Most recent close price
        raw_keep.append(put_call_ratio)  # Options put/call ratio
        raw_keep.append(synthetic_short_cost)  # Cost of shorting
        raw_keep.append(implied_volatility)  # Implied volatility
        
        # Calculate MAX_NEW based on raw features kept
        MAX_NEW = MAX_TOTAL - len(raw_keep)
        
        # 1. Short Interest to Shares Outstanding Ratio
        # Measures what percentage of available shares are being shorted
        si_to_shares = short_interest / max(shares_outstanding, 1e-8)
        eng.append(si_to_shares)
        
        # 2. Short Volume Ratio (recent 5 days)
        # Higher ratio indicates more selling pressure
        recent_short_ratio = np.mean(short_volume[-5:] / np.maximum(total_volume[-5:], 1e-8))
        eng.append(recent_short_ratio)
        
        # 3. Short Volume Trend (slope over last 10 days)
        # Positive slope indicates increasing short selling activity
        if len(short_volume) >= 10:
            x = np.arange(10)
            y = short_volume[-10:]
            slope = np.polyfit(x, y, 1)[0] if np.any(y) else 0.0
            # Normalize by average short volume
            slope = slope / max(np.mean(short_volume[-10:]), 1e-8)
            eng.append(slope)
        else:
            eng.append(0.0)
        
        # 4. Price Momentum (5-day)
        # Captures recent price trend direction and strength
        if len(close_prices) >= 5:
            momentum_5d = (close_prices[-1] / max(close_prices[-5], 1e-8)) - 1.0
            eng.append(momentum_5d)
        else:
            eng.append(0.0)
        
        # 5. Price Volatility (standard deviation of returns)
        # Higher volatility often correlates with higher short interest
        if len(close_prices) >= 5:
            returns = np.diff(close_prices[-6:]) / np.maximum(close_prices[-6:-1], 1e-8)
            volatility = np.std(returns) if len(returns) > 0 else 0.0
            eng.append(volatility)
        else:
            eng.append(0.0)
        
        # 6. Short Interest Momentum
        # Rate of change in short interest, indicating acceleration/deceleration
        if t > 0 and data[t-1, 0] > 1e-8:
            si_momentum = (short_interest / max(data[t-1, 0], 1e-8)) - 1.0
            eng.append(si_momentum)
        else:
            eng.append(0.0)
        
        # 7. Short Interest to Volume Ratio
        # Indicates how many days of average volume the short interest represents
        si_to_volume = short_interest / max(avg_volume, 1e-8)
        eng.append(si_to_volume)
        
        # 8. Relative Strength Index (RSI)
        # Overbought/oversold indicator that may correlate with short interest changes
        if len(close_prices) >= 14:
            delta = np.diff(close_prices[-15:])
            gain = np.sum(np.where(delta > 0, delta, 0))
            loss = np.sum(np.where(delta < 0, -delta, 0))
            
            if loss > 1e-8:
                rs = gain / max(loss, 1e-8)
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0 if gain > 0 else 50.0
            eng.append(rsi)
        else:
            eng.append(50.0)  # Neutral RSI value
        
        # If we have space for more features, add these high-importance ones
        if MAX_TOTAL > 15:
            # 9. Short Volume Acceleration
            # Second derivative of short volume
            if len(short_volume) >= 3:
                short_vol_diff1 = short_volume[-1] - short_volume[-2]
                short_vol_diff2 = short_volume[-2] - short_volume[-3]
                short_accel = short_vol_diff1 - short_vol_diff2
                short_accel = short_accel / max(np.mean(short_volume[-3:]), 1e-8)  # Normalize
                eng.append(short_accel)
            else:
                eng.append(0.0)
            
            # 10. VWAP Deviation
            # Distance of current price from volume-weighted average price
            if len(close_prices) >= 5 and np.sum(total_volume[-5:]) > 0:
                vwap = np.sum(close_prices[-5:] * total_volume[-5:]) / max(np.sum(total_volume[-5:]), 1e-8)
                vwap_dev = (close_prices[-1] / max(vwap, 1e-8)) - 1.0
                eng.append(vwap_dev)
            else:
                eng.append(0.0)
            
            # 11. Bollinger Band Position
            # Where price is within volatility bands
            if len(close_prices) >= 10:
                sma = np.mean(close_prices[-10:])
                std = np.std(close_prices[-10:])
                bb_position = (close_prices[-1] - sma) / max(2 * std, 1e-8)  # Normalized position
                eng.append(bb_position)
            else:
                eng.append(0.0)
            
            # 12. Implied Volatility to Historical Volatility Ratio
            # Compares market expectations to realized volatility
            if volatility > 0:
                iv_hv_ratio = implied_volatility / max(volatility, 1e-8)
                eng.append(iv_hv_ratio)
            else:
                eng.append(1.0)  # Default when historical volatility is unavailable
            
            # 13. Short Interest to Days to Cover Ratio
            # Relates absolute short interest to relative coverage difficulty
            si_to_dtc = short_interest / max(days_to_cover, 1e-8)
            eng.append(si_to_dtc)
            
            # 14. Average True Range (ATR) - Normalized
            # Volatility measure normalized by price
            if len(high_prices) >= 5 and len(low_prices) >= 5 and len(close_prices) >= 6:
                tr_values = []
                for i in range(1, min(5, len(high_prices))):
                    high_low = high_prices[-i] - low_prices[-i]
                    high_close = abs(high_prices[-i] - close_prices[-(i+1)])
                    low_close = abs(low_prices[-i] - close_prices[-(i+1)])
                    tr = max(high_low, high_close, low_close)
                    tr_values.append(tr)
                atr = np.mean(tr_values) if tr_values else 0
                normalized_atr = atr / max(close_prices[-1], 1e-8)
                eng.append(normalized_atr)
            else:
                eng.append(0.0)
            
            # 15. Short Volume to Total Volume Ratio Change
            # Measures the acceleration in short selling relative to total volume
            if len(short_volume) >= 5 and len(total_volume) >= 5:
                recent_ratio = short_volume[-1] / max(total_volume[-1], 1e-8)
                past_ratio = np.mean(short_volume[-5:-1]) / max(np.mean(total_volume[-5:-1]), 1e-8)
                ratio_change = recent_ratio - past_ratio
                eng.append(ratio_change)
            else:
                eng.append(0.0)
            
            # 16. Price Gap Analysis
            # Identifies significant overnight price gaps
            if len(close_prices) >= 2 and len(open_prices) >= 1:
                overnight_gap = (open_prices[-1] / max(close_prices[-2], 1e-8)) - 1.0
                eng.append(overnight_gap)
            else:
                eng.append(0.0)
            
            # 17. Short Cost to Price Ratio
            # Relates borrowing cost to price level
            short_cost_to_price = synthetic_short_cost / max(close_prices[-1], 1e-8)
            eng.append(short_cost_to_price)
            
            # 18. Normalized Short Interest Position
            # Current SI relative to recent range
            if t >= 5:
                historical_si = [data[max(0, t-i), 0] for i in range(5)]
                si_min, si_max = min(historical_si), max(historical_si)
                si_range = si_max - si_min
                if si_range > 1e-8:
                    norm_si = (short_interest - si_min) / si_range
                else:
                    norm_si = 0.5  # Default to middle if no range
                eng.append(norm_si)
            else:
                eng.append(0.5)
        
        # Ensure we don't exceed MAX_NEW
        if len(eng) > MAX_NEW:
            eng = eng[:MAX_NEW]
        
        # Combine raw and engineered features
        row = np.array(raw_keep + eng, dtype=np.float32)
        
        # Ensure consistent width by padding or truncating
        if row.size < MAX_TOTAL:
            row = np.pad(row, (0, MAX_TOTAL - row.size), 'constant')
        elif row.size > MAX_TOTAL:
            row = row[:MAX_TOTAL]
        
        features[t] = row
    
    # Handle NaN and infinity values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features