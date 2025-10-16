"""
Universal Feature Engineering Code for Short Interest Prediction
Generated from best practices across multiple tickers
Generated on: 2025-10-09 15:59:36
Source tickers: ABCB, EIG, FSS, ABM, IART, SRPT, EXTR, SCSC, SLG, HL, ANDE, AROC
"""

import numpy as np

def construct_features(data):
    """
    Universal feature construction function for short interest prediction.
    
    Args:
        data: numpy array of shape (lookback_window, 97)
        
    Returns:
        numpy array of shape (lookback_window, 25)
    """
    RAW_DIM = 97
    MAX_TOTAL = 25
    
    lookback_window = data.shape[0]
    features = np.zeros((lookback_window, MAX_TOTAL), dtype=np.float32)
    
    for t in range(lookback_window):
        # Essential raw features to keep (consistently important across tickers)
        raw_keep = [
            data[t, 0],  # short interest
            data[t, 1],  # average daily volume
            data[t, 2],  # days to cover
        ]
        
        # Extract OHLC data for the past 15 days
        ohlc = data[t, 3:63].reshape(15, 4)
        open_prices, high_prices, low_prices, close_prices = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        
        # Keep most recent close price
        raw_keep.append(close_prices[-1])
        
        # Keep key options data
        raw_keep.append(data[t, 63])  # options_put_call_volume_ratio
        raw_keep.append(data[t, 65])  # options_avg_implied_volatility
        raw_keep.append(data[t, 66])  # shares_outstanding
        
        # Extract short volume and total volume data
        short_volume = data[t, 67:82]
        total_volume = data[t, 82:97]
        
        # Calculate MAX_NEW based on raw features kept
        MAX_NEW = MAX_TOTAL - len(raw_keep)
        
        # Initialize engineered features list
        eng = []
        
        # 1. Short Volume Ratio (daily short volume / daily total volume)
        short_volume_ratio = np.zeros_like(short_volume)
        for i in range(len(short_volume)):
            denom = max(abs(total_volume[i]), 1e-8)
            short_volume_ratio[i] = short_volume[i] / denom
        
        # Average short volume ratio (recent days weighted more)
        if len(short_volume_ratio) >= 5:
            weights = np.exp(np.linspace(0, 1, 5))
            weights = weights / np.sum(weights)
            weighted_svr = np.sum(weights * short_volume_ratio[-5:])
            eng.append(weighted_svr)
        else:
            eng.append(np.mean(short_volume_ratio))
        
        # 2. Short Interest to Float Ratio
        si_to_float = data[t, 0] / max(abs(data[t, 66]), 1e-8)
        eng.append(si_to_float)
        
        # 3. Short Interest to Volume Ratio
        si_to_volume = data[t, 0] / max(abs(data[t, 1]), 1e-8)
        eng.append(si_to_volume)
        
        # 4. Price Momentum (5-day)
        if len(close_prices) >= 5:
            momentum_5d = close_prices[-1] / max(abs(close_prices[-5]), 1e-8) - 1
        else:
            momentum_5d = 0
        eng.append(momentum_5d)
        
        # 5. Price Volatility
        if len(close_prices) >= 5:
            returns = np.diff(close_prices[-5:]) / np.maximum(close_prices[-5:-1], 1e-8)
            volatility = np.std(returns) if len(returns) > 0 else 0
        else:
            volatility = 0
        eng.append(volatility)
        
        # 6. RSI (Relative Strength Index)
        if len(close_prices) >= 5:
            delta = np.diff(close_prices[-5:])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain) if len(gain) > 0 else 0
            avg_loss = np.mean(loss) if len(loss) > 0 else 0
            denom = max(abs(avg_loss), 1e-8)
            rs = avg_gain / denom
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = 50  # Default value if not enough data
        eng.append(rsi)
        
        # 7. Short Volume Trend
        if len(short_volume) >= 5:
            recent_short = np.mean(short_volume[-3:])
            prev_short = np.mean(short_volume[-5:-2])
            denom = max(abs(prev_short), 1e-8)
            short_vol_trend = recent_short / denom - 1
        else:
            short_vol_trend = 0
        eng.append(short_vol_trend)
        
        # 8. Bollinger Band Position
        if len(close_prices) >= 10:
            sma = np.mean(close_prices[-10:])
            std = np.std(close_prices[-10:])
            upper_band = sma + 2 * std
            lower_band = sma - 2 * std
            band_width = upper_band - lower_band
            denom = max(abs(band_width), 1e-8)
            bb_position = (close_prices[-1] - lower_band) / denom
            bb_position = 2 * bb_position - 1  # Normalize to [-1, 1]
        else:
            bb_position = 0
        eng.append(bb_position)
        
        # 9. Short Interest Momentum
        if t > 0:
            prev_si = data[t-1, 0]
            denom = max(abs(prev_si), 1e-8)
            si_momentum = (data[t, 0] / denom) - 1
        else:
            si_momentum = 0
        eng.append(si_momentum)
        
        # 10. Options Pressure Indicator
        put_call_ratio = data[t, 63]
        implied_vol = data[t, 65]
        options_pressure = (put_call_ratio - 1.0) * (1 + implied_vol/100)
        eng.append(options_pressure)
        
        # 11. Volume Spike Indicator
        if len(total_volume) >= 5:
            recent_vol = total_volume[-1]
            avg_vol = np.mean(total_volume[-5:])
            denom = max(abs(avg_vol), 1e-8)
            vol_spike = recent_vol / denom - 1
        else:
            vol_spike = 0
        eng.append(vol_spike)
        
        # 12. Short Squeeze Potential
        squeeze_potential = si_to_float * data[t, 2] * (1 + volatility)
        eng.append(squeeze_potential)
        
        # 13. MACD Signal
        if len(close_prices) >= 12:
            ema12 = np.mean(close_prices[-12:])
            ema26 = np.mean(close_prices[-min(26, len(close_prices)):])
            macd = ema12 - ema26
            denom = max(abs(np.mean(close_prices[-12:])), 1e-8)
            macd_normalized = macd / denom
        else:
            macd_normalized = 0
        eng.append(macd_normalized)
        
        # 14. Short Volume Acceleration
        if len(short_volume) >= 3:
            diff1 = short_volume[-1] - short_volume[-2]
            diff2 = short_volume[-2] - short_volume[-3]
            short_vol_accel = diff1 - diff2
            denom = max(abs(np.mean(short_volume[-3:])), 1e-8)
            short_vol_accel_norm = short_vol_accel / denom
        else:
            short_vol_accel_norm = 0
        eng.append(short_vol_accel_norm)
        
        # 15. Short Interest to Implied Volatility Ratio
        si_to_iv_ratio = data[t, 0] / max(abs(data[t, 65]), 1e-8)
        eng.append(si_to_iv_ratio)
        
        # 16. Price Gap Analysis
        if len(open_prices) >= 2 and len(close_prices) >= 2:
            denom = max(abs(close_prices[-2]), 1e-8)
            gap = open_prices[-1] / denom - 1
            gap_indicator = np.tanh(gap * 10)  # Scale to emphasize significant gaps
        else:
            gap_indicator = 0
        eng.append(gap_indicator)
        
        # 17. Short Interest Efficiency Ratio
        if len(close_prices) >= 5 and len(short_volume) >= 5:
            returns = []
            for i in range(1, 5):
                denom = max(abs(close_prices[-i-1]), 1e-8)
                ret = close_prices[-i] / denom - 1
                returns.append(ret)
            
            short_vol_norm = short_volume[-5:-1] / np.mean(short_volume[-5:-1])
            returns_norm = np.array(returns)
            
            # Simple dot product as correlation proxy
            efficiency = -np.sum(short_vol_norm * returns_norm) / 4
            short_efficiency = np.tanh(efficiency * 3)
        else:
            short_efficiency = 0
        eng.append(short_efficiency)
        
        # 18. Intraday Range Volatility
        if len(high_prices) >= 5 and len(low_prices) >= 5:
            ranges = []
            for i in range(5):
                denom = max(abs(low_prices[-i-1]), 1e-8)
                daily_range = (high_prices[-i-1] - low_prices[-i-1]) / denom
                ranges.append(daily_range)
            
            range_volatility = np.mean(ranges)
        else:
            range_volatility = 0
        eng.append(range_volatility)
        
        # Ensure we don't exceed MAX_NEW
        eng = eng[:MAX_NEW]
        
        # Combine raw and engineered features
        row = np.array(raw_keep + eng, dtype=np.float32)
        
        # Ensure consistent size by padding or truncating
        if row.size < MAX_TOTAL:
            row = np.pad(row, (0, MAX_TOTAL - row.size), 'constant')
        elif row.size > MAX_TOTAL:
            row = row[:MAX_TOTAL]
        
        features[t] = row
    
    # Handle NaN, inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features