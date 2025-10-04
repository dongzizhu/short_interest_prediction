"""
Universal Feature Engineering Code for Short Interest Prediction
Generated from best practices across multiple tickers
Generated on: 2025-10-04 03:11:47
Source tickers: CYRX, ZEUS, DXLG, SMBK, FCEL
"""

import numpy as np

def construct_features(data):
    RAW_DIM = 62
    MAX_TOTAL = 25
    
    lookback_window = data.shape[0]
    features = np.zeros((lookback_window, MAX_TOTAL), dtype=np.float32)
    
    for t in range(lookback_window):
        # Extract raw data for this timestep
        short_interest = data[t, 0]
        avg_volume = data[t, 1]
        ohlc = data[t, 2:].reshape(15, 4)
        open_prices, high_prices, low_prices, close_prices = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        
        # Keep essential raw features that consistently showed importance across all tickers
        raw_keep = [
            short_interest,  # Always highest importance
            avg_volume,      # Always high importance
            close_prices[-1] # Most recent close price
        ]
        
        # Calculate MAX_NEW based on raw features kept
        MAX_NEW = MAX_TOTAL - len(raw_keep)
        eng = []
        
        #---------- CORE SHORT INTEREST FEATURES ----------#
        
        # 1. Days to cover / SI-to-volume ratio (consistently highest importance)
        # Measures how many days of average trading volume would be needed to cover all short positions
        days_to_cover = short_interest / max(avg_volume, 1e-8)
        eng.append(days_to_cover)
        
        # 2. Short interest momentum (high importance across all tickers)
        # Measures the rate of change in short interest
        si_momentum = 0.0
        if t >= 1:
            prev_si = data[t-1, 0]
            si_momentum = (short_interest - prev_si) / max(abs(prev_si), 1e-8)
        eng.append(si_momentum)
        
        # 3. Short interest acceleration (2nd derivative)
        # Measures how the rate of change in short interest is itself changing
        si_accel = 0.0
        if t >= 2:
            prev_si = data[t-1, 0]
            prev_prev_si = data[t-2, 0]
            prev_change = (prev_si - prev_prev_si) / max(abs(prev_prev_si), 1e-8)
            current_change = si_momentum
            si_accel = current_change - prev_change
        eng.append(si_accel)
        
        # 4. Short interest to price ratio (high importance in multiple tickers)
        # Relates short interest to current price level
        si_price_ratio = short_interest / max(close_prices[-1], 1e-8)
        eng.append(si_price_ratio)
        
        #---------- PRICE MOMENTUM FEATURES ----------#
        
        # 5. Recent price momentum (5-day, consistently important)
        price_change_5d = 0.0
        if len(close_prices) >= 5:
            price_change_5d = (close_prices[-1] - close_prices[-5]) / max(close_prices[-5], 1e-8)
        eng.append(price_change_5d)
        
        # 6. Longer-term price momentum (10-day)
        price_change_10d = 0.0
        if len(close_prices) >= 10:
            price_change_10d = (close_prices[-1] - close_prices[-10]) / max(close_prices[-10], 1e-8)
        eng.append(price_change_10d)
        
        # 7. Price trend strength (linear regression slope)
        price_trend = 0.0
        if len(close_prices) >= 5:
            # Use adaptive window size based on available data
            window = min(len(close_prices), 10)
            x = np.arange(window)
            y = close_prices[-window:]
            
            # Apply exponential weights to emphasize recent price action
            weights = np.exp(np.linspace(0, 1, window)) 
            weights = weights / np.sum(weights)
            
            x_mean = np.sum(weights * x)
            y_mean = np.sum(weights * y)
            numerator = np.sum(weights * (x - x_mean) * (y - y_mean))
            denominator = np.sum(weights * (x - x_mean) ** 2)
            denominator = max(denominator, 1e-8)
            price_trend = numerator / denominator
            
            # Normalize by average price
            avg_price = max(np.mean(y), 1e-8)
            price_trend = price_trend / avg_price
        eng.append(price_trend)
        
        #---------- VOLATILITY FEATURES ----------#
        
        # 8. Normalized volatility (ATR-based)
        atr = 0.0
        if len(close_prices) >= 2:
            tr_values = []
            for i in range(1, min(5, len(close_prices))):
                high_low = high_prices[-i] - low_prices[-i]
                high_close = abs(high_prices[-i] - close_prices[-(i+1)] if i+1 < len(close_prices) else close_prices[-i])
                low_close = abs(low_prices[-i] - close_prices[-(i+1)] if i+1 < len(close_prices) else close_prices[-i])
                tr = max(high_low, high_close, low_close)
                tr_values.append(tr)
            atr = np.mean(tr_values) if tr_values else 0
            atr = atr / max(close_prices[-1], 1e-8)  # Normalize by price
        eng.append(atr)
        
        # 9. Bollinger Band width (volatility measure)
        bb_width = 0.0
        if len(close_prices) >= 10:
            sma = np.mean(close_prices[-10:])
            std = np.std(close_prices[-10:])
            bb_width = (2 * std) / max(sma, 1e-8)
        eng.append(bb_width)
        
        # 10. Bollinger Band position (mean reversion indicator)
        bb_position = 0.0
        if len(close_prices) >= 10:
            sma = np.mean(close_prices[-10:])
            std = np.std(close_prices[-10:])
            bb_position = (close_prices[-1] - sma) / max(2 * std, 1e-8)
            # Clip to reasonable range
            bb_position = max(min(bb_position, 3.0), -3.0)
        eng.append(bb_position)
        
        #---------- TECHNICAL INDICATORS ----------#
        
        # 11. RSI (Relative Strength Index)
        rsi = 50.0  # Default neutral value
        if len(close_prices) >= 14:
            delta = np.diff(close_prices[-14:])
            gains = np.where(delta > 0, delta, 0)
            losses = np.where(delta < 0, -delta, 0)
            
            # Use exponential weighting for more responsive indicator
            weights = np.exp(np.linspace(0, 1, len(gains)))
            weights = weights / np.sum(weights)
            avg_gain = np.sum(weights * gains)
            
            weights = np.exp(np.linspace(0, 1, len(losses)))
            weights = weights / np.sum(weights)
            avg_loss = np.sum(weights * losses)
            
            rs = avg_gain / max(avg_loss, 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # Normalize to [0,1] range
            rsi = rsi / 100.0
        eng.append(rsi)
        
        # 12. Short interest to RSI ratio
        si_rsi_ratio = short_interest / max(rsi * 100, 1e-8)
        # Scale to avoid extreme values
        si_rsi_ratio = min(si_rsi_ratio, 10.0)
        eng.append(si_rsi_ratio)
        
        # 13. MACD (Moving Average Convergence/Divergence)
        macd = 0.0
        if len(close_prices) >= 12:
            # Simple approximation of EMA with different weights
            ema_short = np.average(close_prices[-6:], weights=np.linspace(1, 2, 6))
            ema_long = np.average(close_prices[-12:], weights=np.linspace(1, 2, 12))
            macd = (ema_short - ema_long) / max(ema_long, 1e-8)
        eng.append(macd)
        
        #---------- VOLUME FEATURES ----------#
        
        # 14. Volume trend
        vol_trend = 0.0
        if t > 0:
            prev_volume = data[t-1, 1]
            vol_trend = (avg_volume - prev_volume) / max(prev_volume, 1e-8)
        eng.append(vol_trend)
        
        # 15. Relative volume (compared to recent history)
        rel_volume = 1.0  # Default neutral value
        if t > 0:
            vol_history = [data[max(0, t-i), 1] for i in range(1, min(t+1, 6))]
            if vol_history:
                avg_hist_vol = max(np.mean(vol_history), 1e-8)
                rel_volume = avg_volume / avg_hist_vol
        eng.append(rel_volume)
        
        # 16. Volume-weighted price momentum
        vol_price_momentum = 0.0
        if len(close_prices) >= 5:
            vol_price_momentum = price_change_5d * (avg_volume / max(np.mean(data[max(0, t-5):t+1, 1]), 1e-8))
        eng.append(vol_price_momentum)
        
        #---------- COMPOSITE INDICATORS ----------#
        
        # 17. Short squeeze potential
        # Combines days to cover with recent price momentum and volume
        squeeze_potential = 0.0
        if price_change_5d > 0:  # Only positive when price is rising
            squeeze_potential = days_to_cover * price_change_5d * rel_volume
            # Scale to avoid extreme values
            squeeze_potential = min(squeeze_potential, 10.0)
        eng.append(squeeze_potential)
        
        # 18. Short interest to volatility ratio
        si_vol_ratio = short_interest / max(atr * 100, 1e-8)
        eng.append(si_vol_ratio)
        
        # 19. Price reversal indicator
        reversal = 0.0
        if len(close_prices) >= 5 and bb_width > 0:
            # Calculate recent price trend
            recent_trend = price_change_5d
            
            # Reversal signal based on price position and momentum
            if recent_trend > 0 and bb_position > 0.8:  # Uptrend near upper band
                reversal = -bb_position  # Potential downward reversal
            elif recent_trend < 0 and bb_position < -0.8:  # Downtrend near lower band
                reversal = -bb_position  # Potential upward reversal
        eng.append(reversal)
        
        # 20. Intraday volatility
        intraday_vol = 0.0
        if len(high_prices) >= 5 and len(low_prices) >= 5:
            intraday_ranges = (high_prices[-5:] - low_prices[-5:]) / np.maximum(open_prices[-5:], 1e-8)
            intraday_vol = np.mean(intraday_ranges)
        eng.append(intraday_vol)
        
        # 21. Gap analysis
        avg_gap = 0.0
        if len(close_prices) >= 2 and len(open_prices) >= 1:
            gap = (open_prices[-1] - close_prices[-2]) / max(close_prices[-2], 1e-8)
            # Normalize to typical range
            avg_gap = np.clip(gap / 0.02, -5.0, 5.0)
        eng.append(avg_gap)
        
        # 22. Short interest change relative to price change
        si_price_change_ratio = 0.0
        if t > 0 and len(close_prices) >= 2:
            prev_si = data[t-1, 0]
            prev_close = close_prices[-2]
            
            si_pct_change = (short_interest - prev_si) / max(prev_si, 1e-8)
            price_pct_change = (close_prices[-1] - prev_close) / max(prev_close, 1e-8)
            
            if abs(price_pct_change) > 1e-8:
                si_price_change_ratio = si_pct_change / price_pct_change
                # Clamp extreme values
                si_price_change_ratio = max(min(si_price_change_ratio, 10), -10)
        eng.append(si_price_change_ratio)
        
        # Ensure we don't exceed MAX_NEW
        eng = eng[:MAX_NEW]
        
        # Combine raw and engineered features
        row = np.array(raw_keep + eng, dtype=np.float32)
        
        # Ensure consistent size
        if row.size < MAX_TOTAL:
            row = np.pad(row, (0, MAX_TOTAL - row.size), 'constant')
        elif row.size > MAX_TOTAL:
            row = row[:MAX_TOTAL]
        
        features[t] = row
    
    # Handle NaN, inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features