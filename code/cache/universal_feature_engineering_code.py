"""
Universal Feature Engineering Code for Short Interest Prediction
Generated from best practices across multiple tickers
Generated on: 2025-10-03 17:44:55
Source tickers: AAPL, TSLA
"""

import numpy as np

def construct_features(data):
    """
    Constructs optimized features for short interest prediction.
    
    Args:
        data: numpy array with shape (lookback_window, 62)
            - data[t, 0]: short interest (SI_t)
            - data[t, 1]: average daily volume (past 15 days)
            - data[t, 2:62]: OHLC for past 15 days, flattened as 15×4
    
    Returns:
        features: numpy array with shape (lookback_window, 50)
    """
    MAX_FEATURES = 50  # Reduced from 80 to focus on most significant features
    
    lookback_window = data.shape[0]
    features = np.zeros((lookback_window, MAX_FEATURES), dtype=np.float32)
    
    for t in range(lookback_window):
        # Extract raw data for this timestep
        short_interest = data[t, 0]
        avg_volume = data[t, 1]
        ohlc = data[t, 2:].reshape(15, 4)
        open_prices, high_prices, low_prices, close_prices = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        
        # Initialize feature vector
        feat = []
        
        # 1. CORE RAW FEATURES - Essential baseline signals
        feat.extend([
            short_interest,                # Short interest (highest importance)
            avg_volume,                    # Average volume
            close_prices[-1],              # Most recent close price
            high_prices[-1],               # Most recent high price
            low_prices[-1],                # Most recent low price
        ])
        
        # 2. SHORT INTEREST DYNAMICS - Key SI relationships
        # SI/Volume ratio (consistently important)
        si_vol_ratio = short_interest / max(avg_volume, 1e-8)
        feat.append(si_vol_ratio)
        
        # SI momentum and acceleration
        if t > 0:
            prev_si = data[t-1, 0]
            si_change = (short_interest / max(prev_si, 1e-8)) - 1
            feat.append(si_change)
            
            # SI acceleration (second derivative)
            if t > 1:
                prev_prev_si = data[t-2, 0]
                prev_si_change = (prev_si / max(prev_prev_si, 1e-8)) - 1
                si_acceleration = si_change - prev_si_change
                feat.append(si_acceleration)
            else:
                feat.append(0.0)
        else:
            feat.extend([0.0, 0.0])
        
        # SI relative to price
        si_price_ratio = short_interest / max(close_prices[-1], 1e-8)
        feat.append(si_price_ratio)
        
        # SI to free float proxy
        si_float_proxy = short_interest / max(avg_volume * 20, 1e-8)
        feat.append(min(si_float_proxy, 5.0))  # Cap at 5 to avoid extreme values
        
        # 3. PRICE MOMENTUM - Recent price movements
        if len(close_prices) > 1:
            # 1-day return
            daily_return = (close_prices[-1] / max(close_prices[-2], 1e-8)) - 1
            feat.append(daily_return)
            
            # 5-day return
            if len(close_prices) >= 6:
                five_day_return = (close_prices[-1] / max(close_prices[-6], 1e-8)) - 1
                feat.append(five_day_return)
            else:
                feat.append(0.0)
            
            # 10-day return
            if len(close_prices) >= 11:
                ten_day_return = (close_prices[-1] / max(close_prices[-11], 1e-8)) - 1
                feat.append(ten_day_return)
            else:
                feat.append(0.0)
        else:
            feat.extend([0.0, 0.0, 0.0])
        
        # 4. VOLATILITY METRICS - Measure of price dispersion
        # True Range and ATR
        true_range = []
        for i in range(1, len(close_prices)):
            tr = max(
                high_prices[i] - low_prices[i],
                abs(high_prices[i] - close_prices[i-1]),
                abs(low_prices[i] - close_prices[i-1])
            )
            true_range.append(tr)
        
        if true_range:
            # ATR (Average True Range)
            atr = np.mean(true_range[-5:]) if len(true_range) >= 5 else np.mean(true_range)
            feat.append(atr)
            
            # Normalized ATR (relative to price)
            atr_rel = atr / max(close_prices[-1], 1e-8)
            feat.append(atr_rel)
        else:
            feat.extend([0.0, 0.0])
        
        # 5. VOLUME DYNAMICS
        # Volume to price ratio
        vol_price_ratio = avg_volume / max(close_prices[-1], 1e-8)
        feat.append(vol_price_ratio)
        
        # Volume momentum
        if t > 0:
            prev_volume = data[t-1, 1]
            vol_change = (avg_volume / max(prev_volume, 1e-8)) - 1
            feat.append(vol_change)
        else:
            feat.append(0.0)
        
        # 6. TECHNICAL INDICATORS - RSI
        if len(close_prices) >= 3:
            delta = np.diff(close_prices)
            gain = np.copy(delta)
            loss = np.copy(delta)
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            lookback = min(14, len(gain))
            avg_gain = np.mean(gain[-lookback:])
            avg_loss = np.mean(loss[-lookback:])
            
            if avg_loss > 1e-8:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100.0 if avg_gain > 0 else 50.0
            
            feat.append(rsi)
            
            # RSI extremes (overbought/oversold)
            rsi_extreme = 0.0
            if rsi > 70:  # Overbought
                rsi_extreme = (rsi - 70) / 30
            elif rsi < 30:  # Oversold
                rsi_extreme = (30 - rsi) / 30
            feat.append(rsi_extreme)
        else:
            feat.extend([50.0, 0.0])
        
        # 7. MOVING AVERAGES
        if len(close_prices) >= 5:
            sma5 = np.mean(close_prices[-5:])
            
            # Price relative to 5-day SMA
            price_sma_ratio = close_prices[-1] / max(sma5, 1e-8) - 1
            feat.append(price_sma_ratio)
            
            if len(close_prices) >= 10:
                sma10 = np.mean(close_prices[-10:])
                
                # 5-day SMA relative to 10-day SMA (trend indicator)
                sma_ratio = sma5 / max(sma10, 1e-8) - 1
                feat.append(sma_ratio)
            else:
                feat.append(0.0)
        else:
            feat.extend([0.0, 0.0])
        
        # 8. BOLLINGER BANDS
        if len(close_prices) >= 5:
            sma = np.mean(close_prices[-5:])
            std = np.std(close_prices[-5:])
            
            upper_band = sma + 2 * std
            lower_band = sma - 2 * std
            
            # Bollinger Band Width (volatility)
            bb_width = (upper_band - lower_band) / max(sma, 1e-8)
            feat.append(bb_width)
            
            # Bollinger Band Position
            bb_pos = (close_prices[-1] - lower_band) / max(upper_band - lower_band, 1e-8)
            bb_pos = max(min(bb_pos, 1.0), 0.0)  # Clamp to [0, 1]
            feat.append(bb_pos)
        else:
            feat.extend([0.0, 0.0])
        
        # 9. PRICE PATTERNS
        if len(close_prices) > 0:
            # Doji pattern (open ≈ close)
            range_day = max(high_prices[-1] - low_prices[-1], 1e-8)
            body_size = abs(open_prices[-1] - close_prices[-1])
            doji = 1.0 - (body_size / range_day)
            feat.append(doji)
        else:
            feat.append(0.0)
        
        # 10. COMBINED SI & TECHNICAL INDICATORS
        # SI relative to RSI (potential reversal signal)
        if 'rsi' in locals():
            # Higher when SI high and RSI low (potential short squeeze)
            si_rsi = short_interest * (100 - rsi) / 100
            si_rsi_norm = si_rsi / max(avg_volume, 1e-8)
            feat.append(si_rsi_norm)
        else:
            feat.append(0.0)
        
        # SI combined with price momentum
        if len(close_prices) > 1:
            price_momentum = (close_prices[-1] / max(close_prices[-2], 1e-8)) - 1
            
            # Higher when SI high and momentum negative (potential short squeeze)
            si_momentum = short_interest * (-1 * price_momentum if price_momentum < 0 else 0)
            si_momentum_norm = si_momentum / max(avg_volume, 1e-8)
            feat.append(si_momentum_norm)
        else:
            feat.append(0.0)
        
        # SI combined with volatility
        if 'atr_rel' in locals():
            # Higher when SI high and volatility high (potential for rapid moves)
            si_vol = short_interest * atr_rel
            si_vol_norm = si_vol / max(avg_volume, 1e-8)
            feat.append(si_vol_norm)
        else:
            feat.append(0.0)
        
        # 11. MEAN REVERSION POTENTIAL
        if len(close_prices) >= 10:
            # Z-score of current price
            mean_price = np.mean(close_prices[-10:])
            std_price = np.std(close_prices[-10:])
            
            if std_price > 1e-8:
                z_score = (close_prices[-1] - mean_price) / std_price
            else:
                z_score = 0.0
                
            # Mean reversion potential (higher when z-score extreme)
            mean_rev = abs(z_score) if abs(z_score) > 1.5 else 0.0
            feat.append(mean_rev)
            
            # Direction of potential mean reversion
            mean_rev_dir = -1.0 * np.sign(z_score) if abs(z_score) > 1.5 else 0.0
            feat.append(mean_rev_dir)
        else:
            feat.extend([0.0, 0.0])
        
        # 12. SI DIVERGENCE WITH PRICE
        if t > 0 and len(close_prices) > 1:
            si_change = short_interest / max(data[t-1, 0], 1e-8) - 1
            price_change = close_prices[-1] / max(close_prices[-2], 1e-8) - 1
            
            # Divergence occurs when SI and price move in same direction
            divergence = si_change * price_change
            feat.append(divergence)
        else:
            feat.append(0.0)
        
        # 13. PRICE EFFICIENCY RATIO
        if len(close_prices) >= 5:
            # Measure of how efficiently price is moving in a direction
            price_path = 0
            for i in range(1, 5):
                price_path += abs(close_prices[-i] - close_prices[-(i+1)])
            
            price_displacement = abs(close_prices[-1] - close_prices[-5])
            
            if price_path > 1e-8:
                efficiency = price_displacement / price_path
            else:
                efficiency = 1.0
                
            feat.append(efficiency)
        else:
            feat.append(0.0)
        
        # Ensure we don't exceed MAX_FEATURES
        feat = np.array(feat, dtype=np.float32)
        if feat.size > MAX_FEATURES:
            feat = feat[:MAX_FEATURES]
        elif feat.size < MAX_FEATURES:
            # Pad with zeros if needed
            padding = np.zeros(MAX_FEATURES - feat.size, dtype=np.float32)
            feat = np.concatenate([feat, padding])
        
        features[t] = feat
    
    # Handle NaN, inf values
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features