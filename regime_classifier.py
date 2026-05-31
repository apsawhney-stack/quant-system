"""
regime_classifier.py - Multi-Factor Quant Research & Trading Platform
Classifies market regimes (Risk-On, Neutral, Risk-Off) using 1-year rolling
percentile-normalized metrics and trend indicators.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

logger = logging.getLogger("quant_system.regime_classifier")

class RegimeClassifier:
    """
    Classifies current market volatility and trend regimes.
    Uses 1-year rolling percentile-normalized metrics to handle non-stationarity
    (e.g., VIX 20 is high or low depending on the trailing 12 months).
    """
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        logger.info(f"Regime classifier initialized with trailing lookback window of {self.lookback_window} days.")

    def calculate_indicators(self, underlying_close: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
        """
        Calculates all intermediate metrics needed for regime classification:
        - 20-day realized volatility of underlying
        - 50-day moving average of underlying & its 5-day slope
        - 1-year rolling percentiles of VIX and realized volatility
        - VIX rate-of-change and VIX / 10-day MA ratio for PANIC detection
        """
        df = pd.DataFrame(index=underlying_close.index)
        df['underlying_close'] = underlying_close
        df['vix_close'] = vix_close
        
        # 1. Calculate realized volatility (20-day rolling standard deviation of log returns, annualized)
        log_returns = np.log(underlying_close / underlying_close.shift(1))
        df['realized_vol_20d'] = log_returns.rolling(window=20).std() * np.sqrt(252)
        
        # 2. Calculate moving average and slope
        df['ma_50'] = underlying_close.rolling(window=50).mean()
        # 5-day difference of the 50-DMA to capture the trend slope
        df['ma_50_slope'] = df['ma_50'].diff(periods=5)
        
        # 3. Trailing 1-year percentiles
        def pct_rank(window):
            if len(window) < 2:
                return np.nan
            current_val = window[-1]
            less_than_count = np.sum(window[:-1] < current_val)
            return (less_than_count / (len(window) - 1))  # Returns range 0.0 to 1.0
            
        df['vix_percentile_1y'] = vix_close.rolling(
            window=self.lookback_window, min_periods=self.lookback_window
        ).apply(pct_rank, raw=True)
        
        df['realized_vol_percentile_1y'] = df['realized_vol_20d'].rolling(
            window=self.lookback_window, min_periods=self.lookback_window
        ).apply(pct_rank, raw=True)
        
        # 4. Volatility acceleration metrics for PANIC detection
        df['vix_roc_5d'] = (vix_close - vix_close.shift(5)) / vix_close.shift(5)
        vix_ma_10 = vix_close.rolling(window=10).mean()
        df['vix_ratio'] = vix_close / vix_ma_10
        
        return df

    def classify_regime_row(
        self, 
        vix_percentile_1y: float, 
        price: float, 
        ma_50: float, 
        ma_50_slope: float, 
        realized_vol_percentile_1y: float
    ) -> str:
        """
        Classifies a single row/timestamp into a regime state.
        Rules:
        - RISK_OFF: VIX percentile > 80% OR price < 50-DMA with negative slope
        - RISK_ON: VIX percentile < 30% AND price > 50-DMA AND realized vol percentile < 30%
        - NEUTRAL: Default middle ground
        """
        if pd.isna(vix_percentile_1y) or pd.isna(realized_vol_percentile_1y) or pd.isna(ma_50):
            return "NEUTRAL" # Default to neutral during warmup
            
        if vix_percentile_1y > 0.80 or (price < ma_50 and ma_50_slope < 0):
            return "RISK_OFF"
        elif vix_percentile_1y < 0.30 and price > ma_50 and realized_vol_percentile_1y < 0.30:
            return "RISK_ON"
        else:
            return "NEUTRAL"

    def classify_history(self, underlying_close: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
        """
        Stateful classification across historical time-series data with PANIC hysteresis.
        Returns a DataFrame containing intermediate metrics and the stateful 'regime'.
        """
        df = self.calculate_indicators(underlying_close, vix_close)
        
        regimes = []
        current_regime = "NEUTRAL"
        panic_exit_counter = 0
        
        for idx, row in df.iterrows():
            vix_p = row['vix_percentile_1y']
            price = row['underlying_close']
            ma_50 = row['ma_50']
            ma_50_slope = row['ma_50_slope']
            vol_p = row['realized_vol_percentile_1y']
            
            roc = row['vix_roc_5d']
            ratio = row['vix_ratio']
            
            # Check PANIC triggers
            is_panic_trigger = (roc > 0.05) or (ratio > 1.15)
            is_stabilized = (roc < 0.01) and (ratio < 1.05)
            
            if current_regime == "PANIC":
                if is_panic_trigger:
                    panic_exit_counter = 0 # reset
                elif is_stabilized:
                    panic_exit_counter += 1
                else:
                    panic_exit_counter = 0
                    
                if panic_exit_counter >= 3:
                    current_regime = "RISK_OFF" # Transition out of panic
                    panic_exit_counter = 0
            else:
                if is_panic_trigger:
                    current_regime = "PANIC"
                    panic_exit_counter = 0
                    
            # If not in PANIC, run normal regime classification rules
            if current_regime != "PANIC":
                current_regime = self.classify_regime_row(
                    vix_percentile_1y=vix_p,
                    price=price,
                    ma_50=ma_50,
                    ma_50_slope=ma_50_slope,
                    realized_vol_percentile_1y=vol_p
                )
                    
            regimes.append(current_regime)
            
        df['regime'] = regimes
        return df

# =========================================================================
# TEST DRIVER
# =========================================================================
if __name__ == "__main__":
    logger.info("Running RegimeClassifier basic validation...")
    
    # Generate synthetic trend and vol data
    dates = pd.date_range(start="2020-01-01", periods=600, freq="B")
    
    # Create synthetic price series (increasing then crashing then chopping)
    prices = [400.0]
    for i in range(1, 600):
        if i < 200:
            # Uptrend
            prices.append(prices[-1] * (1.0 + np.random.normal(0.0005, 0.01)))
        elif i < 300:
            # Crash
            prices.append(prices[-1] * (1.0 + np.random.normal(-0.002, 0.02)))
        else:
            # Chop
            prices.append(prices[-1] * (1.0 + np.random.normal(0.0, 0.01)))
            
    underlying_close = pd.Series(prices, index=dates)
    
    # Create VIX series (inversely correlated, with spikes)
    vix = [15.0]
    for i in range(1, 600):
        prev = vix[-1]
        ret = (underlying_close.iloc[i] - underlying_close.iloc[i-1]) / underlying_close.iloc[i-1]
        change = -150.0 * ret + np.random.normal(0.0, 1.0)
        new_vix = max(prev + change, 9.0)
        vix.append(new_vix)
        
    vix_close = pd.Series(vix, index=dates)
    
    classifier = RegimeClassifier(lookback_window=252)
    regime_df = classifier.classify_history(underlying_close, vix_close)
    
    logger.info(f"Regime counts:\n{regime_df['regime'].value_counts()}")
    logger.info("First few classified dates:")
    print(regime_df.dropna().head(10))
