"""
kronos_signal.py - Multi-Factor Quant Research & Trading Platform
Level-1 stacked signal engine. Trains separate, independent binary classifiers
for Bullish and Bearish boundaries using walk-forward out-of-sample training
with a strict 10-day embargo.
"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from data_pipeline import DataPipeline

logger = logging.getLogger("quant_system.kronos_signal")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class StackedSignalEngine:
    """
    Stacked Level-1 classifier signal engine.
    Trains separate HistGradientBoostingClassifier models for Bullish
    and Bearish volatility-scaled boundaries. Enforces walk-forward training
    with a 10-day embargo to eliminate target leakage.
    """
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.pipeline = DataPipeline(cache_dir=cache_dir)
        self.cache_dir = self.pipeline.cache_dir
        
    def generate_calibrated_signals(self, ticker: str, k: float = 0.75, train_warmup: int = 252) -> pd.DataFrame:
        logger.info(f"Generating stacked Level-1 signals for {ticker} (k={k})...")
        
        # 1. Prepare/Load classifier dataset
        features_path = self.cache_dir / f"{ticker}_classifier_features.csv"
        
        # Build features if not present
        if not features_path.exists():
            logger.info(f"Classifier features not found. Generating...")
            # Use standard 5-year range with dynamic end date
            import datetime
            end_date_str = datetime.date.today().strftime("%Y-%m-%d")
            df = self.pipeline.prepare_classifier_dataset(ticker, "2021-05-31", end_date_str, k=k)
        else:
            logger.info(f"Loading features from {features_path}")
            df = pd.read_csv(features_path, parse_dates=True, index_col=0)
            
        # Define feature columns
        feature_cols = [
            'kronos_predicted_return', 
            'log_ret_5d', 
            'log_ret_10d', 
            'log_ret_20d', 
            'vix_close', 
            'iv_rv_spread',
            'vol_momentum',
            'empirical_vrp_mult',
            'vix_slope'
        ]
        
        # Ensure data is sorted by index (date)
        df = df.sort_index()
        
        # Allocate output probability arrays
        prob_bulls = np.zeros(len(df))
        prob_bears = np.zeros(len(df))
        
        # Warmup period is filled with nan/neutral
        prob_bulls[:train_warmup] = np.nan
        prob_bears[:train_warmup] = np.nan
        
        X = df[feature_cols].values
        y_bull = df['target_bullish'].values
        y_bear = df['target_bearish'].values
        
        # 2. Walk-forward Daily Training Loop with Embargo
        logger.info("Executing daily walk-forward out-of-sample training (this will take about 10 seconds)...")
        embargo = 10  # Enforce 10-day embargo gap to prevent leakage from overlapping target windows
        
        for t in range(train_warmup, len(df)):
            # Training index bound (exclude the latest 'embargo' days)
            train_end = t - embargo
            if train_end <= 10:
                continue
                
            X_train = X[:train_end]
            y_train_bull = y_bull[:train_end]
            y_train_bear = y_bear[:train_end]
            
            X_test = X[t:t+1] # Row t
            
            # Clean NaN rows from training set
            non_nan_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train_bull) & ~np.isnan(y_train_bear)
            X_train = X_train[non_nan_mask]
            y_train_bull = y_train_bull[non_nan_mask]
            y_train_bear = y_train_bear[non_nan_mask]
            
            if len(X_train) < 20 or np.isnan(X_test).any():
                prob_bulls[t] = 0.5
                prob_bears[t] = 0.5
                continue
                
            # Fit scaler and scale features to prevent L2 regularisation scale distortion
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train independent models (HistGradientBoostingClassifier)
            model_bull = HistGradientBoostingClassifier(max_iter=50, max_depth=3, l2_regularization=5.0, learning_rate=0.05, random_state=42)
            model_bear = HistGradientBoostingClassifier(max_iter=50, max_depth=3, l2_regularization=5.0, learning_rate=0.05, random_state=42)
            
            # Bullish training: check if we have both classes represented in training set
            if len(np.unique(y_train_bull)) > 1:
                model_bull.fit(X_train_scaled, y_train_bull)
                prob_bulls[t] = model_bull.predict_proba(X_test_scaled)[0][1]
            else:
                prob_bulls[t] = 0.5
                
            # Bearish training
            if len(np.unique(y_train_bear)) > 1:
                model_bear.fit(X_train_scaled, y_train_bear)
                prob_bears[t] = model_bear.predict_proba(X_test_scaled)[0][1]
            else:
                prob_bears[t] = 0.5
                
        df['pred_prob_bull'] = prob_bulls
        df['pred_prob_bear'] = prob_bears
        
        # Save output calibrated predictions
        out_path = self.cache_dir / f"{ticker}_classifier_predictions.csv"
        df.to_csv(out_path)
        logger.info(f"Saved stacked predictions to {out_path} (shape: {df.shape})")
        
        return df

def generate_signals():
    engine = StackedSignalEngine()
    engine.generate_calibrated_signals("QQQ")
    engine.generate_calibrated_signals("SPY")

if __name__ == "__main__":
    generate_signals()
