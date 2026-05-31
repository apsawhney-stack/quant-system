"""
meta_classifier.py - Multi-Factor Quant Research & Trading Platform
Level-2 Meta-Labeling Classifier. Learns trade execution viability
using a highly regularized shallow model trained via purged and embargoed
walk-forward cross-validation.
"""

import os
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("quant_system.meta_classifier")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class MetaLabelingClassifier:
    """
    Implements Level-2 Meta-Labeling.
    Trains a HistGradientBoostingClassifier model out-of-sample
    with strict purging (10-day target overlap) and embargo (10-day post-split window).
    """
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        
    def generate_meta_signals(self, ticker: str, prob_threshold: float = 0.55, train_warmup: int = 350) -> pd.DataFrame:
        logger.info(f"Generating Level-2 meta-signals for {ticker}...")
        
        # Load Level-1 predictions
        pred_path = self.cache_dir / f"{ticker}_classifier_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Level-1 prediction file {pred_path} not found. Run kronos_signal.py first.")
            
        df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
        df = df.sort_index()
        
        # 1. Label the Level-1 trades as correct (1) or incorrect (0)
        # We classify a trade as bullish if pred_prob_bull >= threshold
        # and bearish if pred_prob_bear >= threshold (with bullish taking priority if both trigger)
        trade_taken = np.zeros(len(df))
        trade_correct = np.zeros(len(df))
        
        fwd_return = df['fwd_log_ret_10d'].values
        prob_bull = df['pred_prob_bull'].values
        prob_bear = df['pred_prob_bear'].values
        
        for i in range(len(df)):
            if pd.isna(prob_bull[i]) or pd.isna(prob_bear[i]):
                continue
                
            if prob_bull[i] >= prob_threshold and prob_bull[i] >= prob_bear[i]:
                trade_taken[i] = 1 # Bullish trade
                trade_correct[i] = 1 if fwd_return[i] > 0.00 else 0
            elif prob_bear[i] >= prob_threshold:
                trade_taken[i] = -1 # Bearish trade
                trade_correct[i] = 1 if fwd_return[i] < 0.00 else 0
                
        df['level1_trade'] = trade_taken
        df['level2_target'] = trade_correct
        
        # 2. Define Level-2 features
        meta_features = [
            'vix_close',
            'vix_roc_5d',
            'vix_ratio',
            'realized_vol_20d',
            'iv_rv_spread',
            'ma_50_slope',
            'vol_momentum',
            'empirical_vrp_mult',
            'vix_slope'
        ]
        
        X = df[meta_features].values
        y_meta = df['level2_target'].values
        is_trade = (df['level1_trade'] != 0).values
        
        # Allocate output meta probability array
        meta_probs = np.zeros(len(df))
        meta_probs[:] = np.nan
        
        # 3. Purged & Embargoed Daily Walk-Forward Loop
        # We train only on rows where a trade was taken, and apply a 10-day purging + 10-day embargo gap.
        logger.info("Executing Level-2 walk-forward training with 10-day purging and 10-day embargo...")
        
        for t in range(train_warmup, len(df)):
            # If no Level-1 trade is taken at t, we don't need to predict meta-probability
            if not is_trade[t]:
                continue
                
            # Define training mask:
            # - Must be before t
            # - Must be a trade row
            # - Must NOT overlap with the forward target of t
            # - Since target at t is log_return_10d, the target overlaps for t-10 to t.
            # - Enforce 10-day embargo: also purge t-20 to t-10.
            # - Therefore, training rows must be <= t - 21.
            train_mask = (np.arange(len(df)) <= t - 21) & is_trade
            
            X_train = X[train_mask]
            y_train = y_meta[train_mask]
            
            X_test = X[t:t+1]
            
            # Filter out NaN rows from training set
            non_nan_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            X_train = X_train[non_nan_mask]
            y_train = y_train[non_nan_mask]
            
            # If we don't have enough training samples or class representation
            if len(X_train) < 20 or len(np.unique(y_train)) < 2 or np.isnan(X_test).any():
                meta_probs[t] = 0.5 # Default to neutral
                continue
                
            # Fit scaler and scale features to prevent L2 regularisation scale distortion
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train independent models (HistGradientBoostingClassifier)
            meta_model = HistGradientBoostingClassifier(max_iter=50, max_depth=3, l2_regularization=10.0, learning_rate=0.05, random_state=42)
            meta_model.fit(X_train_scaled, y_train)
            
            meta_probs[t] = meta_model.predict_proba(X_test_scaled)[0][1]
            
        df['meta_prob'] = meta_probs
        
        # Save output meta predictions
        out_path = self.cache_dir / f"{ticker}_meta_predictions.csv"
        df.to_csv(out_path)
        logger.info(f"Saved Level-2 meta-predictions to {out_path} (shape: {df.shape})")
        
        return df

def generate_meta_signals():
    classifier = MetaLabelingClassifier()
    classifier.generate_meta_signals("QQQ")
    classifier.generate_meta_signals("SPY")

if __name__ == "__main__":
    generate_meta_signals()
