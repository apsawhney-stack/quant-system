"""
plot_edge_decay.py - Multi-Factor Quant Research & Trading Platform
Standalone edge decay curve analysis for newly transformed probability signals.
"""

import os
import pandas as pd
import numpy as np

def newey_west_se(series: pd.Series, max_lag: int = 9) -> float:
    """
    Computes the autocorrelation-adjusted standard error of the mean
    using the Newey-West (Bartlett kernel) formula to correct for overlapping returns.
    """
    n = len(series)
    if n <= 1:
        return 0.0
    
    # Center series values
    x = series.values - series.mean()
    
    # Lag 0 autocovariance (variance)
    gamma = np.var(x, ddof=1)
    
    # Add weighted lag covariances
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        # Sample covariance at lag
        cov = np.sum(x[lag:] * x[:-lag]) / (n - 1)
        # Bartlett kernel weight
        weight = 1.0 - (lag / (max_lag + 1))
        gamma += 2.0 * weight * cov
        
    # Standard error of the mean
    return np.sqrt(max(gamma, 0.0) / n)

def run_edge_decay_analysis(ticker: str):
    print(f"\n🚀 Running Edge Decay Curve Analysis for {ticker} (Calibrated Probabilities)...")
    
    # Resolve dynamic absolute paths relative to script file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pred_path = os.path.join(base_dir, "data_cache", f"{ticker}_meta_predictions.csv")
    if not os.path.exists(pred_path):
        print(f"Meta-prediction file {pred_path} not found. Run backtest pipeline first.")
        return
        
    df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
    
    # Ensure forward returns are computed
    if 'fwd_log_ret_10d' not in df.columns:
        df['fwd_log_ret_10d'] = np.log(df['close'].shift(-10) / df['close'])
        
    df = df.dropna(subset=['pred_prob_bull', 'pred_prob_bear', 'fwd_log_ret_10d']).copy()
    
    if len(df) < 50:
        print("Insufficient samples to run Edge Decay analysis.")
        return
        
    def pct_rank(window):
        if len(window) < 2:
            return np.nan
        current_val = window[-1]
        less_than_count = np.sum(window[:-1] < current_val)
        return less_than_count / (len(window) - 1)

    # Calculate backward-looking rolling 252-day percentiles for probabilities
    print("Calculating rolling percentiles (this may take a few seconds)...")
    df['prob_bull_pct'] = df['pred_prob_bull'].rolling(252, min_periods=100).apply(pct_rank, raw=True)
    df['prob_bear_pct'] = df['pred_prob_bear'].rolling(252, min_periods=100).apply(pct_rank, raw=True)
    
    df = df.dropna(subset=['prob_bull_pct', 'prob_bear_pct']).copy()
    
    # Define signal buckets
    def get_bucket(pct):
        if pct >= 0.99:
            return "Top 1%"
        elif pct >= 0.95:
            return "Top 5%"
        elif pct >= 0.90:
            return "Top 10%"
        elif pct >= 0.80:
            return "Top 20%"
        else:
            return "Middle 80%"
            
    df['bull_bucket'] = df['prob_bull_pct'].apply(get_bucket)
    df['bear_bucket'] = df['prob_bear_pct'].apply(get_bucket)
    
    bucket_order = ["Top 1%", "Top 5%", "Top 10%", "Top 20%", "Middle 80%"]
    
    # 1. Bullish Classifier Edge Decay
    bull_results = []
    grouped_bull = df.groupby('bull_bucket')
    mid_bull = grouped_bull.get_group("Middle 80%") if "Middle 80%" in grouped_bull.groups else None
    
    for b in bucket_order:
        if b in grouped_bull.groups:
            group = grouped_bull.get_group(b)
            avg_ret = group['fwd_log_ret_10d'].mean() * 100.0
            std_ret = group['fwd_log_ret_10d'].std() * 100.0
            count = len(group)
            
            t_stat = 0.0
            if mid_bull is not None and b != "Middle 80%":
                mean_diff = avg_ret - (mid_bull['fwd_log_ret_10d'].mean() * 100.0)
                se_group = newey_west_se(group['fwd_log_ret_10d'], max_lag=9) * 100.0
                se_mid = newey_west_se(mid_bull['fwd_log_ret_10d'], max_lag=9) * 100.0
                se = np.sqrt(se_group**2 + se_mid**2)
                t_stat = mean_diff / se if se > 0 else 0.0
                
            bull_results.append({
                "Bucket": b,
                "Trades": count,
                "Avg 10d Fwd Return %": f"{avg_ret:.4f}%",
                "Std Dev %": f"{std_ret:.4f}%",
                "t-stat vs Mid 80%": f"{t_stat:.4f}"
            })
            
    # 2. Bearish Classifier Edge Decay
    bear_results = []
    grouped_bear = df.groupby('bear_bucket')
    mid_bear = grouped_bear.get_group("Middle 80%") if "Middle 80%" in grouped_bear.groups else None
    
    for b in bucket_order:
        if b in grouped_bear.groups:
            group = grouped_bear.get_group(b)
            avg_ret = group['fwd_log_ret_10d'].mean() * 100.0
            std_ret = group['fwd_log_ret_10d'].std() * 100.0
            count = len(group)
            
            t_stat = 0.0
            if mid_bear is not None and b != "Middle 80%":
                mean_diff = avg_ret - (mid_bear['fwd_log_ret_10d'].mean() * 100.0)
                se_group = newey_west_se(group['fwd_log_ret_10d'], max_lag=9) * 100.0
                se_mid = newey_west_se(mid_bear['fwd_log_ret_10d'], max_lag=9) * 100.0
                se = np.sqrt(se_group**2 + se_mid**2)
                t_stat = mean_diff / se if se > 0 else 0.0
                
            bear_results.append({
                "Bucket": b,
                "Trades": count,
                "Avg 10d Fwd Return %": f"{avg_ret:.4f}%",
                "Std Dev %": f"{std_ret:.4f}%",
                "t-stat vs Mid 80%": f"{t_stat:.4f}"
            })
            
    df_bull = pd.DataFrame(bull_results)
    print("\n" + "="*80)
    print(f"📊 BULLISH CLASSIFIER EDGE DECAY FOR {ticker}")
    print("="*80)
    print(df_bull.to_string(index=False))
    print("="*80 + "\n")
    
    df_bear = pd.DataFrame(bear_results)
    print("="*80)
    print(f"📊 BEARISH CLASSIFIER EDGE DECAY FOR {ticker}")
    print("="*80)
    print(df_bear.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    for ticker in ["QQQ", "SPY"]:
        run_edge_decay_analysis(ticker)
