"""
empirical_validation.py - Multi-Factor Quant Research & Trading Platform
Executes strict empirical validation, stress testing, and diagnostics on options backtests.
"""

import os
import logging
import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from data_pipeline import DataPipeline
from backtester import OptionsBacktester

# Set up logging
logger = logging.getLogger("quant_system.empirical_validation")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Computes Brier Score and Expected Calibration Error (ECE) for probability validation.
    """
    # Remove NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = np.array(y_true)[mask]
    y_prob = np.array(y_prob)[mask]
    
    if len(y_true) == 0:
        return np.nan, np.nan
        
    # Brier Score
    brier_score = np.mean((y_prob - y_true) ** 2)
    
    # ECE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_prob)
    
    for m in range(n_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        # Find indices of samples falling into current bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if m < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_bin = np.mean(in_bin)
        
        if prop_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += prop_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
    return brier_score, ece

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

def load_data(ticker: str, pipeline: DataPipeline):
    # Resolve dynamic absolute paths relative to script file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "data_cache")
    pred_path = os.path.join(cache_dir, f"{ticker}_meta_predictions.csv")
    
    if os.path.exists(pred_path):
        meta_predictions_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
    else:
        logger.warning(f"No meta prediction cache found for {ticker}. Running pipeline to generate it...")
        from kronos_signal import StackedSignalEngine
        from meta_classifier import MetaLabelingClassifier
        
        l1 = StackedSignalEngine(cache_dir=cache_dir)
        l1.generate_calibrated_signals(ticker)
        l2 = MetaLabelingClassifier(cache_dir=cache_dir)
        l2.generate_meta_signals(ticker)
        meta_predictions_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
        
    # Compute dynamic start and end dates based on predictions data range
    start_date = meta_predictions_df.index.min().strftime("%Y-%m-%d")
    end_date = meta_predictions_df.index.max().strftime("%Y-%m-%d")
    
    underlying_df = pipeline.get_underlying_data(ticker, start_date, end_date)
    vix_series = pipeline.get_vix_data(start_date, end_date)
    
    return underlying_df, vix_series, meta_predictions_df

def run_exposure_stress_tests(ticker: str, underlying_df, vix_series, meta_predictions_df, pipeline):
    logger.info(f"\n--- 📈 Running Exposure Sizing Stress Tests for {ticker} ---")
    
    # Test different risk limits per trade, scaling the exposure ceiling accordingly
    scenarios = [
        {"name": "1% Baseline", "risk_pct": 0.01, "exposure_pct": 0.05},
        {"name": "5% Risk", "risk_pct": 0.05, "exposure_pct": 0.25},
        {"name": "10% Risk", "risk_pct": 0.10, "exposure_pct": 0.50},
        {"name": "15% Risk", "risk_pct": 0.15, "exposure_pct": 0.75},
        {"name": "25% Risk", "risk_pct": 0.25, "exposure_pct": 0.90}
    ]
    
    results = []
    trades_5pct = []
    
    for sc in scenarios:
        backtester = OptionsBacktester(
            initial_capital=200000.0,
            max_position_risk_pct=sc["risk_pct"],
            max_portfolio_exposure_pct=sc["exposure_pct"]
        )
        
        res = backtester.run_backtest(
            underlying_df=underlying_df,
            vix_series=vix_series,
            meta_predictions_df=meta_predictions_df,
            pipeline=pipeline,
            ticker=ticker,
            enable_regime=True,
            enable_expectancy_pause=False
        )
        
        if res.get("status") == "NO_TRADES":
            logger.warning(f"No trades executed for {sc['name']}")
            continue
            
        if sc["risk_pct"] == 0.05:
            trades_5pct = list(backtester.trade_log)
            
        results.append({
            "Scenario": sc["name"],
            "Risk Per Trade": f"{sc['risk_pct']*100}%",
            "Max Exposure": f"{sc['exposure_pct']*100}%",
            "Total Trades": res["total_trades"],
            "Win Rate": f"{res['win_rate']:.2f}%",
            "Return %": f"{res['total_return_pct']:.2f}%",
            "Max DD": f"{res['max_drawdown_pct']:.2f}%",
            "Sharpe": f"{res['sharpe_ratio']:.4f}",
            "Sortino": f"{res['sortino_ratio']:.4f}",
            "Avg Margin Util": f"{res['avg_portfolio_utilization_pct']:.2f}%",
            "Profit Factor": f"{res['profit_factor']:.4f}"
        })
        
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print(f"📊 EXPOSURE STRESS TEST RESULTS FOR {ticker}")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80 + "\n")
    return trades_5pct

def run_trade_level_attribution(trades: List[Dict[str, Any]], ticker: str):
    logger.info(f"--- 🔍 Running Trade-Level Strategy Attribution for {ticker} ---")
    
    if not trades:
        logger.warning("No trades logged for attribution.")
        return pd.DataFrame()
        
    df_trades = pd.DataFrame(trades)
    
    # Group by strategy
    attribution = []
    for strat, group in df_trades.groupby("strategy"):
        wins = group[group["net_pnl"] > 0]
        win_rate = len(wins) / len(group) * 100
        total_pnl = group["net_pnl"].sum()
        avg_pnl = group["net_pnl"].mean()
        avg_r = group["r_multiple"].mean()
        
        gross_profits = wins["net_pnl"].sum()
        gross_losses = abs(group[group["net_pnl"] < 0]["net_pnl"].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        attribution.append({
            "Strategy": strat,
            "Trades": len(group),
            "Win Rate": f"{win_rate:.2f}%",
            "Total PnL": f"${total_pnl:,.2f}",
            "Avg PnL": f"${avg_pnl:,.2f}",
            "Avg R-Mult": f"{avg_r:.4f}R",
            "Profit Factor": f"{profit_factor:.4f}"
        })
        
    df_attr = pd.DataFrame(attribution)
    print("="*80)
    print(f"📊 STRATEGY ATTRIBUTION FOR {ticker} (5% Risk Sizing)")
    print("="*80)
    print(df_attr.to_string(index=False))
    print("="*80 + "\n")
    return df_trades

def run_monte_carlo_resampling(ticker: str, df_trades: pd.DataFrame, num_simulations: int = 10000):
    logger.info(f"--- 🎲 Running Monte Carlo Sequence Resampling (10,000 runs) for {ticker} ---")
    
    if df_trades.empty or len(df_trades) < 5:
        logger.warning("Insufficient trades to execute Monte Carlo bootstrap resampling.")
        return
        
    pnl_series = df_trades["net_pnl"].values
    r_multiples = df_trades["r_multiple"].values
    
    max_drawdowns = []
    ruined_count = 0
    initial_capital = 200000.0
    ruin_threshold = initial_capital * 0.50 # 50% loss ($100k)
    
    # Circular Block Bootstrap parameters
    block_size = 5
    n_trades = len(pnl_series)
    
    for _ in range(num_simulations):
        # Circular block bootstrap sampling to preserve regime dependency and serial correlation
        bootstrap_pnl = []
        while len(bootstrap_pnl) < n_trades:
            # Pick a random trade index
            start_idx = np.random.randint(0, n_trades)
            # Create a block of contiguous indices wrapping circularly
            block_indices = np.arange(start_idx, start_idx + block_size) % n_trades
            bootstrap_pnl.extend(pnl_series[block_indices])
            
        bootstrap_pnl = np.array(bootstrap_pnl[:n_trades])
        
        # Build equity curve
        equity = initial_capital + np.cumsum(bootstrap_pnl)
        
        # Check for ruin
        if np.any(equity <= ruin_threshold):
            ruined_count += 1
            
        # Drawdown calculation
        peaks = np.maximum.accumulate(equity)
        drawdowns = (equity - peaks) / peaks
        max_drawdowns.append(drawdowns.min() * 100.0)
        
    max_drawdowns = np.array(max_drawdowns)
    prob_ruin = (ruined_count / num_simulations) * 100.0
    median_dd = np.median(max_drawdowns)
    worst_5pct_dd = np.percentile(max_drawdowns, 5) # 5th percentile (worst drawdowns are negative, so 5th percentile is the tail-risk value)
    
    print("="*80)
    print(f"🎲 MONTE CARLO ANALYSIS RESULTS FOR {ticker}")
    print("="*80)
    print(f"Simulations Run:           {num_simulations:,}")
    print(f"Probability of Ruin (50%): {prob_ruin:.4f}%")
    print(f"Median Max Drawdown:       {median_dd:.2f}%")
    print(f"95% Value-at-Risk Drawdown: {worst_5pct_dd:.2f}%")
    print("="*80 + "\n")

def run_regime_crisis_period_attribution(trades: List[Dict[str, Any]], ticker: str):
    logger.info(f"--- ⚠️ Running Crisis-Period & Regime Attribution for {ticker} ---")
    
    if not trades:
        return
        
    df_trades = pd.DataFrame(trades)
    df_trades['entry_date'] = pd.to_datetime(df_trades['entry_date'])
    df_trades['exit_date'] = pd.to_datetime(df_trades['exit_date'])
    
    # 2022 Crisis Period
    df_2022 = df_trades[(df_trades['entry_date'] >= '2022-01-01') & (df_trades['entry_date'] <= '2022-12-31')]
    # Other Periods
    df_others = df_trades[(df_trades['entry_date'] < '2022-01-01') | (df_trades['entry_date'] > '2022-12-31')]
    
    periods = [
        ("2022 Bear Market", df_2022),
        ("Other Periods", df_others)
    ]
    
    print("="*80)
    print(f"📊 CRISIS-PERIOD PERFORMANCE ATTRIBUTION FOR {ticker}")
    print("="*80)
    for name, df_p in periods:
        if df_p.empty:
            print(f"{name:<25} | No trades executed.")
            continue
        wins = df_p[df_p['net_pnl'] > 0]
        win_rate = len(wins) / len(df_p) * 100
        total_pnl = df_p['net_pnl'].sum()
        avg_r = df_p['r_multiple'].mean()
        print(f"{name:<25} | Trades: {len(df_p):<3} | Win Rate: {win_rate:.2f}% | Total PnL: ${total_pnl:,.2f} | Avg R: {avg_r:.4f}R")
    print("="*80 + "\n")

def run_integrity_leakage_audit(ticker: str, underlying_df, meta_predictions_df):
    logger.info("--- 🔒 Running Integrity Leakage Audit ---")
    
    # Check if index matches underlying price index
    diff_index = meta_predictions_df.index.difference(underlying_df.index)
    if not diff_index.empty:
        logger.warning(f"Index mismatch found: {len(diff_index)} prediction index dates do not exist in price data.")
        
    # Check for direct lookahead bias: is today's score correlated with tomorrow's actual return?
    # Shift daily returns back to match them with the prediction score at time t
    fwd_return = underlying_df['close'].pct_change(10).shift(-10) # 10-day forward return
    
    df_audit = pd.DataFrame({
        "pred_prob_bull": meta_predictions_df['pred_prob_bull'],
        "pred_prob_bear": meta_predictions_df['pred_prob_bear'],
        "fwd_return": fwd_return
    }).dropna()
    
    corr_bull = df_audit["pred_prob_bull"].corr(df_audit["fwd_return"])
    corr_bear = df_audit["pred_prob_bear"].corr(df_audit["fwd_return"])
    logger.info(f"Correlation between pred_prob_bull and true 10-day forward return: {corr_bull:.4f}")
    logger.info(f"Correlation between pred_prob_bear and true 10-day forward return: {corr_bear:.4f}")
    
    # Warning sign if prediction score is perfectly correlated with future daily returns
    tomorrow_return = underlying_df['close'].pct_change().shift(-1)
    df_leak = pd.DataFrame({
        "pred_prob_bull": meta_predictions_df['pred_prob_bull'],
        "tomorrow_return": tomorrow_return
    }).dropna()
    leak_corr = df_leak["pred_prob_bull"].corr(df_leak["tomorrow_return"])
    
    if abs(leak_corr) > 0.85:
        logger.critical(f"CRITICAL WARNING: Direct future return leakage suspected (corr = {leak_corr:.4f} > 0.85)!")
    else:
        logger.info(f"Direct future return leakage test passed (corr = {leak_corr:.4f}).")

def run_calibration_and_edge_decay_diagnostics(ticker: str, meta_predictions_df: pd.DataFrame):
    logger.info(f"\n--- 📊 Running Calibration and Edge Decay Diagnostics for {ticker} ---")
    
    # 1. Calibration Metrics (Brier Score and Expected Calibration Error)
    if 'target_bullish' in meta_predictions_df.columns and 'pred_prob_bull' in meta_predictions_df.columns:
        bull_brier, bull_ece = calculate_calibration_metrics(
            meta_predictions_df['target_bullish'], 
            meta_predictions_df['pred_prob_bull']
        )
        logger.info(f"Bullish Classifier Calibration: Brier Score = {bull_brier:.4f}, ECE = {bull_ece:.4f}")
    else:
        logger.warning("Bullish calibration metrics skipped: target_bullish or pred_prob_bull not in columns.")
        
    if 'target_bearish' in meta_predictions_df.columns and 'pred_prob_bear' in meta_predictions_df.columns:
        bear_brier, bear_ece = calculate_calibration_metrics(
            meta_predictions_df['target_bearish'], 
            meta_predictions_df['pred_prob_bear']
        )
        logger.info(f"Bearish Classifier Calibration: Brier Score = {bear_brier:.4f}, ECE = {bear_ece:.4f}")
    else:
        logger.warning("Bearish calibration metrics skipped: target_bearish or pred_prob_bear not in columns.")
        
    # 2. Non-linear Edge Decay Curves
    df = meta_predictions_df.copy()
    if 'fwd_log_ret_10d' not in df.columns:
        df['fwd_log_ret_10d'] = np.log(df['close'].shift(-10) / df['close'])
        
    df = df.dropna(subset=['pred_prob_bull', 'pred_prob_bear', 'fwd_log_ret_10d'])
    
    if len(df) < 50:
        logger.warning("Insufficient samples to run Edge Decay analysis.")
        return
        
    def pct_rank(window):
        if len(window) < 2:
            return np.nan
        current_val = window[-1]
        less_than_count = np.sum(window[:-1] < current_val)
        return less_than_count / (len(window) - 1)
        
    df['prob_bull_pct'] = df['pred_prob_bull'].rolling(window=252, min_periods=100).apply(pct_rank, raw=True)
    df['prob_bear_pct'] = df['pred_prob_bear'].rolling(window=252, min_periods=100).apply(pct_rank, raw=True)
    
    df = df.dropna(subset=['prob_bull_pct', 'prob_bear_pct']).copy()
    
    # Buckets formatting
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
    
    # Bullish Edge Decay
    bull_results = []
    grouped_bull = df.groupby('bull_bucket')
    mid_bull = grouped_bull.get_group("Middle 80%") if "Middle 80%" in grouped_bull.groups else None
    
    for b in bucket_order:
        if b in grouped_bull.groups:
            group = grouped_bull.get_group(b)
            avg_ret = group['fwd_log_ret_10d'].mean() * 100.0
            count = len(group)
            t_stat = 0.0
            if mid_bull is not None and b != "Middle 80%":
                mean_diff = avg_ret - (mid_bull['fwd_log_ret_10d'].mean() * 100.0)
                se_group = newey_west_se(group['fwd_log_ret_10d'], max_lag=9) * 100.0
                se_mid = newey_west_se(mid_bull['fwd_log_ret_10d'], max_lag=9) * 100.0
                se = np.sqrt(se_group**2 + se_mid**2)
                t_stat = mean_diff / se if se > 0 else 0.0
            bull_results.append({
                "Bucket": b, "Trades": count, "Avg Return %": f"{avg_ret:.4f}%", "t-stat vs Mid": f"{t_stat:.4f}"
            })
            
    # Bearish Edge Decay
    bear_results = []
    grouped_bear = df.groupby('bear_bucket')
    mid_bear = grouped_bear.get_group("Middle 80%") if "Middle 80%" in grouped_bear.groups else None
    
    for b in bucket_order:
        if b in grouped_bear.groups:
            group = grouped_bear.get_group(b)
            avg_ret = group['fwd_log_ret_10d'].mean() * 100.0
            count = len(group)
            t_stat = 0.0
            if mid_bear is not None and b != "Middle 80%":
                mean_diff = avg_ret - (mid_bear['fwd_log_ret_10d'].mean() * 100.0)
                se_group = newey_west_se(group['fwd_log_ret_10d'], max_lag=9) * 100.0
                se_mid = newey_west_se(mid_bear['fwd_log_ret_10d'], max_lag=9) * 100.0
                se = np.sqrt(se_group**2 + se_mid**2)
                t_stat = mean_diff / se if se > 0 else 0.0
            bear_results.append({
                "Bucket": b, "Trades": count, "Avg Return %": f"{avg_ret:.4f}%", "t-stat vs Mid": f"{t_stat:.4f}"
            })
            
    print("\n" + "="*80)
    print(f"📊 BULLISH CLASSIFIER EDGE DECAY FOR {ticker}")
    print("="*80)
    print(pd.DataFrame(bull_results).to_string(index=False))
    print("="*80 + "\n")
    
    print("="*80)
    print(f"📊 BEARISH CLASSIFIER EDGE DECAY FOR {ticker}")
    print("="*80)
    print(pd.DataFrame(bear_results).to_string(index=False))
    print("="*80 + "\n")

def run_validation_pipeline():
    logger.info("🚀 Starting Empirical Validation Test Harness...")
    pipeline = DataPipeline(cache_dir="./data_cache")
    
    for ticker in ["QQQ", "SPY"]:
        underlying_df, vix_series, meta_predictions_df = load_data(ticker, pipeline)
        
        # 1. Run Probability Calibration & Edge Decay Diagnostics
        run_calibration_and_edge_decay_diagnostics(ticker, meta_predictions_df)
        
        # 2. Run Exposure Sizing Stress Tests
        trades_5pct = run_exposure_stress_tests(ticker, underlying_df, vix_series, meta_predictions_df, pipeline)
        
        # 3. Run Trade-Level Attribution
        df_trades = run_trade_level_attribution(trades_5pct, ticker)
        
        # 4. Monte Carlo sequence resampling
        if df_trades is not None and not df_trades.empty:
            run_monte_carlo_resampling(ticker, df_trades)
            
        # 5. Regime & Crisis-Period Attribution
        run_regime_crisis_period_attribution(trades_5pct, ticker)
        
        # 6. Integrity Leakage Audit
        run_integrity_leakage_audit(ticker, underlying_df, meta_predictions_df)

if __name__ == "__main__":
    run_validation_pipeline()
