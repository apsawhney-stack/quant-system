"""
parameter_sweep.py - Multi-Factor Quant Research & Trading Platform
Executes a systematic walk-forward percentile parameter sweep
to identify the "convexity knee" and evaluate timing significance (p-values).
"""

import os
import logging
import numpy as np
import pandas as pd
import concurrent.futures
from data_pipeline import DataPipeline
from backtester import OptionsBacktester

# Silence verbose logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("quant_system.parameter_sweep")
logger.setLevel(logging.INFO)

logging.getLogger("quant_system.backtester").setLevel(logging.WARNING)
logging.getLogger("quant_system.data_pipeline").setLevel(logging.WARNING)
logging.getLogger("quant_system.risk_engine").setLevel(logging.WARNING)
logging.getLogger("quant_system.volatility_engine").setLevel(logging.WARNING)

_worker_pipeline = None

def run_single_permutation(shuffled_meta, underlying_df, vix_series, ticker, thresh, cache_dir):
    import logging
    logging.getLogger("quant_system.backtester").setLevel(logging.ERROR)
    logging.getLogger("quant_system.data_pipeline").setLevel(logging.ERROR)
    logging.getLogger("quant_system.risk_engine").setLevel(logging.ERROR)
    logging.getLogger("quant_system.volatility_engine").setLevel(logging.ERROR)
    logging.getLogger("quant_system.regime_classifier").setLevel(logging.ERROR)
    
    global _worker_pipeline
    if _worker_pipeline is None:
        _worker_pipeline = DataPipeline(cache_dir=cache_dir)
        
    backtester = OptionsBacktester(
        initial_capital=200000.0,
        max_position_risk_pct=0.01,
        max_portfolio_exposure_pct=0.25,
        commission_per_contract=0.65,
        slippage_pct_spread=0.50
    )
    res = backtester.run_backtest(
        underlying_df=underlying_df,
        vix_series=vix_series,
        meta_predictions_df=shuffled_meta,
        pipeline=_worker_pipeline,
        ticker=ticker,
        enable_regime=True,
        bullish_threshold=thresh,
        bearish_threshold=thresh,
        enable_expectancy_pause=False
    )
    return res.get("total_return_pct", 0.0) if res.get("status") != "NO_TRADES" else 0.0

def run_parameter_sweep(ticker: str):
    logger.info(f"🚀 Starting Quantile Parameter Sweep for {ticker}...")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "data_cache")
    pred_path = os.path.join(cache_dir, f"{ticker}_meta_predictions.csv")
    
    if not os.path.exists(pred_path):
        logger.error(f"Meta predictions file {pred_path} not found. Run backtest pipeline first.")
        return
        
    meta_predictions_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
    
    # Compute dates dynamically from prediction range
    start_date = meta_predictions_df.index.min().strftime("%Y-%m-%d")
    end_date = meta_predictions_df.index.max().strftime("%Y-%m-%d")
    
    pipeline = DataPipeline(cache_dir=cache_dir)
    backtester = OptionsBacktester(
        initial_capital=200000.0,
        max_position_risk_pct=0.01,
        max_portfolio_exposure_pct=0.25,
        commission_per_contract=0.65,
        slippage_pct_spread=0.50
    )
    
    underlying_df = pipeline.get_underlying_data(ticker, start_date, end_date)
    vix_series = pipeline.get_vix_data(start_date, end_date)
    
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]
    results = []
    
    print("\n" + "="*80)
    print(f"🔍 RUNNING SWEEP OVER PERCENTILES FOR {ticker}")
    print("="*80)
    print(f"{'Threshold':<10} | {'Trades':<6} | {'Win Rate':<8} | {'Return %':<8} | {'Max DD':<7} | {'Sharpe':<7} | {'p-value':<7}")
    print("-" * 80)
    
    for thresh in thresholds:
        # 1. Run baseline backtest with current threshold
        res = backtester.run_backtest(
            underlying_df=underlying_df,
            vix_series=vix_series,
            meta_predictions_df=meta_predictions_df,
            pipeline=pipeline,
            ticker=ticker,
            enable_regime=True,
            bullish_threshold=thresh,
            bearish_threshold=thresh,
            enable_expectancy_pause=False
        )
        
        if res.get("status") == "NO_TRADES":
            print(f"{thresh:<10.2f} | {'0':<6} | {'N/A':<8} | {'0.00%':<8} | {'0.00%':<7} | {'N/A':<7} | {'N/A':<7}")
            continue
            
        ret_pct = res.get("total_return_pct", 0.0)
        trades = res.get("total_trades", 0)
        win_rate = res.get("win_rate", 0.0)
        max_dd = res.get("max_drawdown_pct", 0.0)
        sharpe = res.get("sharpe_ratio", 0.0)
        
        # 2. Run 200 temporal permutations to compute the out-of-sample timing p-value in parallel (C14)
        num_perms = 200
        better_count = 0
        
        permutations_args = []
        for _ in range(num_perms):
            shuffled_meta = meta_predictions_df.sample(frac=1.0).copy()
            shuffled_meta.index = meta_predictions_df.index
            shuffled_meta = shuffled_meta.sort_index()
            permutations_args.append(shuffled_meta)
            
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(
                    run_single_permutation,
                    shm,
                    underlying_df,
                    vix_series,
                    ticker,
                    thresh,
                    cache_dir
                )
                for shm in permutations_args
            ]
            for fut in concurrent.futures.as_completed(futures):
                try:
                    p_ret = fut.result()
                    if p_ret >= ret_pct:
                        better_count += 1
                except Exception as e:
                    logger.error(f"Error in permutation worker: {e}")
                    
        p_val = better_count / num_perms
        
        print(f"{thresh:<10.2f} | {trades:<6} | {win_rate:<7.2f}% | {ret_pct:<7.2f}% | {max_dd:<6.2f}% | {sharpe:<7.4f} | {p_val:<7.2f}")
        
        results.append({
            "Threshold": thresh,
            "Trades": trades,
            "Win Rate %": win_rate,
            "Return %": ret_pct,
            "Max DD %": max_dd,
            "Sharpe": sharpe,
            "p-value": p_val
        })
        
    print("="*80 + "\n")
    return results

if __name__ == "__main__":
    for ticker in ["QQQ", "SPY"]:
        run_parameter_sweep(ticker)
