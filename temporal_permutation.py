import os
import sys
import pandas as pd
import numpy as np
import logging
import concurrent.futures
from data_pipeline import DataPipeline
from backtester import OptionsBacktester

# Silence verbose logging
logging.basicConfig(level=logging.WARNING)

_worker_pipeline = None

def run_single_permutation(shuffled_meta, underlying_df, vix_series, cache_dir):
    """
    Worker function to execute a single scrambled backtest permutation.
    Defined at module level to allow multiprocessing pickling.
    """
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
        max_portfolio_exposure_pct=0.05
    )
    res = backtester.run_backtest(
        underlying_df=underlying_df,
        vix_series=vix_series,
        meta_predictions_df=shuffled_meta,
        pipeline=_worker_pipeline,
        ticker="QQQ",
        enable_regime=True,
        enable_expectancy_pause=False
    )
    status = res.get("status")
    return {
        "status": status,
        "total_return_pct": res.get("total_return_pct", 0.0) if status != "NO_TRADES" else 0.0,
        "total_trades": res.get("total_trades", 0)
    }

def run_temporal_permutation_test(num_permutations: int = 1000):
    print(f"🚀 Running Parallel Temporal Permutation Test for QQQ ({num_permutations} runs)...")
    
    # Silence detailed system logging to prevent excessive log writing overhead
    logging.getLogger("quant_system.backtester").setLevel(logging.WARNING)
    logging.getLogger("quant_system.data_pipeline").setLevel(logging.WARNING)
    logging.getLogger("quant_system.risk_engine").setLevel(logging.WARNING)
    logging.getLogger("quant_system.volatility_engine").setLevel(logging.WARNING)
    
    # Resolve dynamic paths relative to file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "data_cache")
    pred_path = os.path.join(cache_dir, "QQQ_meta_predictions.csv")
    
    if not os.path.exists(pred_path):
        print(f"Meta predictions file {pred_path} not found. Run backtest pipeline first.")
        return
        
    meta_predictions_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
    
    # Determine dates dynamically from predictions
    start_date = meta_predictions_df.index.min().strftime("%Y-%m-%d")
    end_date = meta_predictions_df.index.max().strftime("%Y-%m-%d")
    
    pipeline = DataPipeline(cache_dir=cache_dir)
    underlying_df = pipeline.get_underlying_data("QQQ", start_date, end_date)
    vix_series = pipeline.get_vix_data(start_date, end_date)
    
    # 1. Run Baseline (Unscrambled)
    backtester = OptionsBacktester(
        initial_capital=200000.0,
        max_position_risk_pct=0.01,
        max_portfolio_exposure_pct=0.05
    )
    res_baseline = backtester.run_backtest(
        underlying_df=underlying_df,
        vix_series=vix_series,
        meta_predictions_df=meta_predictions_df,
        pipeline=pipeline,
        ticker="QQQ",
        enable_regime=True,
        enable_expectancy_pause=False
    )
    baseline_return = res_baseline.get("total_return_pct", 0.0) if res_baseline.get("status") != "NO_TRADES" else 0.0
    baseline_trades = res_baseline.get("total_trades", 0)
    print(f"Baseline Unscrambled Return: {baseline_return:.4f}% (Trades: {baseline_trades})")
    
    # 2. Run Permutations (Scrambled) in parallel
    permuted_returns = []
    permuted_trades = []
    
    permutations_args = []
    for _ in range(num_permutations):
        shuffled_meta = meta_predictions_df.sample(frac=1.0).copy()
        shuffled_meta.index = meta_predictions_df.index
        shuffled_meta = shuffled_meta.sort_index()
        permutations_args.append(shuffled_meta)
        
    print(f"Executing permutations using ProcessPoolExecutor...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_single_permutation,
                shm,
                underlying_df,
                vix_series,
                cache_dir
            )
            for shm in permutations_args
        ]
        
        for fut in concurrent.futures.as_completed(futures):
            try:
                res = fut.result()
                permuted_returns.append(res["total_return_pct"])
                permuted_trades.append(res["total_trades"])
            except Exception as e:
                # Log error and ignore failed permutation runs
                pass
                
    permuted_returns = np.array(permuted_returns)
    permuted_trades = np.array(permuted_trades)
    
    if len(permuted_returns) == 0:
        print("Error: All permutation runs failed. Cannot calculate p-value.")
        return
        
    print("\n" + "="*80)
    print("📊 TEMPORAL PERMUTATION TEST RESULTS FOR QQQ")
    print("="*80)
    print(f"Baseline Return:            {baseline_return:.4f}%")
    print(f"Baseline Trade Count:       {baseline_trades}")
    print(f"Mean Permuted Return:       {permuted_returns.mean():.4f}%")
    print(f"Median Permuted Return:     {np.median(permuted_returns):.4f}%")
    print(f"Std Dev of Permuted Return: {permuted_returns.std():.4f}%")
    print(f"Mean Permuted Trade Count:  {permuted_trades.mean():.1f}")
    
    # Calculate p-value: fraction of scrambled runs that outperformed or equaled baseline
    better_count = np.sum(permuted_returns >= baseline_return)
    p_val = better_count / len(permuted_returns)
    print(f"Empirical p-value:          {p_val:.4f} ({better_count}/{len(permuted_returns)} runs)")
    print("="*80 + "\n")
    
    if p_val > 0.05:
        print("Conclusion: The unscrambled signal return is NOT statistically distinguishable from random temporal alignment (p > 0.05).")
    else:
        print("Conclusion: The unscrambled signal displays statistical timing edge compared to scrambled noise (p <= 0.05).")

if __name__ == "__main__":
    run_temporal_permutation_test(num_permutations=1000)
