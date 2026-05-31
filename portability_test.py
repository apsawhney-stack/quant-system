"""
portability_test.py - Multi-Factor Quant Research & Trading Platform
Executes multi-asset portability tests on IWM, GLD, and TLT.
Orchestrates stacked classifier models and runs comparative backtests.
"""

import os
import logging
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from data_pipeline import DataPipeline
from regime_classifier import RegimeClassifier
from volatility_engine import VolatilityEngine
from backtester import OptionsBacktester
from kronos_signal import StackedSignalEngine
from meta_classifier import MetaLabelingClassifier

# Set up logging
logger = logging.getLogger("quant_system.portability_test")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def generate_mock_predictions(ticker: str, start_date: str, end_date: str, cache_dir: Path):
    """
    Generates realistic, strictly backward-looking mock predictions for assets without fine-tuned models.
    Uses trailing 20-day returns to estimate 10-day forward return to ensure no look-ahead bias.
    """
    pipeline = DataPipeline(cache_dir=str(cache_dir))
    df_stock = pipeline.get_underlying_data(ticker, start_date, end_date)
    
    # Calculate trailing 20-day mean log return
    log_close = np.log(df_stock['close'])
    rolling_mean_ret = log_close.diff(1).rolling(20).mean()
    predicted_return = rolling_mean_ret * 10.0 # Scale to 10-day forward projection
    
    predicted_close_10d = df_stock['close'] * np.exp(predicted_return)
    
    df_pred = pd.DataFrame({
        'date': df_stock.index.strftime('%Y-%m-%d'),
        'current_close': df_stock['close'].values,
        'predicted_close_10d': predicted_close_10d.values,
        'predicted_return': (predicted_close_10d - df_stock['close']).values / df_stock['close'].values
    })
    # Scale score to [-1.0, 1.0] by dividing by 5% baseline target return
    df_pred['kronos_score'] = (df_pred['predicted_return'] / 0.05).clip(-1.0, 1.0)
    
    out_path = cache_dir / f"{ticker}_predictions.csv"
    df_pred.to_csv(out_path, index=False)
    logger.info(f"Generated zero-lookahead mock predictions for {ticker} at {out_path}")

def run_portability_for_ticker(ticker: str):
    logger.info(f"\n==================================================")
    logger.info(f"🌎 RUNNING PORTABILITY TEST FOR {ticker}")
    logger.info(f"==================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = Path(os.path.join(base_dir, "data_cache"))
    start_date = "2021-05-31"
    end_date = "2026-05-22"
    
    # Ensure Level-0 predictions exist
    pred_path = cache_dir / f"{ticker}_predictions.csv"
    if not pred_path.exists():
        generate_mock_predictions(ticker, start_date, end_date, cache_dir)
        
    pipeline = DataPipeline(cache_dir=str(cache_dir))
    backtester = OptionsBacktester(
        initial_capital=200000.0,
        max_position_risk_pct=0.01,
        max_portfolio_exposure_pct=0.25,
        commission_per_contract=0.65,
        slippage_pct_spread=0.50
    )
    
    # 1. Run Level-1 stacked classifier models
    logger.info(f"Executing Level-1 Stacked Signal Engine for {ticker}...")
    l1_engine = StackedSignalEngine(cache_dir=str(cache_dir))
    l1_engine.generate_calibrated_signals(ticker)
        
    # 2. Run Level-2 meta-classifier model
    logger.info(f"Executing Level-2 Meta-Labeling Classifier for {ticker}...")
    l2_engine = MetaLabelingClassifier(cache_dir=str(cache_dir))
    l2_engine.generate_meta_signals(ticker)
        
    # 3. Load Level-2 meta-predictions
    meta_path = cache_dir / f"{ticker}_meta_predictions.csv"
    if not meta_path.exists():
        logger.error(f"Meta predictions file {meta_path} not found! Skipping {ticker} portability test.")
        return
        
    meta_predictions_df = pd.read_csv(meta_path, parse_dates=True, index_col=0)
    
    # Compute dates dynamically from predictions index
    start_date = meta_predictions_df.index.min().strftime("%Y-%m-%d")
    end_date = meta_predictions_df.index.max().strftime("%Y-%m-%d")
    
    # 4. Load market data
    underlying_df = pipeline.get_underlying_data(ticker, start_date, end_date)
    vix_series = pipeline.get_vix_data(start_date, end_date)
    
    # 5. Run Full System
    logger.info("Executing simulation loop WITH VIX Regime Classifier...")
    results_full = backtester.run_backtest(
        underlying_df=underlying_df,
        vix_series=vix_series,
        meta_predictions_df=meta_predictions_df,
        pipeline=pipeline,
        ticker=ticker,
        enable_regime=True
    )
    
    # 6. Run Ablated System
    logger.info("Executing simulation loop WITHOUT VIX Regime Classifier (Ablation)...")
    results_ablated = backtester.run_backtest(
        underlying_df=underlying_df,
        vix_series=vix_series,
        meta_predictions_df=meta_predictions_df,
        pipeline=pipeline,
        ticker=ticker,
        enable_regime=False
    )
    
    # 7. Print results
    print(f"\n==================================================")
    print(f"📊 {ticker} PORTABILITY TEST COMPARISON")
    print(f"==================================================")
    print(f"{'Metric':<30} | {'Full (With VIX)':<15} | {'Ablated (No VIX)':<15}")
    print(f"-" * 68)
    
    metrics = [
        ("Total Trades", "total_trades", False, ""),
        ("Win Rate", "win_rate", True, "%"),
        ("Total PnL", "total_pnl", False, "$"),
        ("Total Return", "total_return_pct", True, "%"),
        ("Avg R-Multiple", "avg_r_multiple", False, "R"),
        ("Avg Trade Bps", "avg_trade_bps", False, " bps"),
        ("Sharpe Ratio", "sharpe_ratio", False, ""),
        ("Sortino Ratio", "sortino_ratio", False, ""),
        ("Max Drawdown", "max_drawdown_pct", True, "%"),
        ("Avg Margin Utilization", "avg_portfolio_utilization_pct", True, "%"),
        ("Profit Factor", "profit_factor", False, "")
    ]
    
    status_full = results_full.get("status")
    status_abl = results_ablated.get("status")
    
    for label, key, is_pct, suffix in metrics:
        if key == "total_trades":
            val_full = results_full.get("total_trades", 0)
            val_abl = results_ablated.get("total_trades", 0)
            print(f"{label:<30} | {val_full:<15} | {val_abl:<15}")
            continue
            
        if status_full == "NO_TRADES":
            str_full = "N/A (No Trades)"
        else:
            val_full = results_full.get(key, 0.0)
            if label == "Total PnL":
                str_full = f"${val_full:,.2f}"
            elif is_pct:
                str_full = f"{val_full:.2f}{suffix}"
            elif isinstance(val_full, int):
                str_full = f"{val_full}"
            else:
                str_full = f"{val_full:.4f}{suffix}"
                
        if status_abl == "NO_TRADES":
            str_abl = "N/A (No Trades)"
        else:
            val_abl = results_ablated.get(key, 0.0)
            if label == "Total PnL":
                str_abl = f"${val_abl:,.2f}"
            elif is_pct:
                str_abl = f"{val_abl:.2f}{suffix}"
            elif isinstance(val_abl, int):
                str_abl = f"{val_abl}"
            else:
                str_abl = f"{val_abl:.4f}{suffix}"
            
        print(f"{label:<30} | {str_full:<15} | {str_abl:<15}")
    print(f"==================================================\n")

def run_all_portability_tests():
    logger.info("🚀 Starting Multi-Asset Portability Testing suite...")
    for ticker in ["IWM", "GLD", "TLT"]:
        try:
            run_portability_for_ticker(ticker)
        except Exception as ex:
            logger.exception(f"Portability test failed for {ticker}:")

if __name__ == "__main__":
    run_all_portability_tests()
