"""
run_backtest_pipeline.py - Multi-Factor Quant Research & Trading Platform
Driver script to run the options backtesting pipeline on historical data.
Ingests real Kronos predictions and runs full and ablated configurations.
"""

import os
import logging
import datetime
import pandas as pd
import numpy as np
from data_pipeline import DataPipeline
from regime_classifier import RegimeClassifier
from volatility_engine import VolatilityEngine
from backtester import OptionsBacktester
from kronos_signal import StackedSignalEngine
from meta_classifier import MetaLabelingClassifier

# Set up logging
logger = logging.getLogger("quant_system.run_backtest_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def run_backtest_for_ticker(ticker: str):
    logger.info(f"\n==================================================")
    logger.info(f"🏁 RUNNING BACKTEST PIPELINE FOR {ticker}")
    logger.info(f"==================================================")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "data_cache")
    
    pipeline = DataPipeline(cache_dir=cache_dir)
    backtester = OptionsBacktester(
        initial_capital=200000.0,
        max_position_risk_pct=0.01,         # 1% per trade
        max_portfolio_exposure_pct=0.25,    # 25% max margin
        commission_per_contract=0.65,
        slippage_pct_spread=0.50,
        max_permissible_fpr=0.03,
        iv_contraction_threshold=25.0
    )
    
    # 1. Run Level-1 stacked classifier models
    logger.info("Executing Level-1 Stacked Signal Engine...")
    l1_engine = StackedSignalEngine(cache_dir=cache_dir)
    try:
        l1_engine.generate_calibrated_signals(ticker)
    except Exception as e:
        logger.error(f"Failed to generate Level-1 signals: {str(e)}")
        return
        
    # 2. Run Level-2 meta-classifier model
    logger.info("Executing Level-2 Meta-Labeling Classifier...")
    l2_engine = MetaLabelingClassifier(cache_dir=cache_dir)
    try:
        l2_engine.generate_meta_signals(ticker)
    except Exception as e:
        logger.error(f"Failed to generate Level-2 meta-signals: {str(e)}")
        return
        
    # 3. Load Level-2 meta-predictions
    pred_path = os.path.join(cache_dir, f"{ticker}_meta_predictions.csv")
    if os.path.exists(pred_path):
        logger.info(f"Loading cached Level-2 meta-predictions from {pred_path}...")
        meta_predictions_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
        logger.info(f"Loaded {len(meta_predictions_df)} meta-predictions.")
    else:
        logger.error(f"Meta-prediction file {pred_path} not found! Aborting.")
        return

    # 4. Compute start and end dates dynamically based on predictions range
    start_date = meta_predictions_df.index.min().strftime("%Y-%m-%d")
    end_date = meta_predictions_df.index.max().strftime("%Y-%m-%d")
    
    logger.info(f"Retrieving market data for {ticker} and VIX from {start_date} to {end_date}...")
    try:
        underlying_df = pipeline.get_underlying_data(ticker, start_date, end_date)
        vix_series = pipeline.get_vix_data(start_date, end_date)
        logger.info(f"Loaded {len(underlying_df)} days of price data.")
    except Exception as e:
        logger.error(f"Failed to fetch market data: {str(e)}")
        return

    # 5. Run Full System (with VIX regime classifier enabled)
    logger.info("Executing simulation loop WITH VIX Regime Classifier...")
    results_full = backtester.run_backtest(
        underlying_df=underlying_df,
        vix_series=vix_series,
        meta_predictions_df=meta_predictions_df,
        pipeline=pipeline,
        ticker=ticker,
        enable_regime=True
    )
    
    # 6. Run Ablated System (WITHOUT VIX regime classifier)
    logger.info("Executing simulation loop WITHOUT VIX Regime Classifier (Ablation)...")
    results_ablated = backtester.run_backtest(
        underlying_df=underlying_df,
        vix_series=vix_series,
        meta_predictions_df=meta_predictions_df,
        pipeline=pipeline,
        ticker=ticker,
        enable_regime=False
    )
    
    # 7. Display comparison
    print(f"\n==================================================")
    print(f"📊 {ticker} ABLATION STUDY COMPARISON")
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
        ("Margin Efficiency (ROBP)", "margin_efficiency_robp", True, "%"),
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

def run_pipeline():
    logger.info("🚀 Starting multi-factor options backtesting pipeline...")
    run_backtest_for_ticker("QQQ")
    run_backtest_for_ticker("SPY")

if __name__ == "__main__":
    run_pipeline()
