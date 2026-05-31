#!/usr/bin/env python3
"""
friction_sweep.py - Systematic Option Transaction Slippage Sensitivity Sweeps
Runs options backtests on QQQ and SPY across various per-leg slippage boundaries
to identify the friction threshold and assess the gross edge of the signal.
"""

import os
import logging
import pandas as pd
from data_pipeline import DataPipeline
from backtester import OptionsBacktester

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("quant_system.friction_sweep")
logger.setLevel(logging.INFO)

# Suppress other loggers to keep output clean
logging.getLogger("quant_system.backtester").setLevel(logging.ERROR)
logging.getLogger("quant_system.data_pipeline").setLevel(logging.ERROR)
logging.getLogger("quant_system.risk_engine").setLevel(logging.ERROR)
logging.getLogger("quant_system.volatility_engine").setLevel(logging.ERROR)
logging.getLogger("quant_system.regime_classifier").setLevel(logging.ERROR)

def run_friction_sweep():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(base_dir, "data_cache")
    pipeline = DataPipeline(cache_dir=cache_dir)
    
    tickers = ["QQQ", "SPY"]
    slippage_values = [0.005, 0.01, 0.015, 0.02, 0.025]
    
    # Store results
    all_results = []
    
    for ticker in tickers:
        pred_path = os.path.join(cache_dir, f"{ticker}_meta_predictions.csv")
        if not os.path.exists(pred_path):
            logger.error(f"Meta predictions file for {ticker} not found at {pred_path}!")
            continue
            
        meta_predictions_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
        start_date = meta_predictions_df.index.min().strftime("%Y-%m-%d")
        end_date = meta_predictions_df.index.max().strftime("%Y-%m-%d")
        
        underlying_df = pipeline.get_underlying_data(ticker, start_date, end_date)
        vix_series = pipeline.get_vix_data(start_date, end_date)
        
        for slippage in slippage_values:
            logger.info(f"Running sweep for {ticker} with per-leg slippage = ${slippage:.4f}...")
            
            backtester = OptionsBacktester(
                initial_capital=200000.0,
                max_position_risk_pct=0.01,
                max_portfolio_exposure_pct=0.25,
                commission_per_contract=0.65,
                option_slippage=slippage
            )
            
            res = backtester.run_backtest(
                underlying_df=underlying_df,
                vix_series=vix_series,
                meta_predictions_df=meta_predictions_df,
                pipeline=pipeline,
                ticker=ticker,
                enable_regime=True,
                enable_expectancy_pause=False  # Run raw matrix without pause to assess raw signal capacity
            )
            
            if res.get("status") == "NO_TRADES":
                all_results.append({
                    "ticker": ticker,
                    "slippage_per_leg": slippage,
                    "package_slippage_2leg": slippage * 2,
                    "trades": 0,
                    "win_rate": 0.0,
                    "return_pct": 0.0,
                    "max_dd": 0.0,
                    "sharpe": 0.0,
                    "sortino": 0.0,
                    "selectivity": 0.0
                })
            else:
                all_results.append({
                    "ticker": ticker,
                    "slippage_per_leg": slippage,
                    "package_slippage_2leg": slippage * 2,
                    "trades": res.get("total_trades", 0),
                    "win_rate": res.get("win_rate", 0.0),
                    "return_pct": res.get("total_return_pct", 0.0),
                    "max_dd": res.get("max_drawdown_pct", 0.0),
                    "sharpe": res.get("sharpe_ratio", 0.0),
                    "sortino": res.get("sortino_ratio", 0.0),
                    "selectivity": res.get("selectivity_pct", 0.0)
                })

    # Compile and display table
    print("\n" + "="*95)
    print(f"📊 OPTION FRICTION SENSITIVITY SWEEP RESULTS")
    print("="*95)
    print(f"{'Ticker':<6} | {'Slippage/Leg':<12} | {'Slippage/Pkg':<12} | {'Trades':<6} | {'Win Rate %':<10} | {'Return %':<9} | {'Max DD %':<9} | {'Sharpe':<7} | {'Sortino':<7}")
    print("-" * 95)
    for r in all_results:
        print(f"{r['ticker']:<6} | "
              f"${r['slippage_per_leg']:<11.3f} | "
              f"${r['package_slippage_2leg']:<11.3f} | "
              f"{r['trades']:<6} | "
              f"{r['win_rate']:<10.2f} | "
              f"{r['return_pct']:<8.2f}% | "
              f"{r['max_dd']:<8.2f}% | "
              f"{r['sharpe']:<7.2f} | "
              f"{r['sortino']:<7.2f}")
    print("="*95)
    
    # Save a CSV copy of the sweep results for reporting/walkthroughs
    df_out = pd.DataFrame(all_results)
    csv_path = os.path.join(base_dir, "friction_sweep_results.csv")
    df_out.to_csv(csv_path, index=False)
    logger.info(f"Sweep results successfully written to {csv_path}")

if __name__ == "__main__":
    run_friction_sweep()
