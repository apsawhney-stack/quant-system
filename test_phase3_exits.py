import os
import pandas as pd
from data_pipeline import DataPipeline
from backtester import OptionsBacktester

base_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(base_dir, "data_cache")
pipeline = DataPipeline(cache_dir=cache_dir)

for ticker in ["QQQ", "SPY"]:
    pred_path = os.path.join(cache_dir, f"{ticker}_meta_predictions.csv")
    if not os.path.exists(pred_path):
        continue
    meta_predictions_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
    start_date = meta_predictions_df.index.min().strftime("%Y-%m-%d")
    end_date = meta_predictions_df.index.max().strftime("%Y-%m-%d")
    
    underlying_df = pipeline.get_underlying_data(ticker, start_date, end_date)
    vix_series = pipeline.get_vix_data(start_date, end_date)
    
    # We will sweep over FPR limits and contraction thresholds
    fpr_limits = [0.015, 0.03, 0.05, 0.08]
    contraction_thresholds = [20.0, 25.0]
    
    print(f"{'FPR Gate':<10} | {'IV Contraction':<15} | {'Total Trades':<12} | {'VOL_Exits':<10} | {'Return %':<10}")
    print("-" * 65)
    
    for fpr in fpr_limits:
        for threshold in contraction_thresholds:
            # We dynamically monkeypatch the backtester's parameter and run the backtest
            backtester = OptionsBacktester(
                initial_capital=200000.0,
                max_position_risk_pct=0.01,
                max_portfolio_exposure_pct=0.25,
                commission_per_contract=0.65,
                slippage_pct_spread=0.50
            )
            # Assuming backtester is updated to accept these parameters in __init__
            backtester.max_permissible_fpr = fpr
            backtester.iv_contraction_threshold = threshold
            
            results = backtester.run_backtest(
                underlying_df=underlying_df,
                vix_series=vix_series,
                meta_predictions_df=meta_predictions_df,
                pipeline=pipeline,
                ticker=ticker,
                enable_regime=True
            )
            vol_exits = [t for t in backtester.trade_log if t['exit_reason'] == "VOL_PROFIT_TAKE"]
            print(f"{fpr:<10} | {threshold:<15} | {results.get('total_trades', 0):<12} | {len(vol_exits):<10} | {results.get('total_return_pct', 0):.2f}%")
