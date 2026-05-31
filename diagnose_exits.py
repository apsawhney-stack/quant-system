import os
import pandas as pd
from data_pipeline import DataPipeline
from backtester import OptionsBacktester
from regime_classifier import RegimeClassifier

base_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(base_dir, "data_cache")
pipeline = DataPipeline(cache_dir=cache_dir)

ticker = "SPY"
pred_path = os.path.join(cache_dir, f"{ticker}_meta_predictions.csv")
meta_predictions_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
start_date = meta_predictions_df.index.min().strftime("%Y-%m-%d")
end_date = meta_predictions_df.index.max().strftime("%Y-%m-%d")

underlying_df = pipeline.get_underlying_data(ticker, start_date, end_date)
vix_series = pipeline.get_vix_data(start_date, end_date)

# Let's run backtest and inspect position tracking
backtester = OptionsBacktester(
    initial_capital=200000.0,
    max_position_risk_pct=0.01,  # 1% risk per trade
    max_portfolio_exposure_pct=0.25,
    commission_per_contract=0.65,
    option_slippage=0.025
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

# Print premium selling trades and their parameters
premium_trades = [t for t in backtester.trade_log if t['strategy'].startswith("SELL_")]
print(f"Total premium selling trades: {len(premium_trades)}")
regime_classifier = RegimeClassifier()
regime_df = regime_classifier.classify_history(underlying_df['close'], vix_series)

for i, t in enumerate(premium_trades):
    print(f"Trade {i+1}: Strategy={t['strategy']}, Entry={t['entry_date']}, Exit={t['exit_date']}, Reason={t['exit_reason']}")
    print(f"  Entry Spot={t['entry_spot']:.2f}, Exit Spot={t['exit_spot']:.2f}")
    # Let's see if we can find VIX values during the trade
    entry_dt = pd.to_datetime(t['entry_date'])
    exit_dt = pd.to_datetime(t['exit_date'])
    # Look up VIX percentiles from the data
    entry_vix_p = regime_df.loc[entry_dt, 'vix_percentile_1y'] * 100.0
    exit_vix_p = regime_df.loc[exit_dt, 'vix_percentile_1y'] * 100.0
    print(f"  Entry VIX Percentile={entry_vix_p:.2f}%, Exit VIX Percentile={exit_vix_p:.2f}%")
    print(f"  Max contraction seen during hold period:")
    # Calculate contraction for each day held
    hold_dates = pd.date_range(start=entry_dt, end=exit_dt)
    contractions = []
    for d in hold_dates:
        if d in regime_df.index:
            v_p = regime_df.loc[d, 'vix_percentile_1y'] * 100.0
            contractions.append(entry_vix_p - v_p)
    if contractions:
        print(f"    Max Contraction={max(contractions):.2f}%")
