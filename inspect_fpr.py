import os
import pandas as pd
import numpy as np
from data_pipeline import DataPipeline
from backtester import OptionsBacktester

# Force logging to print rejections
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quant_system.backtester")
logger.setLevel(logging.INFO)

base_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(base_dir, "data_cache")
pipeline = DataPipeline(cache_dir=cache_dir)

ticker = "QQQ"
pred_path = os.path.join(cache_dir, f"{ticker}_meta_predictions.csv")
meta_predictions_df = pd.read_csv(pred_path, parse_dates=True, index_col=0)
start_date = meta_predictions_df.index.min().strftime("%Y-%m-%d")
end_date = meta_predictions_df.index.max().strftime("%Y-%m-%d")

underlying_df = pipeline.get_underlying_data(ticker, start_date, end_date)
vix_series = pipeline.get_vix_data(start_date, end_date)

# Let's run with 0.025 slippage and see the log output
backtester = OptionsBacktester(
    initial_capital=200000.0,
    max_position_risk_pct=0.01,
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

print(f"Total trades entered: {res.get('total_trades')}")
