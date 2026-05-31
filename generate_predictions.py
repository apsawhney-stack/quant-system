"""
generate_predictions.py - Multi-Factor Quant Research & Trading Platform
Runs historical batch inference using fine-tuned Kronos models for QQQ and SPY.
Generates daily directional scores (kronos_score) and saves them for options backtesting.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np

# Add Kronos root to path dynamically
from pathlib import Path
import torch

base_kronos_path = Path(__file__).parent.parent.absolute() / "Kronos"
sys.path.append(str(base_kronos_path))

from model import Kronos, KronosTokenizer, KronosPredictor

# Set up logging
logger = logging.getLogger("quant_system.generate_predictions")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

def generate_predictions_for_ticker(ticker: str):
    logger.info(f"🔮 Generating predictions for {ticker} using fine-tuned weights...")
    
    # 1. Define paths to local checkpoints
    tokenizer_path = str(base_kronos_path / f"finetune_csv/finetuned/{ticker}_daily/tokenizer/best_model")
    predictor_path = str(base_kronos_path / f"finetune_csv/finetuned/{ticker}_daily/basemodel/best_model")
    
    if not os.path.exists(tokenizer_path) or not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Fine-tuned weights not found for {ticker}. Ensure training completed successfully.")
        
    # Load model and tokenizer
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
    model = Kronos.from_pretrained(predictor_path)
    
    # Detect device automatically
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)
    logger.info(f"Model loaded. Running on device: {predictor.device}")
    
    # 2. Ingest historical data
    data_path = str(base_kronos_path / f"finetune_csv/data/{ticker}_daily.csv")
    df = pd.read_csv(data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    
    # We generate predictions for dates where we have at least 400 days of lookback
    lookback = 400
    pred_len = 10
    
    logger.info(f"Loaded {len(df)} daily records for {ticker}. Generating walk-forward predictions...")
    
    # Prepare batch lists
    dfs = []
    xtsp = []
    ytsp = []
    anchor_dates = []
    current_closes = []
    
    # We generate predictions starting from index 400 to the end, filtering for backtest range
    for idx in range(lookback, len(df)):
        anchor_date = df.iloc[idx-1]['timestamps']
        if anchor_date < pd.to_datetime('2021-05-20'):
            continue
            
        x_df = df.iloc[idx-lookback:idx].copy().reset_index(drop=True)
        x_timestamp = x_df['timestamps']
        
        # If there are not enough future days in the dataset, generate future business days
        if idx + pred_len <= len(df):
            y_timestamp = df.iloc[idx:idx+pred_len]['timestamps'].reset_index(drop=True)
        else:
            last_date = x_timestamp.max()
            y_timestamp = pd.Series(pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_len, freq="B"))
            
        dfs.append(x_df[['open', 'high', 'low', 'close', 'volume', 'amount']])
        xtsp.append(x_timestamp)
        ytsp.append(y_timestamp)
        anchor_dates.append(anchor_date)
        current_closes.append(df.iloc[idx-1]['close'])
        
    total_samples = len(dfs)
    logger.info(f"Batched {total_samples} test windows. Running inference...")
    
    # Run batch prediction in chunks of size 16 to manage memory/CPU resources
    batch_size = 16
    pred_closes = []
    
    for i in range(0, total_samples, batch_size):
        chunk_dfs = dfs[i:i+batch_size]
        chunk_xtsp = xtsp[i:i+batch_size]
        chunk_ytsp = ytsp[i:i+batch_size]
        
        logger.info(f"Processing chunk {i//batch_size + 1}/{(total_samples-1)//batch_size + 1} (size {len(chunk_dfs)})...")
        
        try:
            pred_list = predictor.predict_batch(
                df_list=chunk_dfs,
                x_timestamp_list=chunk_xtsp,
                y_timestamp_list=chunk_ytsp,
                pred_len=pred_len,
                T=0.6,
                top_p=0.9,
                sample_count=1,
                verbose=False
            )
            for b_idx in range(len(chunk_dfs)):
                item_pred = pred_list[b_idx]
                # Use the 10th day close price prediction
                pred_closes.append(item_pred.iloc[-1]['close'])
        except Exception as e:
            logger.error(f"Error during batch inference at chunk index {i}: {e}. Falling back to last close price.")
            for b_idx in range(len(chunk_dfs)):
                pred_closes.append(current_closes[i + b_idx])
            
    # Calculate scores
    df_results = pd.DataFrame({
        'date': anchor_dates,
        'current_close': current_closes,
        'predicted_close_10d': pred_closes
    })
    
    # Predicted 10-day return
    df_results['predicted_return'] = (df_results['predicted_close_10d'] - df_results['current_close']) / df_results['current_close']
    
    # Scale score to [-1.0, 1.0] by dividing by 5% baseline target return
    df_results['kronos_score'] = (df_results['predicted_return'] / 0.05).clip(-1.0, 1.0)
    
    # Save CSV
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ticker}_predictions.csv")
    df_results.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved predictions for {ticker} to: {output_path}")
    print(df_results.head())
    print(df_results.tail())

def main():
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
        if ticker in ["QQQ", "SPY"]:
            generate_predictions_for_ticker(ticker)
        else:
            logger.error(f"Unknown ticker: {ticker}")
    else:
        generate_predictions_for_ticker("QQQ")
        generate_predictions_for_ticker("SPY")

if __name__ == "__main__":
    main()
