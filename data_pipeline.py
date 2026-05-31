"""
data_pipeline.py - Multi-Factor Quant Research & Trading Platform
Ingests underlying price data from yfinance and historical/live options chain data
from paid APIs (Polygon.io or ThetaData) with local caching and logging.
"""

import os
import logging
import datetime
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import requests

def fast_norm_pdf(x: float) -> float:
    try:
        return math.exp(-x * x / 2.0) / 2.5066282746310002
    except OverflowError:
        return 0.0

def fast_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

# Set up logging
logger = logging.getLogger("quant_system.data_pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

class DataPipeline:
    """
    Ingests and manages market and options chain data.
    Supports yfinance for underlying stock data and Polygon.io / ThetaData for options chains.
    Enforces local disk caching to minimize API costs and rate limits.
    """
    
    def __init__(self, cache_dir: str = "./data_cache", polygon_api_key: Optional[str] = None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.polygon_api_key = polygon_api_key or os.getenv("POLYGON_API_KEY")
        if not self.polygon_api_key:
            logger.warning("POLYGON_API_KEY not found. Polygon.io queries will fail unless configured.")
            
        logger.info(f"Data pipeline initialized. Local cache directory: {self.cache_dir.resolve()}")
        self._synthetic_chain_cache = {}

    # =========================================================================
    # UNDERLYING DATA (yfinance)
    # =========================================================================
    
    def get_underlying_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetches daily underlying OHLCV data from yfinance.
        """
        cache_file = self.cache_dir / f"{ticker}_underlying_{start_date}_{end_date}.csv"
        
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached underlying data for {ticker} from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
            
        logger.info(f"Fetching underlying data for {ticker} from {start_date} to {end_date} via yfinance...")
        try:
            # Download daily data
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
            if df.empty:
                raise ValueError(f"No yfinance data returned for ticker {ticker}")
                
            # Flatten multi-level columns if present (yfinance sometimes outputs multi-index columns)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Format to Kronos-compatible columns: open, high, low, close, volume, amount
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            })
            
            # Estimate volume amount (Close * Volume)
            df['amount'] = df['close'] * df['volume']
            df.index.name = 'date'
            
            # Cache the result
            if use_cache:
                df.to_csv(cache_file)
                logger.info(f"Cached underlying data to {cache_file}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching underlying data for {ticker}: {str(e)}")
            raise e

    # =========================================================================
    # VOLATILITY METRICS (IV PERCENTILE & VIX)
    # =========================================================================
    
    def get_vix_data(self, start_date: str, end_date: str, use_cache: bool = True) -> pd.Series:
        """
        Fetches historical CBOE Volatility Index (VIX) close prices.
        """
        df = self.get_underlying_data("^VIX", start_date, end_date, use_cache=use_cache)
        return df['close']

    def calculate_iv_percentile(self, iv_series: pd.Series, lookback_window: int = 252) -> pd.Series:
        """
        Calculates the trailing 52-week (lookback_window) IV Percentile.
        Robust against outlier vol spikes compared to IV Rank.
        Formula: percentage of days in lookback window where IV was lower than today's IV.
        """
        def pct_rank(window):
            if len(window) < 2:
                return np.nan
            current_val = window[-1]
            # Count elements strictly less than current value
            less_than_count = np.sum(window[:-1] < current_val)
            return (less_than_count / (len(window) - 1)) * 100.0

        return iv_series.rolling(window=lookback_window, min_periods=lookback_window).apply(pct_rank, raw=True)

    # =========================================================================
    # HISTORICAL OPTIONS DATA (Polygon.io implementation)
    # =========================================================================
    
    def get_polygon_options_chain(
        self, 
        underlying: str, 
        date: str, 
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetches the end-of-day options chain for a specific underlying on a specific date
        from Polygon.io API.
        Reference: https://polygon.io/docs/options/get_v3_marketdata_options_chains
        """
        if not self.polygon_api_key:
            raise ValueError("Polygon API key is required to query options chains.")
            
        # Clean formatting
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        cache_file = self.cache_dir / f"{underlying}_options_chain_{date_str}.csv"
        
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached options chain for {underlying} on {date_str}")
            return pd.read_csv(cache_file)
            
        logger.info(f"Fetching options chain for {underlying} on {date_str} from Polygon.io...")
        
        url = f"https://api.polygon.io/v3/marketdata/options/chains"
        params = {
            "underlying_ticker": underlying,
            "as_of": date_str,
            "limit": 1000,
            "apiKey": self.polygon_api_key
        }
        
        all_results = []
        next_url = url
        
        try:
            while next_url:
                if next_url == url:
                    response = requests.get(next_url, params=params)
                else:
                    # Next cursor includes API key and parameters already
                    response = requests.get(f"{next_url}&apiKey={self.polygon_api_key}")
                    
                if response.status_code != 200:
                    logger.error(f"Polygon API error: {response.status_code} - {response.text}")
                    break
                    
                data = response.json()
                results = data.get("results", [])
                all_results.extend(results)
                
                # Check for pagination pagination
                next_url = data.get("next_url")
                
            if not all_results:
                logger.warning(f"No options chain results returned from Polygon for {underlying} on {date_str}")
                return pd.DataFrame()
                
            # Parse into a structured DataFrame
            parsed_data = []
            for item in all_results:
                details = item.get("details", {})
                greeks = item.get("greeks", {})
                implied_vol = item.get("implied_volatility")
                
                parsed_data.append({
                    "ticker": item.get("ticker"),
                    "underlying": underlying,
                    "date": date_str,
                    "strike": details.get("strike_price"),
                    "expiry": details.get("expiration_date"),
                    "type": details.get("contract_type"), # 'call' or 'put'
                    "bid": item.get("bid"),
                    "ask": item.get("ask"),
                    "open_interest": item.get("open_interest"),
                    "volume": item.get("volume"),
                    "iv": implied_vol,
                    "delta": greeks.get("delta"),
                    "gamma": greeks.get("gamma"),
                    "vega": greeks.get("vega"),
                    "theta": greeks.get("theta")
                })
                
            df = pd.DataFrame(parsed_data)
            
            # Cache the result
            if use_cache and not df.empty:
                df.to_csv(cache_file, index=False)
                logger.info(f"Cached options chain to {cache_file}")
                
            return df
            
        except Exception as e:
            logger.error(f"Error querying Polygon options chain: {str(e)}")
            raise e

    # =========================================================================
    # HISTORICAL OPTIONS DATA (ThetaData placeholder / client interface)
    # =========================================================================
    
    def get_thetadata_options_chain(
        self, 
        underlying: str, 
        date: str,
        port: int = 11000
    ) -> pd.DataFrame:
        """
        Interface for fetching options chains from a running ThetaTerminal instance via REST API.
        Reference: https://http.thetadata.us/
        """
        date_str = pd.to_datetime(date).strftime("%Y%m%d")
        url = f"http://127.0.0.1:{port}/v2/bulk_snapshot/option/quote"
        
        # ThetaData REST endpoint payloads depend on local terminal config.
        # This is a robust stub structure for connecting to local terminals.
        logger.info(f"Attempting to fetch ThetaData snapshot for {underlying} on {date_str}")
        params = {
            "root": underlying,
            "expDate": "0",  # 0 indicates all expirations
            "date": date_str
        }
        
        try:
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Parse data structure according to ThetaData documentation
                # (returns bulk quote ticks: [strike, expiry, type, bid, ask, iv, delta, gamma, etc.])
                # For development safety, fallback to Mock or Polygon if connection fails.
                return pd.DataFrame(data)
            else:
                logger.warning(f"ThetaTerminal responded with code {response.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"Could not connect to ThetaTerminal on port {port}: {str(e)}")
            return pd.DataFrame()

    # =========================================================================
    # ROBUST STUB / TEST DATA GENERATOR (For offline development and testing)
    # =========================================================================
    
    def generate_synthetic_options_chain(
        self, 
        underlying: str, 
        date: str, 
        spot_price: float, 
        atm_iv: float
    ) -> pd.DataFrame:
        """
        Generates a high-fidelity synthetic options chain using Black-Scholes pricing
        with a realistic implied volatility smile skew and bid-ask spreads.
        Used strictly for testing/offline development.
        """
        cache_key = (underlying, date, round(spot_price, 2), round(atm_iv, 4))
        if hasattr(self, '_synthetic_chain_cache') and cache_key in self._synthetic_chain_cache:
            return self._synthetic_chain_cache[cache_key]
            
        date_dt = pd.to_datetime(date)
        expiries = [
            (date_dt + datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
            (date_dt + datetime.timedelta(days=14)).strftime("%Y-%m-%d"),
            (date_dt + datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
            (date_dt + datetime.timedelta(days=45)).strftime("%Y-%m-%d")
        ]
        
        # Generate strikes centered around spot price with a wider boundary (15% skew offset support)
        strikes = np.arange(int(spot_price * 0.85), int(spot_price * 1.15), 1)
        
        rows = []
        for expiry in expiries:
            days_to_expiry = max((pd.to_datetime(expiry) - date_dt).days, 1)
            t = days_to_expiry / 365.0
            r = 0.045 # Risk-free rate
            
            for strike in strikes:
                for option_type in ["call", "put"]:
                    # Create a simple volatility smile
                    skew = -0.15 * ((strike - spot_price) / spot_price)
                    vol = max(atm_iv + skew + (0.1 * ((strike - spot_price) / spot_price)**2), 0.05)
                    
                    # Black-Scholes formula
                    d1 = (np.log(spot_price / strike) + (r + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
                    d2 = d1 - vol * np.sqrt(t)
                    
                    if option_type == "call":
                        price = spot_price * fast_norm_cdf(d1) - strike * np.exp(-r * t) * fast_norm_cdf(d2)
                        delta = fast_norm_cdf(d1)
                        theta = -(spot_price * fast_norm_pdf(d1) * vol) / (2 * np.sqrt(t)) - r * strike * np.exp(-r * t) * fast_norm_cdf(d2)
                    else:
                        price = strike * np.exp(-r * t) * fast_norm_cdf(-d2) - spot_price * fast_norm_cdf(-d1)
                        delta = fast_norm_cdf(d1) - 1
                        theta = -(spot_price * fast_norm_pdf(d1) * vol) / (2 * np.sqrt(t)) + r * strike * np.exp(-r * t) * fast_norm_cdf(-d2)
                        
                    gamma = fast_norm_pdf(d1) / (spot_price * vol * np.sqrt(t))
                    vega = spot_price * np.sqrt(t) * fast_norm_pdf(d1)
                    
                    # Format prices cleanly
                    price = max(price, 0.01)
                    # Add bid-ask spread
                    spread = max(price * 0.05, 0.05) # 5% spread, min $0.05
                    bid = max(price - spread / 2, 0.01)
                    ask = price + spread / 2
                    
                    rows.append({
                        "ticker": f"{underlying}_{pd.to_datetime(expiry).strftime('%y%m%d')}{option_type[0].upper()}{strike}",
                        "underlying": underlying,
                        "date": date_dt.strftime("%Y-%m-%d"),
                        "strike": float(strike),
                        "expiry": expiry,
                        "type": option_type,
                        "bid": round(bid, 2),
                        "ask": round(ask, 2),
                        "open_interest": int(np.random.randint(100, 10000)),
                        "volume": int(np.random.randint(10, 2000)),
                        "iv": round(vol, 4),
                        "delta": round(delta, 4),
                        "gamma": round(gamma, 4),
                        "vega": round(vega, 4),
                        "theta": round(theta / 365, 4) # Daily theta
                    })
                    
        df_res = pd.DataFrame(rows)
        if hasattr(self, '_synthetic_chain_cache'):
            self._synthetic_chain_cache[cache_key] = df_res
        return df_res

    def prepare_classifier_dataset(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        k: float = 0.75
    ) -> pd.DataFrame:
        """
        Prepares a clean, stationary dataset for stacked classification (Level-1)
        and meta-labeling (Level-2).
        Calculates stationary target variables, dynamic volatility boundaries,
        and features (returns, VIX ratios, IV-RV spreads).
        """
        # Load underlying stock and VIX data
        df_stock = self.get_underlying_data(ticker, start_date, end_date)
        vix_series = self.get_vix_data(start_date, end_date)
        
        # Load Kronos price predictions
        pred_path = self.cache_dir / f"{ticker}_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file {pred_path} not found. Run generate_predictions.py first.")
        df_pred = pd.read_csv(pred_path)
        df_pred['date'] = pd.to_datetime(df_pred['date'])
        df_pred.set_index('date', inplace=True)
        
        # Create aligned DataFrame
        df = df_stock[['close']].join(vix_series.rename("vix_close"), how='inner')
        df = df.join(df_pred[['predicted_close_10d']], how='inner')
        df = df.sort_index()
        
        # 1. Level-0 predicted return (stationary)
        df['kronos_predicted_return'] = (df['predicted_close_10d'] - df['close']) / df['close']
        
        # 2. Trailing log returns (stationary features)
        log_price = np.log(df['close'])
        df['log_ret_5d'] = log_price - log_price.shift(5)
        df['log_ret_10d'] = log_price - log_price.shift(10)
        df['log_ret_20d'] = log_price - log_price.shift(20)
        
        # 3. Volatility calculations (stationary features)
        daily_ret = log_price - log_price.shift(1)
        df['realized_vol_daily'] = daily_ret.rolling(20).std()
        df['realized_vol_20d'] = df['realized_vol_daily'] * np.sqrt(252)
        
        # IV-RV Spread
        df['iv_rv_spread'] = (df['vix_close'] / 100.0) - df['realized_vol_20d']
        
        # VIX term structure proxy (ratio of VIX to 50d MA)
        df['vix_ratio'] = df['vix_close'] / df['vix_close'].rolling(50).mean()
        # VIX 5d rate of change
        df['vix_roc_5d'] = (df['vix_close'] - df['vix_close'].shift(5)) / df['vix_close'].shift(5)
        # 50d DMA slope
        ma_50 = df['close'].rolling(50).mean()
        df['ma_50_slope'] = ma_50.diff(5)
        
        # Phase 2 Feature Injection
        # Volatility Momentum: 5-day rolling std of IVPercentile change rate
        df['iv_percentile'] = self.calculate_iv_percentile(df['vix_close'])
        df['vol_momentum'] = df['iv_percentile'].diff().rolling(5).std().fillna(0.0)
        
        # Empirical VRP Multiplier
        df['empirical_vrp_mult'] = (df['iv_percentile'] / 100.0) * df['iv_rv_spread']
        
        # VIX Term Structure Slope (VIX3M / VIX close) with strict cleaning
        try:
            vix3m_df = self.get_underlying_data("^VIX3M", start_date, end_date, use_cache=True)
            df = df.join(vix3m_df['close'].rename("vix_3m"), how='left')
            
            df['vix3m_clean'] = df['vix_3m'].ffill()
            df['vix_clean'] = df['vix_close'].ffill()
            
            df['vix_slope'] = df['vix3m_clean'] / df['vix_clean']
            df['vix_slope'].replace([np.inf, -np.inf], 1.10, inplace=True)
            df['vix_slope'].fillna(1.10, inplace=True)
            
            # Clean temporary columns
            df.drop(columns=['vix3m_clean', 'vix_clean'], inplace=True, errors='ignore')
        except Exception as e:
            logger.warning(f"Could not load ^VIX3M: {str(e)}. Falling back to proxying vix_slope.")
            df['vix_slope'] = 1.10 / df['vix_ratio'].clip(0.5, 2.0)
            df['vix_slope'].replace([np.inf, -np.inf], 1.10, inplace=True)
            df['vix_slope'].fillna(1.10, inplace=True)
        
        # 4. Volatility-scaled target boundaries (10-day ahead)
        df['fwd_log_ret_10d'] = log_price.shift(-10) - log_price
        # Expected standard deviation over 10 days
        df['theta'] = k * df['realized_vol_daily'] * np.sqrt(10)
        
        # Binary Targets
        df['target_bullish'] = (df['fwd_log_ret_10d'] >= df['theta']).astype(int)
        df['target_bearish'] = (df['fwd_log_ret_10d'] <= -df['theta']).astype(int)
        
        # Fill warm-up NaNs with 0
        df = df.dropna(subset=['realized_vol_daily', 'kronos_predicted_return'])
        
        # Save to disk
        out_path = self.cache_dir / f"{ticker}_classifier_features.csv"
        df.to_csv(out_path)
        logger.info(f"Successfully compiled and saved features for {ticker} to {out_path} (shape: {df.shape})")
        
        return df

# =========================================================================
# TEST DRIVER
# =========================================================================
if __name__ == "__main__":
    logger.info("Running DataPipeline basic validation...")
    pipeline = DataPipeline(cache_dir="./test_cache")
    
    # 1. Fetch underlying QQQ data
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    year_ago = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y-%m-%d")
    
    try:
        qqq_data = pipeline.get_underlying_data("QQQ", year_ago, today)
        logger.info(f"Successfully fetched QQQ underlying daily bars. Shape: {qqq_data.shape}")
        print(qqq_data.head())
        
        # 2. Fetch VIX data
        vix_data = pipeline.get_vix_data(year_ago, today)
        logger.info(f"Successfully fetched VIX close prices. Shape: {vix_data.shape}")
        print(vix_data.head())
        
        # 3. Test IV Percentile calculation
        # Create a mock IV Series
        mock_iv = pd.Series(np.random.normal(0.20, 0.05, 500))
        iv_percentiles = pipeline.calculate_iv_percentile(mock_iv)
        logger.info(f"Calculated IV percentiles for mock series. NaNs: {iv_percentiles.isna().sum()}")
        print(iv_percentiles.dropna().head())
        
        # 4. Generate synthetic options chain (Fallback test)
        synthetic_chain = pipeline.generate_synthetic_options_chain("QQQ", "2026-05-29", 450.0, 0.18)
        logger.info(f"Generated synthetic options chain. Rows: {len(synthetic_chain)}")
        print(synthetic_chain.head())
        
    except Exception as ex:
        logger.exception("Validation failed with error:")
