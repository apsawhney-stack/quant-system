"""
backtester.py - Multi-Factor Quant Research & Trading Platform
Simulates options trading using historical/synthetic options data, underlying price,
and signals. Enforces strict transaction costs, exits, and portfolio constraints.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from data_pipeline import DataPipeline
from regime_classifier import RegimeClassifier
from volatility_engine import VolatilityEngine
from risk_engine import RiskEngine, calculate_option_greeks

logger = logging.getLogger("quant_system.backtester")

class OptionsBacktester:
    """
    Simulates historical options trading with realistic slippage, commissions, 
    exit rules (delta, profit target, DTE, max loss, regime shift), and capital/margin metrics.
    """
    
    def __init__(
        self,
        initial_capital: float = 200000.0,
        max_position_risk_pct: float = 0.01,  # 1% per trade
        max_portfolio_exposure_pct: float = 0.25, # 25% max margin utilization
        commission_per_contract: float = 0.65,
        slippage_pct_spread: float = 0.50,     # Deduct 50% of bid-ask spread
        option_slippage: float = 0.025,          # Option slippage per leg
        max_permissible_fpr: float = 0.03,       # default FPR gate
        iv_contraction_threshold: float = 25.0   # default IV contraction threshold
    ):
        self.initial_capital = initial_capital
        self.max_position_risk_pct = max_position_risk_pct
        self.max_portfolio_exposure_pct = max_portfolio_exposure_pct
        self.commission_per_contract = commission_per_contract
        self.slippage_pct_spread = slippage_pct_spread
        self.option_slippage = option_slippage
        self.max_permissible_fpr = max_permissible_fpr
        self.iv_contraction_threshold = iv_contraction_threshold
        
        # Initialize risk engine
        self.risk_engine = RiskEngine(
            portfolio_value=self.initial_capital,
            max_trade_risk_pct=self.max_position_risk_pct,
            max_total_exposure_pct=self.max_portfolio_exposure_pct
        )
        
        # Reset state
        self.reset()
        
    def reset(self):
        self.cash = self.initial_capital
        self.net_liq = self.initial_capital
        self.active_positions: List[Dict[str, Any]] = []
        self.trade_log: List[Dict[str, Any]] = []
        self.daily_log: List[Dict[str, Any]] = []
        if hasattr(self, 'risk_engine'):
            self.risk_engine.portfolio_value = self.initial_capital
        
    def run_backtest(
        self,
        underlying_df: pd.DataFrame,
        vix_series: pd.Series,
        meta_predictions_df: pd.DataFrame,
        pipeline: DataPipeline,
        ticker: str = "QQQ",
        enable_regime: bool = True,
        bullish_threshold: float = 0.80,
        bearish_threshold: float = 0.80,
        enable_expectancy_pause: bool = True
    ) -> Dict[str, Any]:
        """
        Runs the daily simulation loop.
        """
        self.reset()
        
        # 1. Align data and calculate regimes
        classifier = RegimeClassifier()
        regime_df = classifier.classify_history(underlying_df['close'], vix_series)
        
        # Ensure meta_predictions_df is a DataFrame with expected columns
        df_meta = meta_predictions_df.copy()
        if isinstance(df_meta, pd.Series):
            s_name = df_meta.name or 'score'
            df_temp = pd.DataFrame(index=df_meta.index)
            df_temp[s_name] = df_meta
            df_meta = df_temp

        # Add missing columns with robust fallbacks
        if 'pred_prob_bull' not in df_meta.columns:
            col_candidates = [c for c in df_meta.columns if 'score' in c or 'pred' in c]
            if col_candidates:
                score = df_meta[col_candidates[0]]
                # Map score (typically in range -1 to 1 or log returns) to a probability
                df_meta['pred_prob_bull'] = score.apply(lambda x: 0.5 + 0.1 * x if x > 0 else 0.5)
            else:
                df_meta['pred_prob_bull'] = 0.5
                
        if 'pred_prob_bear' not in df_meta.columns:
            col_candidates = [c for c in df_meta.columns if 'score' in c or 'pred' in c]
            if col_candidates:
                score = df_meta[col_candidates[0]]
                df_meta['pred_prob_bear'] = score.apply(lambda x: 0.5 - 0.1 * x if x < 0 else 0.5)
            else:
                df_meta['pred_prob_bear'] = 0.5
                
        if 'meta_prob' not in df_meta.columns:
            df_meta['meta_prob'] = 0.55  # default to above threshold to avoid blocking
            
        if 'iv_rv_spread' not in df_meta.columns:
            # Estimate from VIX and realized vol
            realized_vol_daily = underlying_df['close'].pct_change().rolling(20).std()
            realized_vol_20d = realized_vol_daily * np.sqrt(252)
            iv_rv_spread = (vix_series / 100.0) - realized_vol_20d
            df_meta['iv_rv_spread'] = iv_rv_spread.reindex(df_meta.index).fillna(0.0)

        if 'vix_roc_5d' not in df_meta.columns:
            df_meta['vix_roc_5d'] = (vix_series.pct_change(5)).reindex(df_meta.index).fillna(0.0)
            
        if 'vix_ratio' not in df_meta.columns:
            vix_ma_10 = vix_series.rolling(window=10).mean()
            df_meta['vix_ratio'] = (vix_series / vix_ma_10).reindex(df_meta.index).fillna(1.0)
            
        # Compute out-of-sample rolling percentiles (252-day window, min 100 days)
        # We calculate the percentile rank trailing-only to strictly prevent look-ahead bias
        def pct_rank(window):
            if len(window) < 2:
                return np.nan
            current_val = window[-1]
            less_than_count = np.sum(window[:-1] < current_val)
            return less_than_count / (len(window) - 1)
            
        df_meta['prob_bull_pct'] = df_meta['pred_prob_bull'].rolling(window=252, min_periods=100).apply(pct_rank, raw=True)
        df_meta['prob_bear_pct'] = df_meta['pred_prob_bear'].rolling(window=252, min_periods=100).apply(pct_rank, raw=True)
        
        # Filter matching dates
        dates = underlying_df.index.intersection(regime_df.index).intersection(df_meta.index)
        dates = sorted(dates)
        
        logger.info(f"Running simulation over {len(dates)} trading days...")
        
        engine = VolatilityEngine(bullish_threshold=bullish_threshold, bearish_threshold=bearish_threshold)
        
        peak_net_liq = self.initial_capital
        shutdown_triggered = False
        
        for i, dt in enumerate(dates):
            date_str = dt.strftime("%Y-%m-%d")
            close_price = underlying_df.loc[dt, 'close']
            regime = regime_df.loc[dt, 'regime'] if enable_regime else "NEUTRAL"
            vix_p = regime_df.loc[dt, 'vix_percentile_1y']
            vix_val = vix_series.loc[dt] if dt in vix_series.index else 18.0
            
            # Update risk engine's portfolio value daily
            self.risk_engine.portfolio_value = self.net_liq
            
            # Step A: Update active positions and mark-to-market
            self._update_positions(date_str, close_price, regime, vix_val)
            
            # Track peak and check drawdown shutdown (15% limit)
            if self.net_liq > peak_net_liq:
                peak_net_liq = self.net_liq
            
            drawdown = (peak_net_liq - self.net_liq) / peak_net_liq
            if drawdown >= 0.15:
                if not shutdown_triggered:
                    logger.critical(f"CRITICAL DRAWDOWN SHUTDOWN: Net liquidation value (${self.net_liq:.2f}) dropped {(drawdown*100):.2f}% below peak (${peak_net_liq:.2f}). Halting all trading activities.")
                    shutdown_triggered = True
                
                # Close all active positions immediately to preserve capital
                active_pos_copy = list(self.active_positions)
                closed_successfully = []
                for pos in active_pos_copy:
                    try:
                        self._close_position(pos, date_str, close_price, "SYSTEM_DRAWDOWN_SHUTDOWN", vix_val)
                        closed_successfully.append(pos)
                    except Exception as e:
                        logger.error(f"Failed to close position {pos['ticker']} {pos['strategy']} during drawdown shutdown: {str(e)}")
                
                closed_set = set(id(p) for p in closed_successfully)
                self.active_positions = [p for p in self.active_positions if id(p) not in closed_set]
            
            # Step B: Check for exit rules on existing positions (if not already shutdown)
            if not shutdown_triggered:
                iv_p = vix_p * 100.0 if not pd.isna(vix_p) else 50.0
                self._check_exits(date_str, close_price, regime, vix_val, iv_p)
                
            # Check rolling expectancy of last 100 closed trades (min 20 trades to activate)
            is_expectancy_paused = False
            if len(self.trade_log) >= 20:
                recent_trades = self.trade_log[-100:]
                rolling_expectancy = sum(t['net_pnl'] for t in recent_trades) / len(recent_trades)
                if rolling_expectancy < 0:
                    is_expectancy_paused = True
                    logger.debug(f"Strategy Entry Paused on {date_str}: Rolling expectancy is negative (${rolling_expectancy:.2f}).")
            
            # Step C: Check for new entry signals
            # Avoid duplicating positions for same ticker if already in trade
            has_active_ticker = any(p['ticker'] == ticker for p in self.active_positions)
            
            # Calculate current margin utilization
            current_margin = sum(p['margin_required'] for p in self.active_positions)
            margin_room = (self.net_liq * self.max_portfolio_exposure_pct) - current_margin
            
            should_block_entry = is_expectancy_paused and enable_expectancy_pause
            
            if not shutdown_triggered and not pd.isna(vix_p) and not should_block_entry and not has_active_ticker and margin_room > 0:
                # Calculate IV Percentile (using QQQ IV proxy or VIX percentile)
                iv_p = vix_p * 100.0 if not pd.isna(vix_p) else 50.0  # Fallback to VIX percentile
                
                # Check earnings proximity (random stub for backtest, filter out 3 days of earnings every 90 days)
                earnings_prox = (i % 90) < 3
                
                # Extract stacked classifier probabilities & metrics
                prob_bull = float(df_meta.loc[dt, 'pred_prob_bull'])
                prob_bear = float(df_meta.loc[dt, 'pred_prob_bear'])
                prob_bull_pct = float(df_meta.loc[dt, 'prob_bull_pct'])
                prob_bear_pct = float(df_meta.loc[dt, 'prob_bear_pct'])
                meta_prob = float(df_meta.loc[dt, 'meta_prob'])
                iv_rv_spread = float(df_meta.loc[dt, 'iv_rv_spread'])
                vix_roc_5d = float(df_meta.loc[dt, 'vix_roc_5d'])
                vix_ratio = float(df_meta.loc[dt, 'vix_ratio'])
                
                rec = engine.select_strategy(
                    ticker=ticker,
                    prob_bull_pct=prob_bull_pct,
                    prob_bear_pct=prob_bear_pct,
                    meta_prob=meta_prob,
                    iv_rv_spread=iv_rv_spread,
                    vix_roc_5d=vix_roc_5d,
                    vix_ratio=vix_ratio,
                    iv_percentile=iv_p,
                    regime=regime,
                    earnings_proximity=earnings_prox,
                    prob_bull=prob_bull,
                    prob_bear=prob_bear
                )
                
                if rec['strategy'] != "HOLD":
                    self._enter_trade(date_str, close_price, rec, pipeline, vix_val)
                    
            # Step D: Log daily state
            total_margin = sum(p['margin_required'] for p in self.active_positions)
            self.daily_log.append({
                "date": date_str,
                "cash": self.cash,
                "net_liq": self.net_liq,
                "active_trades": len(self.active_positions),
                "margin_used": total_margin,
                "utilization_pct": (total_margin / self.net_liq) * 100.0 if self.net_liq > 0.0 else 0.0,
                "expectancy_paused": is_expectancy_paused
            })
            
        return self._compile_results()

    def _update_positions(self, date_str: str, spot_price: float, regime: str, vix_val: float):
        """
        Updates the mark-to-market net liquidation value of active positions.
        """
        total_pos_value = 0.0
        for pos in self.active_positions:
            t_elapsed = (pd.to_datetime(date_str) - pd.to_datetime(pos['entry_date'])).days
            
            # Calculate and update Greeks dynamically daily
            short_dte_limit = pos.get('short_dte', pos['initial_dte'])
            long_dte_limit = pos.get('long_dte', pos['initial_dte'])
            days_left_short = max(short_dte_limit - t_elapsed, 0)
            days_left_long = max(long_dte_limit - t_elapsed, 0)
            
            iv = vix_val / 100.0 if not pd.isna(vix_val) else 0.18
            
            if pos['strategy'] == "SELL_IRON_CONDOR":
                # Compute Greeks and prices for all 4 legs independently
                short_put_greeks = calculate_option_greeks(spot_price, pos['short_put_strike'], days_left_short, iv, False)
                long_put_greeks = calculate_option_greeks(spot_price, pos['long_put_strike'], days_left_long, iv, False)
                short_call_greeks = calculate_option_greeks(spot_price, pos['short_call_strike'], days_left_short, iv, True)
                long_call_greeks = calculate_option_greeks(spot_price, pos['long_call_strike'], days_left_long, iv, True)
                
                # Option package current price (debit buyback cost)
                current_price = (short_put_greeks['price'] + short_call_greeks['price']) - (long_put_greeks['price'] + long_call_greeks['price'])
                current_price = max(current_price, 0.01)
                
                pos['current_val'] = -current_price * pos['contracts'] * 100
                pos['pnl'] = (pos['entry_premium'] - current_price) * pos['contracts'] * 100
                
                # Net Greeks: (Long Put + Long Call) - (Short Put + Short Call)
                pos['delta'] = (long_put_greeks['delta'] + long_call_greeks['delta']) - (short_put_greeks['delta'] + short_call_greeks['delta'])
                pos['gamma'] = (long_put_greeks['gamma'] + long_call_greeks['gamma']) - (short_put_greeks['gamma'] + short_call_greeks['gamma'])
                pos['vega'] = (long_put_greeks['vega'] + long_call_greeks['vega']) - (short_put_greeks['vega'] + short_call_greeks['vega'])
                pos['theta'] = (long_put_greeks['theta'] + long_call_greeks['theta']) - (short_put_greeks['theta'] + short_call_greeks['theta'])
                
                pos['current_option_price'] = current_price
                pos['short_delta'] = max(abs(short_put_greeks['delta']), abs(short_call_greeks['delta']))
            else:
                short_greeks = calculate_option_greeks(spot_price, pos['short_strike'], days_left_short, iv, pos['is_call'])
                long_greeks = calculate_option_greeks(spot_price, pos['long_strike'], days_left_long, iv, pos['is_call'])
                
                # Reprice spread daily using Black-Scholes prices
                bs_price = abs(long_greeks['price'] - short_greeks['price'])
                current_price = max(bs_price, 0.01)
                
                if pos['is_premium_selling']:
                    pos['current_val'] = -current_price * pos['contracts'] * 100
                    pos['pnl'] = (pos['entry_premium'] - current_price) * pos['contracts'] * 100
                else:
                    pos['current_val'] = current_price * pos['contracts'] * 100
                    pos['pnl'] = (current_price - pos['entry_premium']) * pos['contracts'] * 100
                    
                pos['delta'] = long_greeks['delta'] - short_greeks['delta']
                pos['gamma'] = long_greeks['gamma'] - short_greeks['gamma']
                pos['vega'] = long_greeks['vega'] - short_greeks['vega']
                pos['theta'] = long_greeks['theta'] - short_greeks['theta']
                
                # Store current price and short strike delta to avoid duplicate calculations in _check_exits
                pos['current_option_price'] = current_price
                pos['short_delta'] = short_greeks['delta']
            
            total_pos_value += pos['current_val']
            
        self.net_liq = self.cash + total_pos_value

    def _check_exits(self, date_str: str, spot_price: float, regime: str, vix_val: float, iv_percentile: float):
        """
        Evaluates active positions against exit rules using the RiskEngine.
        """
        exited_positions = []
        for pos in self.active_positions:
            days_held = (pd.to_datetime(date_str) - pd.to_datetime(pos['entry_date'])).days
            days_left = max(pos.get('short_dte', pos['initial_dte']) - days_held, 0)
            
            # Volatility Profit Take: Short premium positions only
            if pos.get('is_premium_selling', False) and 'entry_iv_percentile' in pos:
                iv_contraction = pos['entry_iv_percentile'] - iv_percentile
                if iv_contraction >= self.iv_contraction_threshold:
                    self._close_position(pos, date_str, spot_price, "VOL_PROFIT_TAKE", vix_val)
                    exited_positions.append(pos)
                    continue
            
            # Retrieve cached option price and delta from _update_positions
            current_option_price = pos.get('current_option_price', 0.01)
            short_delta = pos.get('short_delta', 0.0)
            
            exit_trade, exit_reason = self.risk_engine.evaluate_exits(
                position=pos,
                spot_price=spot_price,
                current_option_price=current_option_price,
                current_short_strike_delta=short_delta,
                days_to_expiry=days_left,
                regime=regime
            )

            if exit_trade:
                self._close_position(pos, date_str, spot_price, exit_reason, vix_val)
                exited_positions.append(pos)
                
        # Remove exited from active list in O(N) using id set filter
        if exited_positions:
            exited_set = set(id(pos) for pos in exited_positions)
            self.active_positions = [pos for pos in self.active_positions if id(pos) not in exited_set]

    def _enter_trade(self, date_str: str, spot_price: float, rec: Dict[str, Any], pipeline: DataPipeline, vix_val: float):
        """
        Simulates entry execution, including spread pricing, slippage, commissions, and position sizing.
        """
        is_premium_selling = rec['is_premium_selling']
        strategy = rec['strategy']
        ticker = rec['ticker']
        use_wide_buffer = rec.get('use_wide_buffer', False)
        
        initial_dte = 30
        short_dte = 30
        long_dte = 30
        
        # 1. Strike Selection with Volatility-Skew Buffer
        if strategy in ["SELL_BULL_PUT_SPREAD", "SELL_NARROW_BULL_PUT_SPREAD"]:
            if use_wide_buffer:
                short_strike = round(spot_price * 0.94)
                long_strike = round(spot_price * 0.91)
            else:
                short_strike = round(spot_price * 0.97)
                long_strike = round(spot_price * 0.94)
            spread_width = short_strike - long_strike
            is_call = False
            expected_legs = 2
        elif strategy == "SELL_BEAR_CALL_SPREAD":
            if use_wide_buffer:
                short_strike = round(spot_price * 1.06)
                long_strike = round(spot_price * 1.09)
            else:
                short_strike = round(spot_price * 1.03)
                long_strike = round(spot_price * 1.06)
            spread_width = long_strike - short_strike
            is_call = True
            expected_legs = 2
        elif strategy == "SELL_IRON_CONDOR":
            if use_wide_buffer:
                short_put_strike = round(spot_price * 0.92)
                long_put_strike = round(spot_price * 0.88)
                short_call_strike = round(spot_price * 1.08)
                long_call_strike = round(spot_price * 1.12)
            else:
                short_put_strike = round(spot_price * 0.96)
                long_put_strike = round(spot_price * 0.92)
                short_call_strike = round(spot_price * 1.04)
                long_call_strike = round(spot_price * 1.08)
            spread_width = max(short_put_strike - long_put_strike, long_call_strike - short_call_strike)
            is_call = True  # dummy value
            expected_legs = 4
        elif strategy == "BUY_CALL_DEBIT_SPREAD":
            long_strike = round(spot_price * 1.00)
            short_strike = round(spot_price * 1.03)
            spread_width = short_strike - long_strike
            is_call = True
            expected_legs = 2
        elif strategy == "BUY_PUT_DEBIT_SPREAD":
            long_strike = round(spot_price * 1.00)
            short_strike = round(spot_price * 0.97)
            spread_width = long_strike - short_strike
            is_call = False
            expected_legs = 2
        elif strategy == "CALENDAR_SPREAD":
            short_strike = round(spot_price)
            long_strike = round(spot_price)
            spread_width = 0.0
            short_dte = 14
            long_dte = 30
            is_call = True
            expected_legs = 2
        else:
            short_strike = round(spot_price)
            long_strike = round(spot_price * 0.98)
            spread_width = short_strike - long_strike
            is_call = True
            expected_legs = 2
            
        iv_val = vix_val / 100.0 if not pd.isna(vix_val) else 0.18
        option_chain = pipeline.generate_synthetic_options_chain(ticker, date_str, spot_price, iv_val)
        
        date_dt = pd.to_datetime(date_str)
        short_expiry = (date_dt + pd.Timedelta(days=short_dte)).strftime("%Y-%m-%d")
        long_expiry = (date_dt + pd.Timedelta(days=long_dte)).strftime("%Y-%m-%d")
        
        # 2. Query option chain legs individually to avoid row-indexing traps
        if strategy == "SELL_IRON_CONDOR":
            short_put_opt = option_chain[(option_chain['strike'] == short_put_strike) & (option_chain['type'] == 'put') & (option_chain['expiry'] == short_expiry)]
            long_put_opt = option_chain[(option_chain['strike'] == long_put_strike) & (option_chain['type'] == 'put') & (option_chain['expiry'] == long_expiry)]
            short_call_opt = option_chain[(option_chain['strike'] == short_call_strike) & (option_chain['type'] == 'call') & (option_chain['expiry'] == short_expiry)]
            long_call_opt = option_chain[(option_chain['strike'] == long_call_strike) & (option_chain['type'] == 'call') & (option_chain['expiry'] == long_expiry)]
            
            if short_put_opt.empty or long_put_opt.empty or short_call_opt.empty or long_call_opt.empty:
                return # Skip if data is missing
                
            short_mid = ((short_put_opt.iloc[0]['bid'] + short_put_opt.iloc[0]['ask']) / 2 + 
                         (short_call_opt.iloc[0]['bid'] + short_call_opt.iloc[0]['ask']) / 2)
            long_mid = ((long_put_opt.iloc[0]['bid'] + long_put_opt.iloc[0]['ask']) / 2 + 
                        (long_call_opt.iloc[0]['bid'] + long_call_opt.iloc[0]['ask']) / 2)
            
            num_legs = 4
            package_slippage = self.option_slippage * num_legs
            entry_premium = short_mid - long_mid - package_slippage
        else:
            opt_type = "call" if is_call else "put"
            short_opt = option_chain[(option_chain['strike'] == short_strike) & (option_chain['type'] == opt_type) & (option_chain['expiry'] == short_expiry)]
            long_opt = option_chain[(option_chain['strike'] == long_strike) & (option_chain['type'] == opt_type) & (option_chain['expiry'] == long_expiry)]
            
            if short_opt.empty or long_opt.empty:
                return # Skip if data is missing
                
            short_mid = (short_opt.iloc[0]['bid'] + short_opt.iloc[0]['ask']) / 2
            long_mid = (long_opt.iloc[0]['bid'] + long_opt.iloc[0]['ask']) / 2
            
            num_legs = 2
            package_slippage = self.option_slippage * num_legs
            
            if is_premium_selling:
                entry_premium = short_mid - long_mid - package_slippage
            else:
                entry_premium = long_mid - short_mid + package_slippage
                
        # 3. Dynamic Friction-to-Premium Ratio (FPR) Gate
        is_premium_selling = rec.get('is_premium_selling', False)
        gross_premium = short_mid - long_mid if is_premium_selling else long_mid - short_mid
        
        if gross_premium <= 0:
            logger.info(f"Entry REJECTED on {date_str}: Gross premium {gross_premium:.2f} <= 0")
            return
            
        friction_to_premium_ratio = package_slippage / gross_premium
        if friction_to_premium_ratio > self.max_permissible_fpr:
            logger.info(f"Entry REJECTED on {date_str}: Friction ratio {friction_to_premium_ratio:.2%} exceeds limit of {self.max_permissible_fpr:.2%}")
            return
            
        if is_premium_selling:
            if entry_premium < 0.10:
                logger.info(f"Entry REJECTED on {date_str}: Net premium {entry_premium:.2f} < 0.10")
                return
        else:
            if entry_premium >= spread_width and strategy != "CALENDAR_SPREAD":
                logger.info(f"Entry REJECTED on {date_str}: Net debit {entry_premium:.2f} >= Spread {spread_width:.2f}")
                return
                
        entry_premium = max(entry_premium, 0.05)
        
        # 4. Position Sizing
        if is_premium_selling:
            max_loss_per_contract = (spread_width - entry_premium) * 100
            margin_required = spread_width * 100
        else:
            max_loss_per_contract = entry_premium * 100
            margin_required = entry_premium * 100
            
        if max_loss_per_contract <= 0:
            return
            
        max_loss_dollars = self.net_liq * self.max_position_risk_pct
        contracts = int(max_loss_dollars / max_loss_per_contract)
        
        if contracts <= 0:
            return # Capital too small
            
        commissions = contracts * num_legs * self.commission_per_contract
        
        # Verify margin room
        if (margin_required * contracts) > (self.cash - commissions):
            if margin_required <= 0:
                return
            contracts = int((self.cash - commissions) / margin_required)
            if contracts <= 0:
                return
                
        new_trade = {
            "ticker": ticker,
            "direction": rec['direction'],
            "margin_required": margin_required * contracts,
            "max_loss": max_loss_per_contract * contracts,
            "is_premium_selling": is_premium_selling
        }
        
        # Check risk engine
        ok, msg = self.risk_engine.check_exposure_persistence(self.active_positions, new_trade)
        if not ok:
            logger.info(f"Entry BLOCKED by Risk Engine: {msg}")
            return
            
        # Update cash
        if is_premium_selling:
            self.cash += (entry_premium * contracts * 100) - commissions
        else:
            self.cash -= (entry_premium * contracts * 100) + commissions
            
        # 5. Extract initial Greeks
        if strategy == "SELL_IRON_CONDOR":
            pos_delta = (long_put_opt.iloc[0]['delta'] + long_call_opt.iloc[0]['delta']) - (short_put_opt.iloc[0]['delta'] + short_call_opt.iloc[0]['delta'])
            pos_gamma = (long_put_opt.iloc[0]['gamma'] + long_call_opt.iloc[0]['gamma']) - (short_put_opt.iloc[0]['gamma'] + short_call_opt.iloc[0]['gamma'])
            pos_vega = (long_put_opt.iloc[0]['vega'] + long_call_opt.iloc[0]['vega']) - (short_put_opt.iloc[0]['vega'] + short_call_opt.iloc[0]['vega'])
            pos_theta = (long_put_opt.iloc[0]['theta'] + long_call_opt.iloc[0]['theta']) - (short_put_opt.iloc[0]['theta'] + short_call_opt.iloc[0]['theta'])
        else:
            pos_delta = long_opt.iloc[0]['delta'] - short_opt.iloc[0]['delta']
            pos_gamma = long_opt.iloc[0]['gamma'] - short_opt.iloc[0]['gamma']
            pos_vega = long_opt.iloc[0]['vega'] - short_opt.iloc[0]['vega']
            pos_theta = long_opt.iloc[0]['theta'] - short_opt.iloc[0]['theta']
            
        position = {
            "ticker": ticker,
            "strategy": strategy,
            "direction": rec['direction'],
            "entry_date": date_str,
            "entry_spot": spot_price,
            "spread_width": spread_width,
            "initial_dte": short_dte,
            "short_dte": short_dte,
            "long_dte": long_dte,
            "entry_premium": entry_premium,
            "is_premium_selling": is_premium_selling,
            "entry_iv_percentile": rec.get('iv_percentile', 50.0),
            "contracts": contracts,
            "margin_required": margin_required * contracts,
            "max_loss": max_loss_per_contract * contracts,
            "pnl": 0.0,
            "delta": pos_delta,
            "gamma": pos_gamma,
            "vega": pos_vega,
            "theta": pos_theta
        }
        
        if strategy == "SELL_IRON_CONDOR":
            position.update({
                "short_put_strike": short_put_strike,
                "long_put_strike": long_put_strike,
                "short_call_strike": short_call_strike,
                "long_call_strike": long_call_strike,
                "short_strike": short_call_strike,  # Compatibility
                "long_strike": long_call_strike,    # Compatibility
                "is_call": True,                     # Compatibility
                "current_val": -entry_premium * contracts * 100
            })
        else:
            position.update({
                "short_strike": short_strike,
                "long_strike": long_strike,
                "is_call": is_call,
                "current_val": -entry_premium * contracts * 100 if is_premium_selling else entry_premium * contracts * 100
            })
            
        self.active_positions.append(position)
        logger.info(f"Entered {strategy} on {ticker} at spot {spot_price:.2f}. Contracts: {contracts}, Credit/Debit: {entry_premium:.2f}")

    def _close_position(self, pos: Dict[str, Any], date_str: str, spot_price: float, reason: str, vix_val: float):
        """
        Executes position liquidation, records PnL, and logs the trade.
        """
        # Recalculate commissions on exit
        num_legs = 4 if pos['strategy'] == "SELL_IRON_CONDOR" else 2
        exit_commissions = pos['contracts'] * num_legs * self.commission_per_contract
        
        # Calculate exit pricing dynamically using BS on exit date
        t_elapsed = (pd.to_datetime(date_str) - pd.to_datetime(pos['entry_date'])).days
        short_dte_limit = pos.get('short_dte', pos['initial_dte'])
        long_dte_limit = pos.get('long_dte', pos['initial_dte'])
        days_left_short = max(short_dte_limit - t_elapsed, 0)
        days_left_long = max(long_dte_limit - t_elapsed, 0)
        
        is_expiration = (reason == "EXPIRATION" or days_left_short <= 0)
        
        if is_expiration:
            exit_premium = self._calculate_intrinsic_value(pos, spot_price)
            
            # Implementation Guard: check if ETF options sit within 0.5% or ITM of short strikes
            is_near_or_itm = False
            strategy = pos['strategy']
            if strategy in ["SELL_BULL_PUT_SPREAD", "SELL_NARROW_BULL_PUT_SPREAD"]:
                is_near_or_itm = spot_price <= pos['short_strike'] * 1.005
            elif strategy == "SELL_BEAR_CALL_SPREAD":
                is_near_or_itm = spot_price >= pos['short_strike'] * 0.995
            elif strategy == "SELL_IRON_CONDOR":
                is_near_or_itm = (spot_price <= pos['short_put_strike'] * 1.005) or (spot_price >= pos['short_call_strike'] * 0.995)
            elif strategy == "BUY_CALL_DEBIT_SPREAD":
                is_near_or_itm = spot_price >= pos['long_strike'] * 0.995
            elif strategy == "BUY_PUT_DEBIT_SPREAD":
                is_near_or_itm = spot_price <= pos['long_strike'] * 1.005
            else:
                is_near_or_itm = True # Conservative default for other strategies
                
            if is_near_or_itm:
                package_slippage = self.option_slippage * num_legs
            else:
                package_slippage = 0.0 # True expiration without slippage (options expire worthless or cash cleared out of ITM)
        else:
            if pos['strategy'] == "SELL_IRON_CONDOR":
                iv = vix_val / 100.0 if not pd.isna(vix_val) else 0.18
                short_put_greeks = calculate_option_greeks(spot_price, pos['short_put_strike'], days_left_short, iv, False)
                long_put_greeks = calculate_option_greeks(spot_price, pos['long_put_strike'], days_left_long, iv, False)
                short_call_greeks = calculate_option_greeks(spot_price, pos['short_call_strike'], days_left_short, iv, True)
                long_call_greeks = calculate_option_greeks(spot_price, pos['long_call_strike'], days_left_long, iv, True)
                exit_premium = (short_put_greeks['price'] + short_call_greeks['price']) - (long_put_greeks['price'] + long_call_greeks['price'])
            else:
                iv = vix_val / 100.0 if not pd.isna(vix_val) else 0.18
                short_greeks = calculate_option_greeks(spot_price, pos['short_strike'], days_left_short, iv, pos['is_call'])
                long_greeks = calculate_option_greeks(spot_price, pos['long_strike'], days_left_long, iv, pos['is_call'])
                exit_premium = abs(long_greeks['price'] - short_greeks['price'])
                
            exit_premium = max(exit_premium, 0.01)
            package_slippage = self.option_slippage * num_legs
            
        # 2. Apply directional exit slippage
        if pos['is_premium_selling']:
            # Credit exit: buyback cost increases
            exit_premium_adjusted = exit_premium + package_slippage
            cash_received = -exit_premium_adjusted * pos['contracts'] * 100
            gross_pnl = (pos['entry_premium'] - exit_premium_adjusted) * pos['contracts'] * 100
        else:
            # Debit exit: credit recovery decreases
            exit_premium_adjusted = max(exit_premium - package_slippage, 0.01)
            cash_received = exit_premium_adjusted * pos['contracts'] * 100
            gross_pnl = (exit_premium_adjusted - pos['entry_premium']) * pos['contracts'] * 100
            
        net_pnl = gross_pnl - (2 * exit_commissions)
        self.cash += (cash_received - exit_commissions)
        
        # Log trade details
        self.trade_log.append({
            "ticker": pos['ticker'],
            "strategy": pos['strategy'],
            "entry_date": pos['entry_date'],
            "exit_date": date_str,
            "entry_spot": pos['entry_spot'],
            "exit_spot": spot_price,
            "contracts": pos['contracts'],
            "net_pnl": net_pnl,
            "r_multiple": net_pnl / pos['max_loss'] if pos['max_loss'] > 0.0 else 0.0,
            "exit_reason": reason,
            "expectancy_paused_shadow": (
                len(self.trade_log) >= 20 and 
                (sum(t['net_pnl'] for t in self.trade_log[-100:]) / len(self.trade_log[-100:])) < 0
            )
        })
        
        r_mult = net_pnl / pos['max_loss'] if pos['max_loss'] > 0.0 else 0.0
        logger.info(f"Closed {pos['strategy']} on {pos['ticker']} on {date_str} due to {reason}. Net PnL: ${net_pnl:.2f} ({r_mult:.2f}R)")

    def _calculate_intrinsic_value(self, pos: Dict[str, Any], spot_price: float) -> float:
        """
        Calculates the intrinsic value of the option spread at spot price.
        """
        s = spot_price
        
        if pos['strategy'] == "SELL_IRON_CONDOR":
            put_val = max(pos['short_put_strike'] - s, 0) - max(pos['long_put_strike'] - s, 0)
            call_val = max(s - pos['short_call_strike'], 0) - max(s - pos['long_call_strike'], 0)
            return put_val + call_val
            
        k_short = pos['short_strike']
        k_long = pos['long_strike']
        
        if pos['strategy'] in ["SELL_BULL_PUT_SPREAD", "SELL_NARROW_BULL_PUT_SPREAD"]:
            # Put Credit Spread: short_strike > long_strike
            short_value = max(k_short - s, 0)
            long_value = max(k_long - s, 0)
            return short_value - long_value
            
        elif pos['strategy'] == "SELL_BEAR_CALL_SPREAD":
            # Call Credit Spread: short_strike < long_strike
            short_value = max(s - k_short, 0)
            long_value = max(s - k_long, 0)
            return short_value - long_value
            
        elif pos['strategy'] == "BUY_CALL_DEBIT_SPREAD":
            # Call Debit Spread: long_strike < short_strike
            long_value = max(s - k_long, 0)
            short_value = max(s - k_short, 0)
            return long_value - short_value
            
        elif pos['strategy'] == "BUY_PUT_DEBIT_SPREAD":
            # Put Debit Spread: long_strike > short_strike
            long_value = max(k_long - s, 0)
            short_value = max(k_short - s, 0)
            return long_value - short_value
            
        return 0.0

    def _compile_results(self) -> Dict[str, Any]:
        """
        Computes final backtest statistics: expectancy, Sortino, drawdowns, ROBP.
        """
        if not self.trade_log:
            logger.warning("No trades executed during backtest.")
            return {"status": "NO_TRADES"}
            
        df_trades = pd.DataFrame(self.trade_log)
        df_daily = pd.DataFrame(self.daily_log)
        
        # Basic counts
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['net_pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # PnL stats
        total_pnl = self.net_liq - self.initial_capital
        total_return_pct = (total_pnl / self.initial_capital) * 100.0
        
        # Expectancy metrics (portable)
        avg_r_multiple = df_trades['r_multiple'].mean()
        avg_pnl_per_trade = df_trades['net_pnl'].mean()
        
        # Calculate daily returns and drawdown
        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_daily.set_index('date', inplace=True)
        df_daily['daily_return'] = df_daily['net_liq'].pct_change()
        
        # Peak and drawdown
        df_daily['peak'] = df_daily['net_liq'].cummax()
        df_daily['drawdown'] = (df_daily['net_liq'] - df_daily['peak']) / df_daily['peak']
        max_drawdown = df_daily['drawdown'].min() * 100.0
        
        # Sharpe and Sortino (annualized, 252 trading days)
        daily_std = df_daily['daily_return'].std()
        ann_return = df_daily['daily_return'].mean() * 252
        ann_vol = daily_std * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
        
        downside_returns = df_daily[df_daily['daily_return'] < 0]['daily_return']
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = ann_return / downside_std if downside_std > 0 else 0.0
        
        # Capital efficiency metrics
        avg_utilization = df_daily['utilization_pct'].mean()
        robp = (ann_return / (avg_utilization / 100.0)) * 100.0 if avg_utilization > 0 else 0.0
        idle_capital_pct = 100.0 - avg_utilization
        
        # Selectivity (percentage of trading days where we had at least one active trade)
        active_days = len(df_daily[df_daily['active_trades'] > 0])
        selectivity = (active_days / len(df_daily)) * 100.0
        
        # Profit factor
        gross_profits = df_trades[df_trades['net_pnl'] > 0]['net_pnl'].sum()
        gross_losses = abs(df_trades[df_trades['net_pnl'] < 0]['net_pnl'].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Portfolio risk basis points (bps) per trade
        # average pnl as bps of portfolio net liq
        avg_trade_bps = (avg_pnl_per_trade / self.initial_capital) * 10000.0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate * 100.0,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "avg_r_multiple": avg_r_multiple,
            "avg_trade_bps": avg_trade_bps,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_drawdown,
            "avg_portfolio_utilization_pct": avg_utilization,
            "margin_efficiency_robp": robp,
            "idle_capital_pct": idle_capital_pct,
            "selectivity_pct": selectivity,
            "profit_factor": profit_factor
        }

# =========================================================================
# TEST DRIVER
# =========================================================================
if __name__ == "__main__":
    logger.info("Running OptionsBacktester basic validation...")
    
    # Generate mock timeline
    dates = pd.date_range(start="2020-01-01", periods=500, freq="B")
    
    # Underlying prices
    prices = [400.0]
    for i in range(1, 500):
        prices.append(prices[-1] * (1.0 + np.random.normal(0.0005, 0.01)))
    underlying_df = pd.DataFrame({"close": prices}, index=dates)
    
    # VIX close
    vix = np.random.normal(18.0, 3.0, 500)
    vix_series = pd.Series(vix, index=dates)
    
    # Kronos scores
    kronos = np.random.normal(0.1, 0.25, 500)
    kronos_series = pd.Series(kronos, index=dates)
    
    pipeline = DataPipeline(cache_dir="./test_cache")
    backtester = OptionsBacktester()
    
    results = backtester.run_backtest(underlying_df, vix_series, kronos_series, pipeline)
    print("\n--- 📊 Simulated Options Backtest Results ---")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
