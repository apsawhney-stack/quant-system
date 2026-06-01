"""
risk_engine.py - Multi-Factor Quant Research & Trading Platform
Enforces hard limits, Greeks constraints, exposure persistence controls,
and exit validation rules at the trade and portfolio levels.
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

logger = logging.getLogger("quant_system.risk_engine")

def fast_norm_pdf(x: float) -> float:
    try:
        return math.exp(-x * x / 2.0) / 2.5066282746310002
    except OverflowError:
        return 0.0

def fast_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / 1.4142135623730951))

def calculate_option_greeks(
    spot_price: float,
    strike: float,
    dte: int,
    iv: float,
    is_call: bool,
    r: float = 0.045
) -> Dict[str, float]:
    """
    Calculates Black-Scholes Greeks and price for a single option contract.
    """
    if dte <= 0:
        intrinsic = max(spot_price - strike, 0.0) if is_call else max(strike - spot_price, 0.0)
        if is_call:
            delta = 1.0 if spot_price > strike else 0.0
        else:
            delta = -1.0 if spot_price < strike else 0.0
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "price": float(intrinsic)}
        
    t = max(dte, 1) / 365.0
    vol = max(iv, 0.01)
    
    d1 = (np.log(spot_price / strike) + (r + 0.5 * vol**2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    
    if is_call:
        delta = fast_norm_cdf(d1)
        theta = -(spot_price * fast_norm_pdf(d1) * vol) / (2 * np.sqrt(t)) - r * strike * np.exp(-r * t) * fast_norm_cdf(d2)
        price = spot_price * fast_norm_cdf(d1) - strike * np.exp(-r * t) * fast_norm_cdf(d2)
    else:
        delta = fast_norm_cdf(d1) - 1
        theta = -(spot_price * fast_norm_pdf(d1) * vol) / (2 * np.sqrt(t)) + r * strike * np.exp(-r * t) * fast_norm_cdf(-d2)
        price = strike * np.exp(-r * t) * fast_norm_cdf(-d2) - spot_price * fast_norm_cdf(-d1)
        
    gamma = fast_norm_pdf(d1) / (spot_price * vol * np.sqrt(t))
    vega = spot_price * np.sqrt(t) * fast_norm_pdf(d1)
    
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta / 365.0), # daily theta
        "price": float(max(price, 0.0))
    }

class RiskEngine:
    """
    Core safety and risk monitoring component.
    Protects capital against drawdown, concentration, leverage, and volatility shocks.
    """
    
    def __init__(
        self,
        portfolio_value: float = 200000.0,
        max_trade_risk_pct: float = 0.01,        # 1% per trade
        max_total_exposure_pct: float = 0.25,     # 25% max total exposure
        max_correlated_exposure_pct: float = 0.10, # 10% max same-direction/sector
        max_short_vol_exposure_pct: float = 0.15,  # 15% max short volatility
        max_daily_loss_pct: float = 0.02,         # 2% daily loss limit
        max_weekly_drawdown_pct: float = 0.05     # 5% weekly drawdown limit
    ):
        self.portfolio_value = portfolio_value
        self.max_trade_risk_pct = max_trade_risk_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_correlated_exposure_pct = max_correlated_exposure_pct
        self.max_short_vol_exposure_pct = max_short_vol_exposure_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_drawdown_pct = max_weekly_drawdown_pct
        
        logger.info("RiskEngine initialized with strict institutional-grade limits.")

    # =========================================================================
    # EXPOSURE PERSISTENCE CONTROLS
    # =========================================================================
    
    def check_exposure_persistence(
        self, 
        active_positions: List[Dict[str, Any]], 
        new_trade: Dict[str, Any],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Tuple[bool, str]:
        """
        Rejects new trade if it creates excessive directional clustering or
        exceeds correlated exposure thresholds.
        """
        # Rule 1: Portfolio overall exposure limit
        current_exposure = sum(p['margin_required'] for p in active_positions)
        new_exposure = new_trade['margin_required']
        if (current_exposure + new_exposure) > (self.portfolio_value * self.max_total_exposure_pct):
            return False, f"REJECTED: Would exceed portfolio max exposure ceiling ({self.max_total_exposure_pct*100}% of net liq)."

        # Rule 2: Correlated directional clustering
        # Count existing positions in same direction
        same_direction_pos = [p for p in active_positions if p['direction'] == new_trade['direction']]
        
        # Check pairwise correlation
        correlated_count = 0
        for pos in same_direction_pos:
            # Check correlation coefficient between the underlyings
            corr = 1.0 # Default if same underlying
            if pos['ticker'] != new_trade['ticker']:
                corr = 0.0
                if correlation_matrix:
                    if pos['ticker'] in correlation_matrix:
                        corr = correlation_matrix[pos['ticker']].get(new_trade['ticker'], 0.0)
                    # Fallback to reverse lookup if not found
                    if corr == 0.0 and new_trade['ticker'] in correlation_matrix:
                        corr = correlation_matrix[new_trade['ticker']].get(pos['ticker'], 0.0)
                    
            if corr > 0.70:
                correlated_count += 1
                
        if correlated_count >= 2:
            return False, "REJECTED: Risk clustering limit reached. Already hold 2+ correlated positions expressing the same directional thesis."

        # Rule 3: Max Same-Direction Risk Limits
        same_direction_risk = sum(p['max_loss'] for p in same_direction_pos)
        new_risk = new_trade['max_loss']
        if (same_direction_risk + new_risk) > (self.portfolio_value * self.max_correlated_exposure_pct):
            return False, f"REJECTED: Same-direction risk would exceed the {self.max_correlated_exposure_pct*100}% portfolio limit."

        # Rule 4: Max Short-Vol Limit
        if new_trade.get('is_premium_selling', False):
            current_short_vol = sum(p['margin_required'] for p in active_positions if p.get('is_premium_selling', False))
            if (current_short_vol + new_exposure) > (self.portfolio_value * self.max_short_vol_exposure_pct):
                return False, f"REJECTED: Short volatility exposure would exceed the {self.max_short_vol_exposure_pct*100}% limit."

        return True, "ACCEPTED: Position satisfies all risk limits."

    def filter_co_triggers(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enforces correlation filtering: if multiple correlated equity products trigger on the same day,
        only accept the one with the highest Level-1 prediction strength to avoid tail risk concentration.
        """
        equities_group = ["QQQ", "SPY", "IWM"]
        
        # Filter triggers that are not HOLD
        active_triggers = [r for r in recommendations if r.get('strategy', 'HOLD') != 'HOLD']
        if not active_triggers:
            return recommendations
            
        # Find equity triggers
        equity_triggers = [r for r in active_triggers if r['ticker'] in equities_group]
        
        if len(equity_triggers) > 1:
            # Sort by highest Level-1 probability trigger strength
            # (which is the max of prob_bull_pct and prob_bear_pct)
            def get_strength(r):
                p_bull = r.get('prob_bull_pct', r.get('prob_bull', 0.5))
                p_bear = r.get('prob_bear_pct', r.get('prob_bear', 0.5))
                # Fallback to 0 if NaN
                val = max(p_bull if not np.isnan(p_bull) else 0.0, 
                          p_bear if not np.isnan(p_bear) else 0.0)
                return val
                
            equity_triggers.sort(key=get_strength, reverse=True)
            best_ticker = equity_triggers[0]['ticker']
            
            # Block all other equity triggers by setting strategy to HOLD
            for r in recommendations:
                if r['ticker'] in equities_group and r['ticker'] != best_ticker:
                    r['strategy'] = "HOLD"
                    r['rationale'] = f"OVERRIDE: Co-trigger blocked by Risk Engine. {best_ticker} had higher signal strength."
                    
        return recommendations

    # =========================================================================
    # GREEKS-BASED PORTFOLIO MONITORING
    # =========================================================================
    
    def calculate_portfolio_greeks(self, active_positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregates individual Greeks to the portfolio level.
        Normalizes delta and vega to portfolio percentages.
        """
        net_delta = 0.0
        net_gamma = 0.0
        net_vega = 0.0
        net_theta = 0.0
        
        for pos in active_positions:
            contracts = pos.get('contracts', 0)
            # Greek values should be per contract, multiplied by 100 multiplier
            net_delta += pos.get('delta', 0.0) * contracts * 100
            net_gamma += pos.get('gamma', 0.0) * contracts * 100
            net_vega += pos.get('vega', 0.0) * contracts * 100
            net_theta += pos.get('theta', 0.0) * contracts * 100
            
        # Normalize Net Delta to % of Portfolio Value
        # Net Delta / Portfolio Value
        delta_pct = (net_delta / self.portfolio_value) * 100.0
        
        # Normalize Vega to Portfolio Impact per 1% absolute IV shift
        # Net Vega / Portfolio Value
        vega_pct = (net_vega / self.portfolio_value) * 100.0
        
        return {
            "net_delta_dollars": net_delta,
            "net_delta_pct": delta_pct,
            "net_gamma": net_gamma,
            "net_vega_dollars": net_vega,
            "net_vega_pct": vega_pct,
            "net_theta_dollars": net_theta,
            "theta_decay_pct": (net_theta / self.portfolio_value) * 100.0
        }

    def check_greeks_limits(self, greeks: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Verifies aggregated portfolio Greeks fall within safety bounds.
        """
        warnings = []
        is_safe = True
        
        # 1. Net Delta Limit: < ±30% of portfolio value
        if abs(greeks['net_delta_pct']) > 30.0:
            is_safe = False
            warnings.append(f"CRITICAL: Portfolio Net Delta ({greeks['net_delta_pct']:.2f}%) exceeds safety limit of ±30%.")
            
        # 2. Net Vega Limit: < 0.5% impact per 1-point IV move
        if abs(greeks['net_vega_pct']) > 0.50:
            is_safe = False
            warnings.append(f"CRITICAL: Portfolio Net Vega ({greeks['net_vega_pct']:.2f}%) exceeds safety limit of 0.5% per IV point.")
            
        # 3. Theta Decay Limit: < 0.1% concentrated in single expiry (absolute value check)
        if abs(greeks['theta_decay_pct']) > 0.10:
            warnings.append(f"WARNING: Portfolio Theta decay concentration ({greeks['theta_decay_pct']:.2f}%) is elevated.")
            
        return is_safe, warnings

    # =========================================================================
    # TRADE EXIT VALIDATION RULES
    # =========================================================================
    
    def evaluate_exits(
        self, 
        position: Dict[str, Any], 
        spot_price: float, 
        current_option_price: float,
        current_short_strike_delta: float,
        days_to_expiry: int,
        regime: str,
        current_date: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Determines if a trade should be liquidated early based on Greeks, profit targets, or regimes.
        """
        is_premium_selling = position['is_premium_selling']
        entry_premium = position['entry_premium']
        contracts = position['contracts']
        
        # Calculate current position P&L
        if is_premium_selling:
            pnl = (entry_premium - current_option_price) * contracts * 100
        else:
            pnl = (current_option_price - entry_premium) * contracts * 100
            
        # Exit Rule 1: Expiration or Time Exit (Close if < 7 DTE to avoid gamma acceleration)
        if days_to_expiry <= 7:
            return True, f"TIME_EXIT: {days_to_expiry} days to expiry remaining (gamma risk filter)."
            
        # Decaying profit target calculation using calendar math
        profit_target_pct = 0.70
        if is_premium_selling and current_date is not None and 'entry_date' in position:
            try:
                from datetime import datetime
                entry_dt = datetime.strptime(position['entry_date'], "%Y-%m-%d")
                curr_dt = datetime.strptime(current_date, "%Y-%m-%d")
                days_held = (curr_dt - entry_dt).days
                initial_dte = position.get('initial_dte', 30)
                if initial_dte > 0:
                    time_decay_fraction = days_held / initial_dte
                    profit_target_pct = 0.70 - (0.35 * min(time_decay_fraction, 1.0))
            except Exception as e:
                logger.error(f"Error calculating decaying profit target: {str(e)}")

        # Exit Rule 2: Profit Target
        if is_premium_selling and pnl >= (entry_premium * profit_target_pct * contracts * 100):
            return True, f"PROFIT_TARGET: Dynamic {profit_target_pct*100:.1f}% profit target captured."
        elif not is_premium_selling:
            if position.get('strategy') == "CALENDAR_SPREAD":
                if pnl >= (entry_premium * 0.30 * contracts * 100):
                    return True, "PROFIT_TARGET: 30% of debit captured on calendar spread."
            else:
                max_profit = (position['spread_width'] - entry_premium) * contracts * 100
                if max_profit > 0 and pnl >= (max_profit * 0.50):
                    return True, "PROFIT_TARGET: 50% of max potential profit captured."
                elif max_profit <= 0 and pnl >= (position['max_loss'] * 0.50):
                    return True, "PROFIT_TARGET: 50% of debit captured."
            
        # Exit Rule 3: Max Loss Breach (Exit at 95% of max loss to avoid margin wipes)
        if pnl <= -(position['max_loss'] * 0.95):
            return True, "MAX_LOSS_BREACH: Trade hit standard max risk boundary."
            
        # Exit Rule 4: Short Strike Delta Breach (Delta crosses 0.40)
        if is_premium_selling and abs(current_short_strike_delta) > 0.40:
            return True, f"DELTA_BREACH: Short strike delta reached {current_short_strike_delta:.2f} (exceeds 0.40 safety threshold)."
            
        # Exit Rule 5: Regime Shift Override (Disable premium selling, exit in Risk-Off)
        if regime == "RISK_OFF" and is_premium_selling:
            return True, "REGIME_SHIFT: Volatility regime changed to RISK_OFF while holding short premium."

        return False, "HOLD: Position remains within safe boundaries."

# =========================================================================
# TEST DRIVER
# =========================================================================
if __name__ == "__main__":
    logger.info("Running RiskEngine validation checks...")
    engine = RiskEngine(portfolio_value=200000.0)
    
    # Test case 1: Correlation exposure clustering
    active_pos = [
        {"ticker": "QQQ", "direction": "BULLISH", "margin_required": 10000.0, "max_loss": 5000.0},
        {"ticker": "SPY", "direction": "BULLISH", "margin_required": 10000.0, "max_loss": 5000.0}
    ]
    new_tr = {"ticker": "IWM", "direction": "BULLISH", "margin_required": 10000.0, "max_loss": 5000.0}
    corr_matrix = {"QQQ": {"SPY": 0.85, "IWM": 0.75}, "SPY": {"IWM": 0.72}}
    
    ok, msg = engine.check_exposure_persistence(active_pos, new_tr, corr_matrix)
    print(f"Test 1 (Pairwise corr > 0.70): ok={ok} - Msg: {msg}")
    assert not ok
    
    # Test case 2: Portfolio Greeks limits checking
    mock_greeks = {
        "net_delta_dollars": 80000.0,
        "net_delta_pct": 40.0, # Exceeds 30%
        "net_gamma": 50.0,
        "net_vega_dollars": 200.0,
        "net_vega_pct": 0.10,
        "net_theta_dollars": -50.0,
        "theta_decay_pct": -0.025
    }
    safe, warnings = engine.check_greeks_limits(mock_greeks)
    print(f"Test 2 (Greeks delta limit breach): safe={safe} - Warnings: {warnings}")
    assert not safe
    
    # Test case 3: Exit evaluating delta breach
    mock_pos = {"ticker": "QQQ", "strategy": "SELL_BULL_PUT_SPREAD", "is_premium_selling": True, "entry_premium": 1.0, "contracts": 10, "max_loss": 9000.0}
    exit_needed, reason = engine.evaluate_exits(
        position=mock_pos, 
        spot_price=320.0, 
        current_option_price=2.5, 
        current_short_strike_delta=0.45, # delta breach
        days_to_expiry=15, 
        regime="NEUTRAL"
    )
    print(f"Test 3 (Delta breach): exit={exit_needed} - Reason: {reason}")
    assert exit_needed
    
    logger.info("RiskEngine tests completed successfully.")
