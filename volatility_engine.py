"""
volatility_engine.py - Multi-Factor Quant Research & Trading Platform
Selects the optimal options strategy based on Kronos direction signals, 
IV Percentile (IVP), market regime, and earnings proximity.
"""

import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger("quant_system.volatility_engine")

class VolatilityEngine:
    """
    Implements the restructured convexity-capture strategy matrix.
    Uses out-of-sample probability signals, IV-RV spreads, and VIX panic/hysteresis overrides.
    """
    
    def __init__(self, bullish_threshold: float = 0.80, bearish_threshold: float = 0.80):
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        logger.info(f"Volatility engine initialized with thresholds: Bullish {self.bullish_threshold}, Bearish {self.bearish_threshold}")

    def select_strategy(
        self,
        ticker: str,
        prob_bull_pct: float,
        prob_bear_pct: float,
        meta_prob: float,
        iv_rv_spread: float,
        vix_roc_5d: float,
        vix_ratio: float,
        iv_percentile: float,  # Range: 0.0 to 100.0
        regime: str,           # 'RISK_ON', 'NEUTRAL', 'RISK_OFF', 'PANIC'
        earnings_proximity: bool = False,
        prob_bull: float = 0.5,
        prob_bear: float = 0.5
    ) -> Dict[str, Any]:
        """
        Maps inputs to optimal options strategies.
        Returns a dictionary containing the recommended strategy and execution params.
        """
        # Calculate wide buffer conditions (VIX elevated check)
        use_wide_buffer = (iv_percentile > 50.0) or (vix_ratio > 1.05)

        # 0. Check if probability percentiles are NaN (warmup period)
        if pd.isna(prob_bull_pct) or pd.isna(prob_bear_pct):
            return self._hold_recommendation(
                ticker,
                "Warmup period: Percentile indicators are NaN.",
                prob_bull,
                prob_bear,
                iv_percentile,
                regime,
                earnings_proximity,
                use_wide_buffer
            )

        # 1. PANIC Regime Override (Stateful panic block)
        if regime == "PANIC":
            return self._hold_recommendation(ticker, "PANIC regime active. Entry blocked.", prob_bull, prob_bear, iv_percentile, regime, earnings_proximity, use_wide_buffer)
            
        # 2. Level-2 Meta-Classifier Gate
        # Block trade if the success probability is below 50%
        if not pd.isna(meta_prob) and meta_prob < 0.50:
            return self._hold_recommendation(
                ticker, 
                f"Meta-classifier blocked trade (Success Prob: {meta_prob:.2f} < 0.50).", 
                prob_bull, 
                prob_bear, 
                iv_percentile,
                regime,
                earnings_proximity,
                use_wide_buffer
            )
            
        # 3. Determine direction class from out-of-sample probability percentiles
        direction = "NEUTRAL"
        if not pd.isna(prob_bull_pct) and prob_bull_pct >= self.bullish_threshold:
            direction = "BULLISH"
        elif not pd.isna(prob_bear_pct) and prob_bear_pct >= self.bearish_threshold:
            direction = "BEARISH"

        # 4. Determine IV classification
        if iv_percentile > 50.0:
            iv_class = "RICH"
        elif iv_percentile >= 20.0:
            iv_class = "NORMAL"
        else:
            iv_class = "CHEAP"

        # 5. Base Strategy Selection (Convexity Capture focus)
        strategy = "HOLD"
        rationale = ""
        is_premium_selling = False

        if direction == "BULLISH":
            if iv_rv_spread > 0.05:
                strategy = "SELL_BULL_PUT_SPREAD"
                rationale = f"Bullish signal with expensive IV (IV-RV: {iv_rv_spread*100:.2f}%). Sell bull put spread."
                is_premium_selling = True
            else:
                strategy = "BUY_CALL_DEBIT_SPREAD"
                rationale = f"Bullish signal with cheap/fair IV (IV-RV: {iv_rv_spread*100:.2f}%). Buy call debit spread."
                is_premium_selling = False

        elif direction == "BEARISH":
            if iv_rv_spread > 0.05:
                strategy = "SELL_BEAR_CALL_SPREAD"
                rationale = f"Bearish signal with expensive IV (IV-RV: {iv_rv_spread*100:.2f}%). Sell bear call spread."
                is_premium_selling = True
            else:
                strategy = "BUY_PUT_DEBIT_SPREAD"
                rationale = f"Bearish signal with cheap/fair IV (IV-RV: {iv_rv_spread*100:.2f}%). Buy put debit spread."
                is_premium_selling = False

        else: # NEUTRAL
            if iv_rv_spread > 0.05:
                strategy = "SELL_IRON_CONDOR"
                rationale = f"Neutral signal with expensive IV (IV-RV: {iv_rv_spread*100:.2f}%). Sell Iron Condor."
                is_premium_selling = True
            elif regime == "NEUTRAL" and iv_class == "NORMAL" and vix_roc_5d < 0.01:
                strategy = "CALENDAR_SPREAD"
                rationale = "Neutral signal with stable volatility. Buy calendar spread to capture short-term decay."
                is_premium_selling = False
            else:
                strategy = "HOLD"
                rationale = "Neutral signal. Volatility or regime conditions do not support premium selling or calendar trades."
                is_premium_selling = False

        # =========================================================================
        # OVERRIDES (Safety Gates)
        # =========================================================================
        
        # Override 1: RISK_OFF Regime (Disable premium selling, prioritize defensive)
        if regime == "RISK_OFF":
            if is_premium_selling:
                strategy = "HOLD"
                rationale = "OVERRIDE: Risk-Off regime active. Short premium strategy disabled."
                is_premium_selling = False
            elif direction == "BEARISH":
                pass
            else:
                strategy = "HOLD"
                rationale = "OVERRIDE: Risk-Off regime active. Non-bearish positions disabled."

        # Override 2: Earnings Proximity
        if earnings_proximity:
            if is_premium_selling:
                strategy = "HOLD"
                rationale = "OVERRIDE: Earnings announcement within 3 days. Premium-selling disabled."
                is_premium_selling = False

        return {
            "ticker": ticker,
            "direction": direction,
            "iv_class": iv_class,
            "strategy": strategy,
            "is_premium_selling": is_premium_selling,
            "regime": regime,
            "earnings_proximity": earnings_proximity,
            "rationale": rationale,
            "prob_bull": prob_bull,
            "prob_bear": prob_bear,
            "iv_percentile": iv_percentile,
            "use_wide_buffer": use_wide_buffer
        }

    def _hold_recommendation(
        self,
        ticker: str,
        rationale: str,
        prob_bull: float,
        prob_bear: float,
        iv_percentile: float,
        regime: str = "NEUTRAL",
        earnings_proximity: bool = False,
        use_wide_buffer: bool = False
    ) -> Dict[str, Any]:
        """
        Helper to construct a standardized HOLD recommendation.
        """
        # Determine IV classification
        if iv_percentile > 50.0:
            iv_class = "RICH"
        elif iv_percentile >= 20.0:
            iv_class = "NORMAL"
        else:
            iv_class = "CHEAP"

        return {
            "ticker": ticker,
            "direction": "NEUTRAL",
            "iv_class": iv_class,
            "strategy": "HOLD",
            "is_premium_selling": False,
            "regime": regime,
            "earnings_proximity": earnings_proximity,
            "rationale": rationale,
            "prob_bull": prob_bull,
            "prob_bear": prob_bear,
            "iv_percentile": iv_percentile,
            "use_wide_buffer": use_wide_buffer
        }

# =========================================================================
# TEST DRIVER
# =========================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running VolatilityEngine basic validation...")
    engine = VolatilityEngine(bullish_threshold=0.80, bearish_threshold=0.80)
    
    # Test case 1: Bullish + Cheap/Fair IV (Debit Buy)
    res1 = engine.select_strategy(
        ticker="QQQ",
        prob_bull_pct=0.85,
        prob_bear_pct=0.30,
        meta_prob=0.55,
        iv_rv_spread=-0.02,
        vix_roc_5d=0.01,
        vix_ratio=1.0,
        iv_percentile=15.0,
        regime="RISK_ON",
        prob_bull=0.60,
        prob_bear=0.30
    )
    print(f"Test 1: {res1['strategy']} - Rationale: {res1['rationale']}")
    assert res1['strategy'] == "BUY_CALL_DEBIT_SPREAD"
    
    # Test case 2: Bearish + Cheap/Fair IV (Debit Buy)
    res2 = engine.select_strategy(
        ticker="QQQ",
        prob_bull_pct=0.30,
        prob_bear_pct=0.85,
        meta_prob=0.55,
        iv_rv_spread=-0.01,
        vix_roc_5d=0.01,
        vix_ratio=1.0,
        iv_percentile=15.0,
        regime="RISK_ON",
        prob_bull=0.30,
        prob_bear=0.60
    )
    print(f"Test 2: {res2['strategy']} - Rationale: {res2['rationale']}")
    assert res2['strategy'] == "BUY_PUT_DEBIT_SPREAD"
    
    # Test case 3: Neutral + stable vol
    res3 = engine.select_strategy(
        ticker="QQQ",
        prob_bull_pct=0.40,
        prob_bear_pct=0.40,
        meta_prob=0.50,
        iv_rv_spread=0.01,
        vix_roc_5d=0.005,
        vix_ratio=1.0,
        iv_percentile=30.0,
        regime="NEUTRAL",
        prob_bull=0.40,
        prob_bear=0.40
    )
    print(f"Test 3: {res3['strategy']} - Rationale: {res3['rationale']}")
    assert res3['strategy'] == "CALENDAR_SPREAD"
    
    # Test case 4: Panic override
    res4 = engine.select_strategy(
        ticker="QQQ",
        prob_bull_pct=0.85,
        prob_bear_pct=0.30,
        meta_prob=0.55,
        iv_rv_spread=0.06,
        vix_roc_5d=0.10,
        vix_ratio=1.20,
        iv_percentile=80.0,
        regime="PANIC",
        prob_bull=0.60,
        prob_bear=0.30
    )
    print(f"Test 4: {res4['strategy']} - Rationale: {res4['rationale']}")
    assert res4['strategy'] == "HOLD"
    
    # Test case 5: Meta probability block
    res5 = engine.select_strategy(
        ticker="QQQ",
        prob_bull_pct=0.85,
        prob_bear_pct=0.30,
        meta_prob=0.45,
        iv_rv_spread=-0.01,
        vix_roc_5d=0.01,
        vix_ratio=1.0,
        iv_percentile=15.0,
        regime="RISK_ON",
        prob_bull=0.60,
        prob_bear=0.30
    )
    print(f"Test 5: {res5['strategy']} - Rationale: {res5['rationale']}")
    assert res5['strategy'] == "HOLD"
    
    logger.info("VolatilityEngine validation completed successfully.")
