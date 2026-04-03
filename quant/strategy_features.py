from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

from .types import FeatureFn, Row


@dataclass
class FeatureSpec:
    name: str
    fn: FeatureFn


class StrategyFeatureBuilder:
    def __init__(self) -> None:
        self._features: List[FeatureSpec] = []

    def add(self, name: str, fn: FeatureFn) -> "StrategyFeatureBuilder":
        self._features.append(FeatureSpec(name=name, fn=fn))
        return self

    def names(self) -> List[str]:
        return [f.name for f in self._features]

    def transform(self, rows: Sequence[Row]) -> List[List[float]]:
        matrix: List[List[float]] = []
        for row in rows:
            matrix.append([feature.fn(row) for feature in self._features])
        return matrix


FeatureSet = Literal[
    "feature2",
    "new",
    "legacy",
    "fvg2",
    "fvg3",
    "derivative",
    "derivative2",
    "dqn",
    "ema",
    "bollinger_bands",
    "vwap_anchor",
    "vwap_intraday_reversion",
    "vwap_intraday_momentum",
    "vwap_intraday_5m_session",
    "vwap_breakout_reversion_regime",
    "open15_orb_intraday",
    "open15_vwap_reclaim_intraday",
    "open15_trend_momentum_daytrade",
    "open15_dual_breakout_daytrade",
    "open15_dual_breakout_daytrade_plus",
    "open15_dual_breakout_daytrade_scalp",
    "vwap_momentum_trend_5m_conservative",
    "vwap_momentum_trend_5m_pullback",
    "hybrid_sharpe_core",
    "hybrid_sharpe_core_no_stack",
    "hybrid_sharpe_momentum",
    "hybrid_sharpe_selective",
    "hybrid_sharpe_regime",
    "hybrid_sharpe_volume_flow",
    "hybrid_sharpe_volume_regime",
    "close_hold_reversion",
    "close_hold_momentum",
    "war_shock_reversion",
    "war_shock_momentum",
    "rsi_thresholds",
    "stoch_rsi_thresholds",
]


def normalize_feature_set(feature_set: str) -> FeatureSet:
    value = feature_set.strip().lower()
    if value in ("feature2", "v2", "new2", "default"):
        return "feature2"
    if value in ("derivative2", "derivatives2", "derivate2", "deriv2", "ema-derivative2"):
        return "derivative2"
    if value in ("derivative", "derivatives", "deriv", "ema-derivative"):
        return "derivative"
    if value in ("dqn", "deep-q", "deep_q"):
        return "dqn"
    if value in ("ema", "ema-set", "ema_set", "many-ema", "many_ema"):
        return "ema"
    if value in ("bollinger_bands", "bollinger-bands", "bollinger", "bbands", "bb"):
        return "bollinger_bands"
    if value in ("vwap_anchor", "vwap-anchor", "anchored-vwap", "anchored_vwap", "vwap"):
        return "vwap_anchor"
    if value in (
        "vwap_intraday_reversion",
        "vwap-intraday-reversion",
        "intraday-vwap-reversion",
        "vwap_reversion",
        "intraday_reversion",
    ):
        return "vwap_intraday_reversion"
    if value in (
        "vwap_intraday_momentum",
        "vwap-intraday-momentum",
        "intraday-vwap-momentum",
        "vwap_momentum",
        "intraday_momentum",
    ):
        return "vwap_intraday_momentum"
    if value in (
        "vwap_intraday_5m_session",
        "vwap-intraday-5m-session",
        "intraday-vwap-5m-session",
        "vwap_5m_session",
        "session_vwap_5m",
    ):
        return "vwap_intraday_5m_session"
    if value in (
        "vwap_breakout_reversion_regime",
        "vwap-breakout-reversion-regime",
        "vwap_regime",
        "vwap-breakout-vs-reversion",
        "breakout_reversion_vwap",
    ):
        return "vwap_breakout_reversion_regime"
    if value in (
        "open15_orb_intraday",
        "open15-orb-intraday",
        "open15_orb",
        "orb_open15",
        "opening_range_breakout",
    ):
        return "open15_orb_intraday"
    if value in (
        "open15_vwap_reclaim_intraday",
        "open15-vwap-reclaim-intraday",
        "open15_vwap_reclaim",
        "vwap_reclaim_open15",
        "opening_range_vwap_reclaim",
    ):
        return "open15_vwap_reclaim_intraday"
    if value in (
        "open15_trend_momentum_daytrade",
        "open15-trend-momentum-daytrade",
        "open15_trend_momentum",
        "open15_momentum_daytrade",
        "first15_trend_momentum",
    ):
        return "open15_trend_momentum_daytrade"
    if value in (
        "open15_dual_breakout_daytrade",
        "open15-dual-breakout-daytrade",
        "open15_dual_breakout",
        "open15_breakout_followthrough",
        "first15_dual_breakout",
    ):
        return "open15_dual_breakout_daytrade"
    if value in (
        "open15_dual_breakout_daytrade_plus",
        "open15-dual-breakout-daytrade-plus",
        "open15_dual_breakout_plus",
        "open15_breakout_followthrough_plus",
        "first15_dual_breakout_plus",
    ):
        return "open15_dual_breakout_daytrade_plus"
    if value in (
        "open15_dual_breakout_daytrade_scalp",
        "open15-dual-breakout-daytrade-scalp",
        "open15_dual_breakout_scalp",
        "open15_breakout_followthrough_scalp",
        "first15_dual_breakout_scalp",
    ):
        return "open15_dual_breakout_daytrade_scalp"
    if value in (
        "vwap_momentum_trend_5m_conservative",
        "vwap-momentum-trend-5m-conservative",
        "vwap_5m_trend_conservative",
        "vwap_trend_conservative",
    ):
        return "vwap_momentum_trend_5m_conservative"
    if value in (
        "vwap_momentum_trend_5m_pullback",
        "vwap-momentum-trend-5m-pullback",
        "vwap_5m_trend_pullback",
        "vwap_trend_pullback",
    ):
        return "vwap_momentum_trend_5m_pullback"
    if value in ("hybrid_sharpe_core", "hybrid-core", "hybrid_core", "sharpe_core", "core_hybrid"):
        return "hybrid_sharpe_core"
    if value in (
        "hybrid_sharpe_core_no_stack",
        "hybrid-core-no-stack",
        "hybrid_core_no_stack",
        "sharpe_core_no_stack",
        "core_hybrid_no_stack",
        "no_stack",
    ):
        return "hybrid_sharpe_core_no_stack"
    if value in ("hybrid_sharpe_momentum", "hybrid-momentum", "hybrid_momentum", "sharpe_momentum", "momentum_hybrid"):
        return "hybrid_sharpe_momentum"
    if value in ("hybrid_sharpe_selective", "hybrid-selective", "hybrid_selective", "sharpe_selective", "selective_hybrid"):
        return "hybrid_sharpe_selective"
    if value in ("hybrid_sharpe_regime", "hybrid-regime", "hybrid_regime", "sharpe_regime", "regime_hybrid"):
        return "hybrid_sharpe_regime"
    if value in (
        "hybrid_sharpe_volume_flow",
        "hybrid-volume-flow",
        "hybrid_volume_flow",
        "sharpe_volume_flow",
        "volume_flow",
    ):
        return "hybrid_sharpe_volume_flow"
    if value in (
        "hybrid_sharpe_volume_regime",
        "hybrid-volume-regime",
        "hybrid_volume_regime",
        "sharpe_volume_regime",
        "volume_regime",
    ):
        return "hybrid_sharpe_volume_regime"
    if value in (
        "close_hold_reversion",
        "close-hold-reversion",
        "close_reversion",
        "eod_reversion",
        "overnight_reversion",
    ):
        return "close_hold_reversion"
    if value in (
        "close_hold_momentum",
        "close-hold-momentum",
        "close_momentum",
        "eod_momentum",
        "overnight_momentum",
    ):
        return "close_hold_momentum"
    if value in (
        "war_shock_reversion",
        "war-shock-reversion",
        "war_reversion",
        "shock_reversion",
        "volatile_reversion_swing",
    ):
        return "war_shock_reversion"
    if value in (
        "war_shock_momentum",
        "war-shock-momentum",
        "war_momentum",
        "shock_momentum",
        "volatile_momentum_swing",
    ):
        return "war_shock_momentum"
    if value in ("fvg2", "fvg-2", "fvg_2", "legacy2", "legacy-fvg2"):
        return "fvg2"
    if value in ("rsi_thresholds", "rsi-thresholds", "rsi_threshold", "rsi-threshold", "rsi"):
        return "rsi_thresholds"
    if value in ("stoch_rsi_thresholds", "stoch-rsi-thresholds", "stoch_thresholds", "stoch-thresholds", "stoch_rsi"):
        return "stoch_rsi_thresholds"
    if value in ("fvg3", "fvg-3", "fvg_3", "legacy3", "legacy-fvg3"):
        return "fvg3"
    if value in ("legacy", "old"):
        return "legacy"
    return "new"


def build_legacy_strategy_features() -> StrategyFeatureBuilder:
    builder = StrategyFeatureBuilder()
    builder.add("stoch_extreme", lambda r: 1.0 if (r["stoch_rsi"] > 80 or r["stoch_rsi"] < 20) else 0.0)
    builder.add("macd_hist", lambda r: r["macd_hist"])
    builder.add("macd_hist_delta", lambda r: r["macd_hist_delta"])
    builder.add("macd_green_increasing", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_red_recovering", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_green_fading", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] < 0) else 0.0)
    builder.add("macd_red_deepening", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] < 0) else 0.0)
    builder.add("first_green_fvg_dip", lambda r: r["first_green_fvg_dip"])
    builder.add("first_red_fvg_touch", lambda r: r["first_red_fvg_touch"])
    builder.add("fvg_green_size", lambda r: min(r["fvg_green_size"], 3.0))
    builder.add("fvg_red_size", lambda r: min(r["fvg_red_size"], 3.0))
    builder.add("fvg_bull_signal", lambda r: r["first_green_fvg_dip"] * min(r["fvg_green_size"], 3.0))
    builder.add("fvg_bear_signal", lambda r: r["first_red_fvg_touch"] * min(r["fvg_red_size"], 3.0))
    builder.add("fvg_conflict_penalty", lambda r: r["fvg_red_above_green"] * min(r["fvg_green_size"], 3.0))
    return builder


def build_fvg2_strategy_features() -> StrategyFeatureBuilder:
    builder = StrategyFeatureBuilder()
    builder.add("stoch_extreme_80", lambda r: 1.0 if r["stoch_rsi"] > 80 else 0.0)
    builder.add("stoch_extreme_20", lambda r: 1.0 if r["stoch_rsi"] < 20 else 0.0)
    builder.add("macd_hist", lambda r: r["macd_hist"])
    builder.add("macd_hist_delta", lambda r: r["macd_hist_delta"])
    builder.add("macd_green_increasing", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_red_recovering", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_green_fading", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] < 0) else 0.0)
    builder.add("macd_red_deepening", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] < 0) else 0.0)
    builder.add("first_green_fvg_dip", lambda r: r["first_green_fvg_dip"])
    builder.add("first_red_fvg_touch", lambda r: r["first_red_fvg_touch"])
    builder.add("fvg_green_size", lambda r: min(r["fvg_green_size"], 3.0))
    builder.add("fvg_red_size", lambda r: min(r["fvg_red_size"], 3.0))
    builder.add("fvg_bull_signal", lambda r: r["first_green_fvg_dip"] * min(r["fvg_green_size"], 3.0))
    builder.add("fvg_bear_signal", lambda r: r["first_red_fvg_touch"] * min(r["fvg_red_size"], 3.0))
    builder.add("fvg_conflict_penalty", lambda r: r["fvg_red_above_green"] * min(r["fvg_green_size"], 3.0))
    return builder


def build_fvg3_strategy_features() -> StrategyFeatureBuilder:
    builder = StrategyFeatureBuilder()
    builder.add("stoch_extreme", lambda r: 1.0 if r["stoch_rsi"] > 80 else 0.0)
    builder.add("stoch_extreme_neg", lambda r: 1.0 if r["stoch_rsi"] < 20 else 0.0)
    builder.add("macd_hist", lambda r: r["macd_hist"])
    builder.add("macd_hist_delta_absolute", lambda r: abs(r["macd_hist_delta"]))
    builder.add("macd_green_increasing", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_red_recovering", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_green_fading", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] < 0) else 0.0)
    builder.add("macd_red_deepening", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] < 0) else 0.0)
    builder.add("first_green_fvg_dip", lambda r: r["first_green_fvg_dip"])
    builder.add("first_red_fvg_touch", lambda r: r["first_red_fvg_touch"])
    builder.add("fvg_green_size", lambda r: min(r["fvg_green_size"], 3.0))
    builder.add("fvg_red_size", lambda r: min(r["fvg_red_size"], 3.0))
    builder.add("fvg_bull_signal", lambda r: r["first_green_fvg_dip"] * min(r["fvg_green_size"], 3.0))
    builder.add("fvg_bear_signal", lambda r: r["first_red_fvg_touch"] * min(r["fvg_red_size"], 3.0))
    builder.add("fvg_conflict_penalty", lambda r: r["fvg_red_above_green"] * min(r["fvg_green_size"], 3.0))
    return builder


def build_default_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("stoch_rsi", lambda r: g(r, "stoch_rsi") / 100.0)
    builder.add("stoch_velocity", lambda r: g(r, "stoch_velocity"))
    builder.add("stoch_low_zone", lambda r: g(r, "stoch_low_zone"))
    builder.add("stoch_high_zone", lambda r: g(r, "stoch_high_zone"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta"))
    builder.add("ret_1", lambda r: g(r, "ret_1"))
    builder.add("ret_3", lambda r: g(r, "ret_3"))
    builder.add("ret_5", lambda r: g(r, "ret_5"))
    builder.add("trend_20", lambda r: g(r, "trend_20"))
    builder.add("vol_20", lambda r: g(r, "vol_20"))
    builder.add("dist_to_bull_fvg", lambda r: g(r, "dist_to_bull_fvg"))
    builder.add("dist_to_bear_fvg", lambda r: g(r, "dist_to_bear_fvg"))
    builder.add("inside_bull_fvg", lambda r: g(r, "inside_bull_fvg"))
    builder.add("inside_bear_fvg", lambda r: g(r, "inside_bear_fvg"))
    builder.add("oversold_reversal", lambda r: g(r, "oversold_reversal"))
    builder.add("overbought_reversal", lambda r: g(r, "overbought_reversal"))
    builder.add("bull_confluence", lambda r: g(r, "bull_confluence"))
    builder.add("bear_confluence", lambda r: g(r, "bear_confluence"))
    return builder


def build_feature2_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("stoch_rsi", lambda r: g(r, "stoch_rsi") / 100.0)
    builder.add("stoch_velocity", lambda r: g(r, "stoch_velocity"))
    builder.add("stoch_low_zone", lambda r: g(r, "stoch_low_zone"))
    builder.add("stoch_high_zone", lambda r: g(r, "stoch_high_zone"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_delta", lambda r: g(r, "macd_delta", g(r, "macd_hist_delta")))
    builder.add("ret_1", lambda r: g(r, "ret_1"))
    builder.add("ret_3", lambda r: g(r, "ret_3"))
    builder.add("ret_5", lambda r: g(r, "ret_5"))
    builder.add("trend_20", lambda r: g(r, "trend_20"))
    builder.add("vol_20", lambda r: g(r, "vol_20"))
    builder.add("dist_to_bull_fvg", lambda r: g(r, "dist_to_bull_fvg"))
    builder.add("dist_to_bear_fvg", lambda r: g(r, "dist_to_bear_fvg"))
    builder.add("inside_bull_fvg", lambda r: g(r, "inside_bull_fvg"))
    builder.add("inside_bear_fvg", lambda r: g(r, "inside_bear_fvg"))
    return builder


def build_derivative_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("macd_green_increasing", lambda r: g(r, "macd_green_increasing"))
    builder.add("macd_red_recovering", lambda r: g(r, "macd_red_recovering"))
    builder.add("macd_green_fading", lambda r: g(r, "macd_green_fading"))
    builder.add("macd_red_deepening", lambda r: g(r, "macd_red_deepening"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema26", lambda r: g(r, "ema26"))
    builder.add("ema9_derivative_1", lambda r: g(r, "ema9_derivative_1"))
    builder.add("ema9_derivative_2", lambda r: g(r, "ema9_derivative_2"))
    builder.add("ema9_derivative_3", lambda r: g(r, "ema9_derivative_3"))
    builder.add("ema26_derivative_1", lambda r: g(r, "ema26_derivative_1"))
    builder.add("ema26_derivative_2", lambda r: g(r, "ema26_derivative_2"))
    builder.add("ema26_derivative_3", lambda r: g(r, "ema26_derivative_3"))
    return builder


def build_derivative2_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = build_derivative_strategy_features()
    builder.add("ema_derivative_1_diff", lambda r: g(r, "ema_derivative_1_diff"))
    builder.add("ema_derivative_2_diff", lambda r: g(r, "ema_derivative_2_diff"))
    builder.add("ema_derivative_3_diff", lambda r: g(r, "ema_derivative_3_diff"))
    builder.add("ema_derivative_1_cross", lambda r: g(r, "ema_derivative_1_cross"))
    builder.add("ema_derivative_1_cross_positive", lambda r: g(r, "ema_derivative_1_cross_positive"))
    builder.add("ema_derivative_1_cross_negative", lambda r: g(r, "ema_derivative_1_cross_negative"))
    builder.add("ema_derivative_2_cross", lambda r: g(r, "ema_derivative_2_cross"))
    builder.add("ema_derivative_2_cross_positive", lambda r: g(r, "ema_derivative_2_cross_positive"))
    builder.add("ema_derivative_2_cross_negative", lambda r: g(r, "ema_derivative_2_cross_negative"))
    builder.add("ema_derivative_3_cross", lambda r: g(r, "ema_derivative_3_cross"))
    builder.add("ema_derivative_3_cross_positive", lambda r: g(r, "ema_derivative_3_cross_positive"))
    builder.add("ema_derivative_3_cross_negative", lambda r: g(r, "ema_derivative_3_cross_negative"))
    return builder


def build_ema_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("ema3", lambda r: g(r, "ema3"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema21", lambda r: g(r, "ema21"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema_stack_bullish", lambda r: 1.0 if (g(r, "ema3") > g(r, "ema9") > g(r, "ema21")) else 0.0)
    builder.add("ema_stack_bearish", lambda r: 1.0 if (g(r, "ema3") < g(r, "ema9") < g(r, "ema21")) else 0.0)
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("ema21_slope", lambda r: g(r, "ema21_derivative_1"))
    return builder


def build_bollinger_bands_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("ema3", lambda r: g(r, "ema3"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema21", lambda r: g(r, "ema21"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("bb_upper", lambda r: g(r, "bb_upper"))
    builder.add("bb_middle", lambda r: g(r, "bb_middle"))
    builder.add("bb_lower", lambda r: g(r, "bb_lower"))
    builder.add("bb_width", lambda r: g(r, "bb_upper") - g(r, "bb_lower"))
    builder.add("bb_percent_b", lambda r: g(r, "bb_percent_b"))
    builder.add("price_to_bb_mid", lambda r: g(r, "close") - g(r, "bb_middle"))
    return builder


def build_vwap_anchor_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("ema3", lambda r: g(r, "ema3"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema21", lambda r: g(r, "ema21"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("vwap_anchor_high", lambda r: g(r, "vwap_anchor_high"))
    builder.add("vwap_anchor_low", lambda r: g(r, "vwap_anchor_low"))
    builder.add("vwap_anchor_spread", lambda r: g(r, "vwap_anchor_high") - g(r, "vwap_anchor_low"))
    builder.add("price_vs_vwap_high", lambda r: g(r, "close") - g(r, "vwap_anchor_high"))
    builder.add("price_vs_vwap_low", lambda r: g(r, "close") - g(r, "vwap_anchor_low"))
    builder.add("vwap_anchor_mid_bias", lambda r: g(r, "close") - ((g(r, "vwap_anchor_high") + g(r, "vwap_anchor_low")) / 2.0))
    return builder


def build_vwap_intraday_reversion_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("ema3", lambda r: g(r, "ema3"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema21", lambda r: g(r, "ema21"))
    builder.add("vwap_anchor_high", lambda r: g(r, "vwap_anchor_high"))
    builder.add("vwap_anchor_low", lambda r: g(r, "vwap_anchor_low"))
    builder.add("vwap_anchor_spread", lambda r: g(r, "vwap_anchor_high") - g(r, "vwap_anchor_low"))
    builder.add("price_vs_vwap_mid", lambda r: g(r, "close") - ((g(r, "vwap_anchor_high") + g(r, "vwap_anchor_low")) / 2.0))
    builder.add("distance_to_vwap_low", lambda r: g(r, "close") - g(r, "vwap_anchor_low"))
    builder.add("distance_to_vwap_high", lambda r: g(r, "vwap_anchor_high") - g(r, "close"))
    builder.add("zscore_vwap_mid", lambda r: (g(r, "close") - ((g(r, "vwap_anchor_high") + g(r, "vwap_anchor_low")) / 2.0)) / max(1e-9, g(r, "atr_frac", 1.0)))
    builder.add("stoch_rsi_norm", lambda r: g(r, "stoch_rsi") / 100.0)
    builder.add("stoch_velocity", lambda r: g(r, "stoch_velocity"))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    builder.add("mean_revert_long_bias", lambda r: max(0.0, g(r, "vwap_anchor_low") - g(r, "close")))
    builder.add("mean_revert_short_bias", lambda r: max(0.0, g(r, "close") - g(r, "vwap_anchor_high")))
    return builder


def build_vwap_intraday_momentum_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("ema3", lambda r: g(r, "ema3"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema21", lambda r: g(r, "ema21"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("vwap_anchor_high", lambda r: g(r, "vwap_anchor_high"))
    builder.add("vwap_anchor_low", lambda r: g(r, "vwap_anchor_low"))
    builder.add("price_vs_vwap_mid", lambda r: g(r, "close") - ((g(r, "vwap_anchor_high") + g(r, "vwap_anchor_low")) / 2.0))
    builder.add("vwap_breakout_strength", lambda r: (g(r, "close") - g(r, "vwap_anchor_high")) / max(1e-9, g(r, "atr_frac", 1.0)))
    builder.add("vwap_breakdown_strength", lambda r: (g(r, "vwap_anchor_low") - g(r, "close")) / max(1e-9, g(r, "atr_frac", 1.0)))
    builder.add("session_trend_pressure", lambda r: g(r, "ret_1") + g(r, "ret_3"))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    return builder


def build_vwap_intraday_5m_session_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("ema3", lambda r: g(r, "ema3"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema21", lambda r: g(r, "ema21"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("ema21_slope", lambda r: g(r, "ema21_derivative_1"))
    builder.add("session_vwap_5m", lambda r: g(r, "session_vwap_5m"))
    builder.add("session_vwap_delta_5m", lambda r: g(r, "session_vwap_delta_5m"))
    builder.add("session_vwap_delta_to_mean_5m", lambda r: g(r, "session_vwap_delta_to_mean_5m"))
    builder.add("price_vs_session_vwap_5m", lambda r: g(r, "close") - g(r, "session_vwap_5m"))
    builder.add(
        "price_vs_session_vwap_pct_5m",
        lambda r: ((g(r, "close") - g(r, "session_vwap_5m")) / g(r, "session_vwap_5m")) if abs(g(r, "session_vwap_5m")) > 1e-12 else 0.0,
    )
    builder.add("abs_price_vs_session_vwap_5m", lambda r: abs(g(r, "close") - g(r, "session_vwap_5m")))
    builder.add("session_vwap_reversion_signal_5m", lambda r: g(r, "session_vwap_5m") - g(r, "close"))
    builder.add("session_vwap_std_1_5m", lambda r: g(r, "session_vwap_std_1_5m"))
    builder.add("session_vwap_std_2_5m", lambda r: g(r, "session_vwap_std_2_5m"))
    builder.add("session_vwap_std_1_upper_5m", lambda r: g(r, "session_vwap_std_1_upper_5m"))
    builder.add("session_vwap_std_1_lower_5m", lambda r: g(r, "session_vwap_std_1_lower_5m"))
    builder.add("session_vwap_std_2_upper_5m", lambda r: g(r, "session_vwap_std_2_upper_5m"))
    builder.add("session_vwap_std_2_lower_5m", lambda r: g(r, "session_vwap_std_2_lower_5m"))
    builder.add("price_to_session_vwap_std_1_upper_5m", lambda r: g(r, "price_to_session_vwap_std_1_upper_5m"))
    builder.add("price_to_session_vwap_std_1_lower_5m", lambda r: g(r, "price_to_session_vwap_std_1_lower_5m"))
    builder.add("session_vwap_std_1_range_5m", lambda r: g(r, "session_vwap_std_1_range_5m"))
    builder.add("price_to_session_vwap_std_2_upper_5m", lambda r: g(r, "price_to_session_vwap_std_2_upper_5m"))
    builder.add("price_to_session_vwap_std_2_lower_5m", lambda r: g(r, "price_to_session_vwap_std_2_lower_5m"))
    builder.add("session_vwap_std_2_range_5m", lambda r: g(r, "session_vwap_std_2_range_5m"))
    return builder


def build_vwap_breakout_reversion_regime_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def vwap_mid(row: Row) -> float:
        return (g(row, "vwap_anchor_high") + g(row, "vwap_anchor_low")) / 2.0

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    def breakout_strength(row: Row) -> float:
        return max(0.0, (g(row, "close") - g(row, "vwap_anchor_high")) / atr_guard(row))

    def breakdown_strength(row: Row) -> float:
        return max(0.0, (g(row, "vwap_anchor_low") - g(row, "close")) / atr_guard(row))

    def mean_revert_long_bias(row: Row) -> float:
        return max(0.0, g(row, "vwap_anchor_low") - g(row, "close"))

    def mean_revert_short_bias(row: Row) -> float:
        return max(0.0, g(row, "close") - g(row, "vwap_anchor_high"))

    def breakout_pressure(row: Row) -> float:
        return max(breakout_strength(row), breakdown_strength(row))

    def mean_reversion_pressure(row: Row) -> float:
        return max(mean_revert_long_bias(row), mean_revert_short_bias(row)) / atr_guard(row)

    builder = StrategyFeatureBuilder()
    builder.add("ema3", lambda r: g(r, "ema3"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema21", lambda r: g(r, "ema21"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("vwap_anchor_high", lambda r: g(r, "vwap_anchor_high"))
    builder.add("vwap_anchor_low", lambda r: g(r, "vwap_anchor_low"))
    builder.add("vwap_anchor_spread", lambda r: g(r, "vwap_anchor_high") - g(r, "vwap_anchor_low"))
    builder.add("price_vs_vwap_mid", lambda r: g(r, "close") - vwap_mid(r))
    builder.add("zscore_vwap_mid", lambda r: (g(r, "close") - vwap_mid(r)) / atr_guard(r))
    builder.add("mean_revert_long_bias", lambda r: mean_revert_long_bias(r))
    builder.add("mean_revert_short_bias", lambda r: mean_revert_short_bias(r))
    builder.add("vwap_breakout_strength", lambda r: breakout_strength(r))
    builder.add("vwap_breakdown_strength", lambda r: breakdown_strength(r))
    builder.add("breakout_pressure", lambda r: breakout_pressure(r))
    builder.add("mean_reversion_pressure", lambda r: mean_reversion_pressure(r))
    builder.add("vwap_regime_signal", lambda r: breakout_pressure(r) - mean_reversion_pressure(r))
    builder.add("is_breakout_regime", lambda r: 1.0 if breakout_pressure(r) > mean_reversion_pressure(r) else 0.0)
    builder.add("is_reversion_regime", lambda r: 1.0 if breakout_pressure(r) <= mean_reversion_pressure(r) else 0.0)
    builder.add("session_trend_pressure", lambda r: g(r, "ret_1") + g(r, "ret_3"))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    return builder


def build_open15_orb_intraday_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("post_opening_range_window_15m", lambda r: g(r, "post_opening_range_window_15m"))
    builder.add("opening_range_breakout_up_15m", lambda r: g(r, "opening_range_breakout_up_15m"))
    builder.add("opening_range_breakdown_15m", lambda r: g(r, "opening_range_breakdown_15m"))
    builder.add("price_vs_opening_range_high_15m", lambda r: g(r, "price_vs_opening_range_high_15m"))
    builder.add("price_vs_opening_range_low_15m", lambda r: g(r, "price_vs_opening_range_low_15m"))
    builder.add("opening_range_width_pct_15m", lambda r: g(r, "opening_range_width_pct_15m"))
    builder.add("session_vwap_delta_5m", lambda r: g(r, "session_vwap_delta_5m"))
    builder.add("price_vs_session_vwap_5m", lambda r: g(r, "price_vs_session_vwap_5m"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta"))
    builder.add("intraday_trade_window_open", lambda r: g(r, "intraday_trade_window_open"))
    builder.add("near_session_close_5m", lambda r: g(r, "near_session_close_5m"))
    builder.add("bars_remaining_in_session_5m", lambda r: g(r, "bars_remaining_in_session_5m"))
    return builder


def build_open15_vwap_reclaim_intraday_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("post_opening_range_window_15m", lambda r: g(r, "post_opening_range_window_15m"))
    builder.add("opening_range_mid_15m", lambda r: g(r, "opening_range_mid_15m"))
    builder.add("price_vs_opening_range_mid_15m", lambda r: g(r, "price_vs_opening_range_mid_15m"))
    builder.add("opening_range_position_pct_15m", lambda r: g(r, "opening_range_position_pct_15m"))
    builder.add("price_vs_session_vwap_5m", lambda r: g(r, "price_vs_session_vwap_5m"))
    builder.add("session_vwap_delta_to_mean_5m", lambda r: g(r, "session_vwap_delta_to_mean_5m"))
    builder.add("session_vwap_reversion_signal_5m", lambda r: g(r, "session_vwap_reversion_signal_5m"))
    builder.add("vwap_reclaim_long_signal_5m", lambda r: g(r, "vwap_reclaim_long_signal_5m"))
    builder.add("vwap_reclaim_short_signal_5m", lambda r: g(r, "vwap_reclaim_short_signal_5m"))
    builder.add("stoch_rsi_norm", lambda r: g(r, "stoch_rsi") / 100.0)
    builder.add("stoch_velocity", lambda r: g(r, "stoch_velocity"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta"))
    builder.add("intraday_trade_window_open", lambda r: g(r, "intraday_trade_window_open"))
    builder.add("near_session_close_5m", lambda r: g(r, "near_session_close_5m"))
    return builder


def build_open15_trend_momentum_daytrade_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    # First-15-minute observation and breakout context.
    builder.add("post_opening_range_window_15m", lambda r: g(r, "post_opening_range_window_15m"))
    builder.add("opening_range_breakout_up_15m", lambda r: g(r, "opening_range_breakout_up_15m"))
    builder.add("opening_range_breakdown_15m", lambda r: g(r, "opening_range_breakdown_15m"))
    builder.add("opening_range_width_pct_15m", lambda r: g(r, "opening_range_width_pct_15m"))
    builder.add("price_vs_opening_range_high_15m", lambda r: g(r, "price_vs_opening_range_high_15m"))
    builder.add("price_vs_opening_range_low_15m", lambda r: g(r, "price_vs_opening_range_low_15m"))
    # 5m trend/momentum stack.
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("ema_slope_alignment", lambda r: g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("session_trend_pressure", lambda r: g(r, "ret_1") + g(r, "ret_3"))
    # Intraday-only and risk controls.
    builder.add("intraday_trade_window_open", lambda r: g(r, "intraday_trade_window_open"))
    builder.add("near_session_close_5m", lambda r: g(r, "near_session_close_5m"))
    builder.add("bars_remaining_in_session_5m", lambda r: g(r, "bars_remaining_in_session_5m"))
    builder.add("trade_count_today", lambda r: g(r, "trade_count_today"))
    builder.add("trades_remaining_cap_2", lambda r: max(0.0, 2.0 - g(r, "trade_count_today")))
    builder.add("avoid_overnight_bias", lambda r: 1.0 - min(1.0, g(r, "near_session_close_5m")))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    builder.add("open15_breakout_strength_atr", lambda r: max(0.0, g(r, "price_vs_opening_range_high_15m") / atr_guard(r)))
    builder.add("open15_breakdown_strength_atr", lambda r: max(0.0, -g(r, "price_vs_opening_range_low_15m") / atr_guard(r)))
    return builder


def build_open15_dual_breakout_daytrade_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    # Wait for opening range completion, then look for continuation.
    builder.add("post_opening_range_window_15m", lambda r: g(r, "post_opening_range_window_15m"))
    builder.add("opening_range_breakout_up_15m", lambda r: g(r, "opening_range_breakout_up_15m"))
    builder.add("opening_range_breakdown_15m", lambda r: g(r, "opening_range_breakdown_15m"))
    builder.add("opening_range_width_pct_15m", lambda r: g(r, "opening_range_width_pct_15m"))
    builder.add("opening_range_position_pct_15m", lambda r: g(r, "opening_range_position_pct_15m"))
    builder.add("price_vs_opening_range_high_15m", lambda r: g(r, "price_vs_opening_range_high_15m"))
    builder.add("price_vs_opening_range_low_15m", lambda r: g(r, "price_vs_opening_range_low_15m"))
    # Momentum confirmation via VWAP + EMA + MACD.
    builder.add("price_vs_session_vwap_5m", lambda r: g(r, "price_vs_session_vwap_5m"))
    builder.add("session_vwap_delta_5m", lambda r: g(r, "session_vwap_delta_5m"))
    builder.add("vwap_breakout_strength", lambda r: (g(r, "close") - g(r, "vwap_anchor_high")) / atr_guard(r))
    builder.add("vwap_breakdown_strength", lambda r: (g(r, "vwap_anchor_low") - g(r, "close")) / atr_guard(r))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("momentum_alignment", lambda r: (g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1")) + g(r, "macd_hist"))
    # Day-trading constraints.
    builder.add("intraday_trade_window_open", lambda r: g(r, "intraday_trade_window_open"))
    builder.add("near_session_close_5m", lambda r: g(r, "near_session_close_5m"))
    builder.add("bars_remaining_in_session_5m", lambda r: g(r, "bars_remaining_in_session_5m"))
    builder.add("trade_count_today", lambda r: g(r, "trade_count_today"))
    builder.add("second_trade_only_if_trend_intact", lambda r: (1.0 if g(r, "trade_count_today") <= 1.0 else 0.0) * max(0.0, g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1")))
    builder.add("avoid_overnight_bias", lambda r: 1.0 - min(1.0, g(r, "near_session_close_5m")))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    return builder


def build_open15_dual_breakout_daytrade_plus_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    # Open15 follow-through foundation.
    builder.add("post_opening_range_window_15m", lambda r: g(r, "post_opening_range_window_15m"))
    builder.add("opening_range_breakout_up_15m", lambda r: g(r, "opening_range_breakout_up_15m"))
    builder.add("opening_range_breakdown_15m", lambda r: g(r, "opening_range_breakdown_15m"))
    builder.add("opening_range_width_pct_15m", lambda r: g(r, "opening_range_width_pct_15m"))
    builder.add("opening_range_position_pct_15m", lambda r: g(r, "opening_range_position_pct_15m"))
    builder.add("price_vs_opening_range_high_15m", lambda r: g(r, "price_vs_opening_range_high_15m"))
    builder.add("price_vs_opening_range_low_15m", lambda r: g(r, "price_vs_opening_range_low_15m"))
    # Trend + pullback quality controls.
    builder.add("price_vs_session_vwap_5m", lambda r: g(r, "price_vs_session_vwap_5m"))
    builder.add("session_vwap_delta_5m", lambda r: g(r, "session_vwap_delta_5m"))
    builder.add("session_vwap_delta_to_mean_5m", lambda r: g(r, "session_vwap_delta_to_mean_5m"))
    builder.add("vwap_reclaim_long_signal_5m", lambda r: g(r, "vwap_reclaim_long_signal_5m"))
    builder.add("vwap_breakout_strength_atr", lambda r: max(0.0, (g(r, "close") - g(r, "vwap_anchor_high")) / atr_guard(r)))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("trend_pullback_quality", lambda r: (g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1")) + g(r, "session_vwap_reversion_signal_5m"))
    # Looser intraday cap to allow >2 trades/day when trend remains healthy.
    builder.add("intraday_trade_window_open", lambda r: g(r, "intraday_trade_window_open"))
    builder.add("near_session_close_5m", lambda r: g(r, "near_session_close_5m"))
    builder.add("bars_remaining_in_session_5m", lambda r: g(r, "bars_remaining_in_session_5m"))
    builder.add("trade_count_today", lambda r: g(r, "trade_count_today"))
    builder.add("trades_remaining_cap_3", lambda r: max(0.0, 3.0 - g(r, "trade_count_today")))
    builder.add("third_trade_only_if_reclaim_valid", lambda r: (1.0 if g(r, "trade_count_today") <= 2.0 else 0.0) * max(0.0, g(r, "vwap_reclaim_long_signal_5m")))
    builder.add("avoid_overnight_bias", lambda r: 1.0 - min(1.0, g(r, "near_session_close_5m")))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    return builder


def build_open15_dual_breakout_daytrade_scalp_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    # Fast continuation/reclaim setup with explicit higher trade-frequency controls.
    builder.add("post_opening_range_window_15m", lambda r: g(r, "post_opening_range_window_15m"))
    builder.add("opening_range_breakout_up_15m", lambda r: g(r, "opening_range_breakout_up_15m"))
    builder.add("opening_range_breakdown_15m", lambda r: g(r, "opening_range_breakdown_15m"))
    builder.add("opening_range_position_pct_15m", lambda r: g(r, "opening_range_position_pct_15m"))
    builder.add("price_vs_opening_range_high_15m", lambda r: g(r, "price_vs_opening_range_high_15m"))
    builder.add("price_vs_session_vwap_5m", lambda r: g(r, "price_vs_session_vwap_5m"))
    builder.add("session_vwap_delta_5m", lambda r: g(r, "session_vwap_delta_5m"))
    builder.add("session_vwap_reversion_signal_5m", lambda r: g(r, "session_vwap_reversion_signal_5m"))
    builder.add("vwap_reclaim_long_signal_5m", lambda r: g(r, "vwap_reclaim_long_signal_5m"))
    builder.add("vwap_breakout_strength_atr", lambda r: max(0.0, (g(r, "close") - g(r, "vwap_anchor_high")) / atr_guard(r)))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("ret_1", lambda r: g(r, "ret_1"))
    builder.add("ret_3", lambda r: g(r, "ret_3"))
    builder.add("intraday_trade_window_open", lambda r: g(r, "intraday_trade_window_open"))
    builder.add("near_session_close_5m", lambda r: g(r, "near_session_close_5m"))
    builder.add("bars_remaining_in_session_5m", lambda r: g(r, "bars_remaining_in_session_5m"))
    builder.add("trade_count_today", lambda r: g(r, "trade_count_today"))
    builder.add("trades_remaining_cap_5", lambda r: max(0.0, 5.0 - g(r, "trade_count_today")))
    builder.add("fourth_plus_trade_requires_momentum", lambda r: (1.0 if g(r, "trade_count_today") <= 3.0 else 0.0) + max(0.0, g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1")))
    builder.add("avoid_overnight_bias", lambda r: 1.0 - min(1.0, g(r, "near_session_close_5m")))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    return builder


def build_vwap_momentum_trend_5m_conservative_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    builder.add("intraday_trade_window_open", lambda r: g(r, "intraday_trade_window_open"))
    builder.add("near_session_close_5m", lambda r: g(r, "near_session_close_5m"))
    builder.add("bars_remaining_in_session_5m", lambda r: g(r, "bars_remaining_in_session_5m"))
    builder.add("trade_count_today", lambda r: g(r, "trade_count_today"))
    builder.add("trades_remaining_cap_4", lambda r: max(0.0, 4.0 - g(r, "trade_count_today")))
    builder.add("avoid_overnight_bias", lambda r: 1.0 - min(1.0, g(r, "near_session_close_5m")))
    builder.add("price_vs_session_vwap_5m", lambda r: g(r, "price_vs_session_vwap_5m"))
    builder.add("session_vwap_delta_5m", lambda r: g(r, "session_vwap_delta_5m"))
    builder.add("vwap_breakout_strength_atr", lambda r: max(0.0, (g(r, "close") - g(r, "vwap_anchor_high")) / atr_guard(r)))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema_slope_alignment", lambda r: g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("trend_momentum_agreement", lambda r: (g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1")) + g(r, "macd_hist"))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    return builder


def build_vwap_momentum_trend_5m_pullback_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    builder.add("intraday_trade_window_open", lambda r: g(r, "intraday_trade_window_open"))
    builder.add("near_session_close_5m", lambda r: g(r, "near_session_close_5m"))
    builder.add("bars_remaining_in_session_5m", lambda r: g(r, "bars_remaining_in_session_5m"))
    builder.add("trade_count_today", lambda r: g(r, "trade_count_today"))
    builder.add("trades_remaining_cap_3", lambda r: max(0.0, 3.0 - g(r, "trade_count_today")))
    builder.add("avoid_overnight_bias", lambda r: 1.0 - min(1.0, g(r, "near_session_close_5m")))
    builder.add("price_vs_session_vwap_5m", lambda r: g(r, "price_vs_session_vwap_5m"))
    builder.add("session_vwap_delta_to_mean_5m", lambda r: g(r, "session_vwap_delta_to_mean_5m"))
    builder.add("session_vwap_reversion_signal_5m", lambda r: g(r, "session_vwap_reversion_signal_5m"))
    builder.add("vwap_reclaim_long_signal_5m", lambda r: g(r, "vwap_reclaim_long_signal_5m"))
    builder.add("vwap_breakout_strength_atr", lambda r: max(0.0, (g(r, "close") - g(r, "vwap_anchor_high")) / atr_guard(r)))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("stoch_velocity", lambda r: g(r, "stoch_velocity"))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    return builder


def build_hybrid_sharpe_core_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    # EMA structure + slope (from the EMA-focused run).
    builder.add("ema3", lambda r: g(r, "ema3"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema21", lambda r: g(r, "ema21"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema_stack_bullish", lambda r: 1.0 if (g(r, "ema3") > g(r, "ema9") > g(r, "ema21")) else 0.0)
    builder.add("ema_stack_bearish", lambda r: 1.0 if (g(r, "ema3") < g(r, "ema9") < g(r, "ema21")) else 0.0)
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("ema21_slope", lambda r: g(r, "ema21_derivative_1"))
    # Directional momentum state (from the derivative2 run).
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("macd_green_increasing", lambda r: g(r, "macd_green_increasing"))
    builder.add("macd_red_recovering", lambda r: g(r, "macd_red_recovering"))
    builder.add("macd_green_fading", lambda r: g(r, "macd_green_fading"))
    builder.add("macd_red_deepening", lambda r: g(r, "macd_red_deepening"))
    builder.add("ema_derivative_1_diff", lambda r: g(r, "ema_derivative_1_diff"))
    builder.add("ema_derivative_1_cross_positive", lambda r: g(r, "ema_derivative_1_cross_positive"))
    builder.add("ema_derivative_1_cross_negative", lambda r: g(r, "ema_derivative_1_cross_negative"))
    return builder


def build_hybrid_sharpe_core_no_stack_strategy_features() -> StrategyFeatureBuilder:
    builder = build_hybrid_sharpe_core_strategy_features()
    filtered = [feature for feature in builder._features if feature.name not in {"ema_stack_bullish", "ema_stack_bearish"}]
    builder._features = filtered
    return builder


def build_hybrid_sharpe_momentum_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = build_hybrid_sharpe_core_strategy_features()
    # Extra curvature and agreement features to target risk-adjusted quality (Sharpe).
    builder.add("ema9_derivative_2", lambda r: g(r, "ema9_derivative_2"))
    builder.add("ema21_derivative_2", lambda r: g(r, "ema21_derivative_2"))
    builder.add("ema9_derivative_3", lambda r: g(r, "ema9_derivative_3"))
    builder.add("ema21_derivative_3", lambda r: g(r, "ema21_derivative_3"))
    builder.add("ema_derivative_2_diff", lambda r: g(r, "ema_derivative_2_diff"))
    builder.add("ema_derivative_3_diff", lambda r: g(r, "ema_derivative_3_diff"))
    builder.add("ema_derivative_2_cross_positive", lambda r: g(r, "ema_derivative_2_cross_positive"))
    builder.add("ema_derivative_2_cross_negative", lambda r: g(r, "ema_derivative_2_cross_negative"))
    builder.add("ema_derivative_3_cross_positive", lambda r: g(r, "ema_derivative_3_cross_positive"))
    builder.add("ema_derivative_3_cross_negative", lambda r: g(r, "ema_derivative_3_cross_negative"))
    builder.add("ema_slope_alignment", lambda r: g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1"))
    builder.add("ema_spread_balance", lambda r: (g(r, "ema3") - g(r, "ema9")) - (g(r, "ema9") - g(r, "ema21")))
    return builder


def build_hybrid_sharpe_selective_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    # Compact subset leaning on the stronger/less noisy contributors from prior ablations.
    builder.add("ema3", lambda r: g(r, "ema3"))
    builder.add("ema9", lambda r: g(r, "ema9"))
    builder.add("ema21", lambda r: g(r, "ema21"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_green_increasing", lambda r: g(r, "macd_green_increasing"))
    builder.add("macd_red_recovering", lambda r: g(r, "macd_red_recovering"))
    builder.add("ema_derivative_1_diff", lambda r: g(r, "ema_derivative_1_diff"))
    builder.add("ema_derivative_1_cross_positive", lambda r: g(r, "ema_derivative_1_cross_positive"))
    builder.add("ema_derivative_1_cross_negative", lambda r: g(r, "ema_derivative_1_cross_negative"))
    return builder


def build_hybrid_sharpe_regime_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = build_hybrid_sharpe_core_no_stack_strategy_features()
    # Add context features to better handle sideways/high-volatility drag.
    builder.add("trend_20", lambda r: g(r, "trend_20"))
    builder.add("vol_20", lambda r: g(r, "vol_20"))
    builder.add("ret_1", lambda r: g(r, "ret_1"))
    builder.add("ret_3", lambda r: g(r, "ret_3"))
    builder.add("stoch_rsi_norm", lambda r: g(r, "stoch_rsi") / 100.0)
    builder.add("stoch_velocity", lambda r: g(r, "stoch_velocity"))
    builder.add("ema_slope_alignment", lambda r: g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1"))
    return builder


def build_hybrid_sharpe_volume_flow_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = build_hybrid_sharpe_core_no_stack_strategy_features()
    # Volume-pressure overlays for the no-stack Sharpe core.
    builder.add("volume", lambda r: g(r, "volume"))
    builder.add("volume_ma20", lambda r: g(r, "volume_ma20"))
    builder.add(
        "volume_spike_ratio",
        lambda r: g(r, "volume") / max(1e-9, g(r, "volume_ma20", g(r, "volume", 1.0))),
    )
    builder.add("signed_volume_pressure", lambda r: g(r, "ret_1") * g(r, "volume"))
    builder.add("volume_volatility_coupling", lambda r: g(r, "vol_20") * g(r, "volume"))
    builder.add("volume_trend_coupling", lambda r: g(r, "trend_20") * g(r, "volume"))
    return builder


def build_hybrid_sharpe_volume_regime_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = build_hybrid_sharpe_volume_flow_strategy_features()
    # Compact gating-style features for high/low participation regimes.
    builder.add("high_volume_regime", lambda r: 1.0 if g(r, "volume") > g(r, "volume_ma20", g(r, "volume")) else 0.0)
    builder.add("low_volume_regime", lambda r: 1.0 if g(r, "volume") <= g(r, "volume_ma20", g(r, "volume")) else 0.0)
    builder.add("trend_in_high_volume", lambda r: g(r, "trend_20") * (1.0 if g(r, "volume") > g(r, "volume_ma20", g(r, "volume")) else 0.0))
    builder.add("pullback_in_low_volume", lambda r: g(r, "ret_3") * (1.0 if g(r, "volume") <= g(r, "volume_ma20", g(r, "volume")) else 0.0))
    return builder



def build_close_hold_reversion_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    builder.add("ret_1", lambda r: g(r, "ret_1"))
    builder.add("ret_3", lambda r: g(r, "ret_3"))
    builder.add("stoch_rsi_norm", lambda r: g(r, "stoch_rsi") / 100.0)
    builder.add("stoch_low_zone", lambda r: g(r, "stoch_low_zone"))
    builder.add("stoch_high_zone", lambda r: g(r, "stoch_high_zone"))
    builder.add("stoch_velocity", lambda r: g(r, "stoch_velocity"))
    builder.add("rsi", lambda r: g(r, "rsi"))
    builder.add("trend_20", lambda r: g(r, "trend_20"))
    builder.add("vol_20", lambda r: g(r, "vol_20"))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    builder.add("bb_percent_b", lambda r: g(r, "bb_percent_b"))
    builder.add("bb_reversion_distance", lambda r: 0.5 - g(r, "bb_percent_b", 0.5))
    builder.add("vwap_mid_dislocation", lambda r: (g(r, "close") - ((g(r, "vwap_anchor_high") + g(r, "vwap_anchor_low")) / 2.0)) / atr_guard(r))
    builder.add("close_to_vwap_low_atr", lambda r: (g(r, "close") - g(r, "vwap_anchor_low")) / atr_guard(r))
    builder.add("close_to_vwap_high_atr", lambda r: (g(r, "vwap_anchor_high") - g(r, "close")) / atr_guard(r))
    builder.add("bearish_exhaustion", lambda r: max(0.0, -g(r, "ret_1")) * max(0.0, g(r, "stoch_low_zone")))
    builder.add("bullish_exhaustion", lambda r: max(0.0, g(r, "ret_1")) * max(0.0, g(r, "stoch_high_zone")))
    return builder


def build_close_hold_momentum_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    builder.add("ret_1", lambda r: g(r, "ret_1"))
    builder.add("ret_3", lambda r: g(r, "ret_3"))
    builder.add("ret_5", lambda r: g(r, "ret_5"))
    builder.add("trend_20", lambda r: g(r, "trend_20"))
    builder.add("vol_20", lambda r: g(r, "vol_20"))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("ema21_slope", lambda r: g(r, "ema21_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("macd_acceleration", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")) / atr_guard(r))
    builder.add("vwap_breakout_strength", lambda r: (g(r, "close") - g(r, "vwap_anchor_high")) / atr_guard(r))
    builder.add("vwap_breakdown_strength", lambda r: (g(r, "vwap_anchor_low") - g(r, "close")) / atr_guard(r))
    builder.add("momentum_alignment", lambda r: (g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1")) + g(r, "macd_hist"))
    return builder


def build_war_shock_reversion_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    builder.add("ret_1", lambda r: g(r, "ret_1"))
    builder.add("ret_3", lambda r: g(r, "ret_3"))
    builder.add("vol_20", lambda r: g(r, "vol_20"))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    builder.add("shock_intensity", lambda r: g(r, "vol_20") + g(r, "atr_frac"))
    builder.add("stoch_rsi_norm", lambda r: g(r, "stoch_rsi") / 100.0)
    builder.add("stoch_low_zone", lambda r: g(r, "stoch_low_zone"))
    builder.add("stoch_velocity", lambda r: g(r, "stoch_velocity"))
    builder.add("panic_down_move", lambda r: max(0.0, -g(r, "ret_1")) * (1.0 + g(r, "vol_20")))
    builder.add("oversold_snapback_bias", lambda r: max(0.0, g(r, "stoch_low_zone")) * max(0.0, -g(r, "ret_1")))
    builder.add("distance_to_vwap_low_atr", lambda r: (g(r, "close") - g(r, "vwap_anchor_low")) / atr_guard(r))
    builder.add("distance_to_vwap_mid_atr", lambda r: (g(r, "close") - ((g(r, "vwap_anchor_high") + g(r, "vwap_anchor_low")) / 2.0)) / atr_guard(r))
    builder.add("bb_reversion_distance", lambda r: 0.5 - g(r, "bb_percent_b", 0.5))
    builder.add("mean_revert_long_bias", lambda r: g(r, "mean_revert_long_bias"))
    builder.add("session_vwap_reversion_signal_5m", lambda r: g(r, "session_vwap_reversion_signal_5m"))
    return builder


def build_war_shock_momentum_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    def atr_guard(row: Row) -> float:
        return max(1e-9, g(row, "atr_frac", 1.0))

    builder = StrategyFeatureBuilder()
    builder.add("ret_1", lambda r: g(r, "ret_1"))
    builder.add("ret_3", lambda r: g(r, "ret_3"))
    builder.add("ret_5", lambda r: g(r, "ret_5"))
    builder.add("vol_20", lambda r: g(r, "vol_20"))
    builder.add("atr_frac", lambda r: g(r, "atr_frac"))
    builder.add("shock_intensity", lambda r: g(r, "vol_20") + g(r, "atr_frac"))
    builder.add("ema3_9_spread", lambda r: g(r, "ema3") - g(r, "ema9"))
    builder.add("ema9_21_spread", lambda r: g(r, "ema9") - g(r, "ema21"))
    builder.add("ema3_slope", lambda r: g(r, "ema3_derivative_1"))
    builder.add("ema9_slope", lambda r: g(r, "ema9_derivative_1"))
    builder.add("macd_hist", lambda r: g(r, "macd_hist"))
    builder.add("macd_hist_delta", lambda r: g(r, "macd_hist_delta", g(r, "macd_delta")))
    builder.add("breakout_strength_atr", lambda r: (g(r, "close") - g(r, "vwap_anchor_high")) / atr_guard(r))
    builder.add("breakdown_strength_atr", lambda r: (g(r, "vwap_anchor_low") - g(r, "close")) / atr_guard(r))
    builder.add("trend_follow_thrust", lambda r: (g(r, "ema3_derivative_1") - g(r, "ema21_derivative_1")) + g(r, "macd_hist"))
    return builder



def build_rsi_threshold_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("rsi_over_70", lambda r: 1.0 if g(r, "rsi") > 70.0 else 0.0)
    builder.add("rsi_below_30", lambda r: 1.0 if g(r, "rsi") < 30.0 else 0.0)
    return builder


def build_stoch_rsi_threshold_strategy_features() -> StrategyFeatureBuilder:
    def g(row: Row, key: str, default: float = 0.0) -> float:
        return float(row.get(key, default))

    builder = StrategyFeatureBuilder()
    builder.add("stoch_rsi_over_80", lambda r: 1.0 if g(r, "stoch_rsi") > 80.0 else 0.0)
    builder.add("stoch_rsi_under_20", lambda r: 1.0 if g(r, "stoch_rsi") < 20.0 else 0.0)
    return builder

def get_strategy_feature_builder(feature_set: FeatureSet | str = "feature2") -> StrategyFeatureBuilder:
    normalized = normalize_feature_set(feature_set)
    if normalized == "legacy":
        return build_legacy_strategy_features()
    if normalized == "fvg2":
        return build_fvg2_strategy_features()
    if normalized == "fvg3":
        return build_fvg3_strategy_features()
    if normalized == "rsi_thresholds":
        return build_rsi_threshold_strategy_features()
    if normalized == "stoch_rsi_thresholds":
        return build_stoch_rsi_threshold_strategy_features()
    if normalized == "derivative2":
        return build_derivative2_strategy_features()
    if normalized == "ema":
        return build_ema_strategy_features()
    if normalized == "bollinger_bands":
        return build_bollinger_bands_strategy_features()
    if normalized == "vwap_anchor":
        return build_vwap_anchor_strategy_features()
    if normalized == "vwap_intraday_reversion":
        return build_vwap_intraday_reversion_strategy_features()
    if normalized == "vwap_intraday_momentum":
        return build_vwap_intraday_momentum_strategy_features()
    if normalized == "vwap_intraday_5m_session":
        return build_vwap_intraday_5m_session_strategy_features()
    if normalized == "vwap_breakout_reversion_regime":
        return build_vwap_breakout_reversion_regime_strategy_features()
    if normalized == "open15_orb_intraday":
        return build_open15_orb_intraday_strategy_features()
    if normalized == "open15_vwap_reclaim_intraday":
        return build_open15_vwap_reclaim_intraday_strategy_features()
    if normalized == "open15_trend_momentum_daytrade":
        return build_open15_trend_momentum_daytrade_strategy_features()
    if normalized == "open15_dual_breakout_daytrade":
        return build_open15_dual_breakout_daytrade_strategy_features()
    if normalized == "open15_dual_breakout_daytrade_plus":
        return build_open15_dual_breakout_daytrade_plus_strategy_features()
    if normalized == "open15_dual_breakout_daytrade_scalp":
        return build_open15_dual_breakout_daytrade_scalp_strategy_features()
    if normalized == "vwap_momentum_trend_5m_conservative":
        return build_vwap_momentum_trend_5m_conservative_strategy_features()
    if normalized == "vwap_momentum_trend_5m_pullback":
        return build_vwap_momentum_trend_5m_pullback_strategy_features()
    if normalized == "hybrid_sharpe_core":
        return build_hybrid_sharpe_core_strategy_features()
    if normalized == "hybrid_sharpe_core_no_stack":
        return build_hybrid_sharpe_core_no_stack_strategy_features()
    if normalized == "hybrid_sharpe_momentum":
        return build_hybrid_sharpe_momentum_strategy_features()
    if normalized == "hybrid_sharpe_selective":
        return build_hybrid_sharpe_selective_strategy_features()
    if normalized == "hybrid_sharpe_regime":
        return build_hybrid_sharpe_regime_strategy_features()
    if normalized == "hybrid_sharpe_volume_flow":
        return build_hybrid_sharpe_volume_flow_strategy_features()
    if normalized == "hybrid_sharpe_volume_regime":
        return build_hybrid_sharpe_volume_regime_strategy_features()
    if normalized == "close_hold_reversion":
        return build_close_hold_reversion_strategy_features()
    if normalized == "close_hold_momentum":
        return build_close_hold_momentum_strategy_features()
    if normalized == "war_shock_reversion":
        return build_war_shock_reversion_strategy_features()
    if normalized == "war_shock_momentum":
        return build_war_shock_momentum_strategy_features()
    if normalized == "derivative":
        return build_derivative_strategy_features()
    if normalized == "feature2":
        return build_feature2_strategy_features()
    if normalized == "dqn":
        return build_feature2_strategy_features()
    return build_default_strategy_features()


def infer_bundle_feature_set(bundle: Dict[str, object]) -> FeatureSet:
    explicit = bundle.get("feature_set")
    if isinstance(explicit, str):
        return normalize_feature_set(explicit)
    names = bundle.get("feature_names", [])
    if isinstance(names, list) and "stoch_extreme" in names:
        return "legacy"
    if isinstance(names, list) and "stoch_extreme_80" in names and "stoch_extreme_20" in names:
        return "fvg2"
    if isinstance(names, list) and "rsi_over_70" in names and "rsi_below_30" in names:
        return "rsi_thresholds"
    if isinstance(names, list) and "stoch_rsi_over_80" in names and "stoch_rsi_under_20" in names:
        return "stoch_rsi_thresholds"
    if isinstance(names, list) and "stoch_extreme_neg" in names and "macd_hist_delta_absolute" in names:
        return "fvg3"
    if isinstance(names, list) and "ema_derivative_3_diff" in names:
        return "derivative2"
    if isinstance(names, list) and "bb_upper" in names and "bb_lower" in names:
        return "bollinger_bands"
    if isinstance(names, list) and "vwap_anchor_high" in names and "vwap_anchor_low" in names:
        return "vwap_anchor"
    if isinstance(names, list) and "vwap_regime_signal" in names and "breakout_pressure" in names:
        return "vwap_breakout_reversion_regime"
    if isinstance(names, list) and "opening_range_breakout_up_15m" in names and "post_opening_range_window_15m" in names:
        return "open15_orb_intraday"
    if isinstance(names, list) and "vwap_reclaim_long_signal_5m" in names and "opening_range_position_pct_15m" in names:
        return "open15_vwap_reclaim_intraday"
    if isinstance(names, list) and "trades_remaining_cap_2" in names and "open15_breakout_strength_atr" in names:
        return "open15_trend_momentum_daytrade"
    if isinstance(names, list) and "second_trade_only_if_trend_intact" in names and "momentum_alignment" in names:
        return "open15_dual_breakout_daytrade"
    if isinstance(names, list) and "third_trade_only_if_reclaim_valid" in names and "trades_remaining_cap_3" in names:
        return "open15_dual_breakout_daytrade_plus"
    if isinstance(names, list) and "trades_remaining_cap_5" in names and "fourth_plus_trade_requires_momentum" in names:
        return "open15_dual_breakout_daytrade_scalp"
    if isinstance(names, list) and "trades_remaining_cap_4" in names and "trend_momentum_agreement" in names:
        return "vwap_momentum_trend_5m_conservative"
    if isinstance(names, list) and "trades_remaining_cap_3" in names and "vwap_reclaim_long_signal_5m" in names:
        return "vwap_momentum_trend_5m_pullback"
    if isinstance(names, list) and "mean_revert_long_bias" in names and "mean_revert_short_bias" in names:
        return "vwap_intraday_reversion"
    if isinstance(names, list) and "vwap_breakout_strength" in names and "vwap_breakdown_strength" in names:
        return "vwap_intraday_momentum"
    if isinstance(names, list) and "session_vwap_5m" in names and "price_vs_session_vwap_5m" in names:
        return "vwap_intraday_5m_session"
    if isinstance(names, list) and "ema_slope_alignment" in names and "ema_spread_balance" in names:
        return "hybrid_sharpe_momentum"
    if isinstance(names, list) and "trend_20" in names and "vol_20" in names and "stoch_rsi_norm" in names:
        return "hybrid_sharpe_regime"
    if isinstance(names, list) and "volume_spike_ratio" in names and "high_volume_regime" in names:
        return "hybrid_sharpe_volume_regime"
    if isinstance(names, list) and "bb_reversion_distance" in names and "bearish_exhaustion" in names:
        return "close_hold_reversion"
    if isinstance(names, list) and "momentum_alignment" in names and "macd_acceleration" in names:
        return "close_hold_momentum"
    if isinstance(names, list) and "oversold_snapback_bias" in names and "panic_down_move" in names:
        return "war_shock_reversion"
    if isinstance(names, list) and "trend_follow_thrust" in names and "breakout_strength_atr" in names:
        return "war_shock_momentum"
    if isinstance(names, list) and "volume_spike_ratio" in names:
        return "hybrid_sharpe_volume_flow"
    if isinstance(names, list) and "ema3_slope" in names and "ema9_slope" in names and "ema21_slope" not in names and "macd_hist_delta" not in names:
        return "hybrid_sharpe_selective"
    if isinstance(names, list) and "ema3_9_spread" in names and "macd_green_increasing" in names and "ema_stack_bullish" not in names and "ema_slope_alignment" not in names:
        return "hybrid_sharpe_core_no_stack"
    if isinstance(names, list) and "ema_stack_bullish" in names and "macd_green_increasing" in names and "ema_slope_alignment" not in names:
        return "hybrid_sharpe_core"
    if isinstance(names, list) and "ema3_9_spread" in names and "bb_upper" not in names and "vwap_anchor_high" not in names:
        return "ema"
    if isinstance(names, list) and "ema9_derivative_3" in names:
        return "derivative"
    if isinstance(names, list) and "macd_delta" in names and "oversold_reversal" not in names:
        return "feature2"
    return "new"
