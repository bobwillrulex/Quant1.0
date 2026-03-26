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
    "hybrid_sharpe_core",
    "hybrid_sharpe_core_no_stack",
    "hybrid_sharpe_momentum",
    "hybrid_sharpe_selective",
    "hybrid_sharpe_regime",
    "hybrid_sharpe_volume_flow",
    "hybrid_sharpe_volume_regime",
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
    if value in ("fvg2", "fvg-2", "fvg_2", "legacy2", "legacy-fvg2"):
        return "fvg2"
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


def get_strategy_feature_builder(feature_set: FeatureSet | str = "feature2") -> StrategyFeatureBuilder:
    normalized = normalize_feature_set(feature_set)
    if normalized == "legacy":
        return build_legacy_strategy_features()
    if normalized == "fvg2":
        return build_fvg2_strategy_features()
    if normalized == "fvg3":
        return build_fvg3_strategy_features()
    if normalized == "derivative2":
        return build_derivative2_strategy_features()
    if normalized == "ema":
        return build_ema_strategy_features()
    if normalized == "bollinger_bands":
        return build_bollinger_bands_strategy_features()
    if normalized == "vwap_anchor":
        return build_vwap_anchor_strategy_features()
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
    if isinstance(names, list) and "stoch_extreme_neg" in names and "macd_hist_delta_absolute" in names:
        return "fvg3"
    if isinstance(names, list) and "ema_derivative_3_diff" in names:
        return "derivative2"
    if isinstance(names, list) and "bb_upper" in names and "bb_lower" in names:
        return "bollinger_bands"
    if isinstance(names, list) and "vwap_anchor_high" in names and "vwap_anchor_low" in names:
        return "vwap_anchor"
    if isinstance(names, list) and "ema_slope_alignment" in names and "ema_spread_balance" in names:
        return "hybrid_sharpe_momentum"
    if isinstance(names, list) and "trend_20" in names and "vol_20" in names and "stoch_rsi_norm" in names:
        return "hybrid_sharpe_regime"
    if isinstance(names, list) and "volume_spike_ratio" in names and "high_volume_regime" in names:
        return "hybrid_sharpe_volume_regime"
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
