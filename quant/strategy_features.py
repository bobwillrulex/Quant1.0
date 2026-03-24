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


FeatureSet = Literal["feature2", "new", "legacy", "fvg2", "derivative", "derivative2", "dqn"]


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
    if value in ("fvg2", "fvg-2", "fvg_2", "legacy2", "legacy-fvg2"):
        return "fvg2"
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


def get_strategy_feature_builder(feature_set: FeatureSet | str = "feature2") -> StrategyFeatureBuilder:
    normalized = normalize_feature_set(feature_set)
    if normalized == "legacy":
        return build_legacy_strategy_features()
    if normalized == "fvg2":
        return build_fvg2_strategy_features()
    if normalized == "derivative2":
        return build_derivative2_strategy_features()
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
    if isinstance(names, list) and "ema_derivative_3_diff" in names:
        return "derivative2"
    if isinstance(names, list) and "ema9_derivative_3" in names:
        return "derivative"
    if isinstance(names, list) and "macd_delta" in names and "oversold_reversal" not in names:
        return "feature2"
    return "new"
