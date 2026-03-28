from quant.ml import get_strategy_feature_builder, normalize_feature_set


def test_feature2_is_supported_and_default_builder_can_be_selected() -> None:
    assert normalize_feature_set("feature2") == "feature2"
    builder = get_strategy_feature_builder("feature2")
    names = builder.names()
    assert "macd_delta" in names
    assert "inside_bull_fvg" in names
    assert "oversold_reversal" not in names


def test_derivative_feature_set_is_supported() -> None:
    assert normalize_feature_set("derivative") == "derivative"
    builder = get_strategy_feature_builder("derivative")
    names = builder.names()
    expected = {
        "macd_hist",
        "macd_hist_delta",
        "macd_green_increasing",
        "macd_red_recovering",
        "macd_green_fading",
        "macd_red_deepening",
        "ema9",
        "ema26",
        "ema9_derivative_1",
        "ema9_derivative_2",
        "ema9_derivative_3",
        "ema26_derivative_1",
        "ema26_derivative_2",
        "ema26_derivative_3",
    }
    assert expected.issubset(set(names))


def test_fvg2_feature_set_splits_stoch_extremes() -> None:
    assert normalize_feature_set("fvg_2") == "fvg2"
    builder = get_strategy_feature_builder("fvg2")
    names = builder.names()
    assert "stoch_extreme_80" in names
    assert "stoch_extreme_20" in names
    assert "stoch_extreme" not in names
    assert "fvg_conflict_penalty" in names


def test_fvg3_feature_set_is_supported() -> None:
    assert normalize_feature_set("fvg_3") == "fvg3"
    builder = get_strategy_feature_builder("fvg3")
    names = builder.names()
    assert "stoch_extreme" in names
    assert "stoch_extreme_neg" in names
    assert "macd_hist_delta_absolute" in names
    assert "fvg_conflict_penalty" in names


def test_derivative2_feature_set_is_supported() -> None:
    assert normalize_feature_set("derivate2") == "derivative2"
    builder = get_strategy_feature_builder("derivative2")
    names = builder.names()
    expected = {
        "macd_hist",
        "macd_hist_delta",
        "macd_green_increasing",
        "macd_red_recovering",
        "macd_green_fading",
        "macd_red_deepening",
        "ema9",
        "ema26",
        "ema9_derivative_1",
        "ema9_derivative_2",
        "ema9_derivative_3",
        "ema26_derivative_1",
        "ema26_derivative_2",
        "ema26_derivative_3",
        "ema_derivative_1_diff",
        "ema_derivative_2_diff",
        "ema_derivative_3_diff",
        "ema_derivative_1_cross",
        "ema_derivative_1_cross_positive",
        "ema_derivative_1_cross_negative",
        "ema_derivative_2_cross",
        "ema_derivative_2_cross_positive",
        "ema_derivative_2_cross_negative",
        "ema_derivative_3_cross",
        "ema_derivative_3_cross_positive",
        "ema_derivative_3_cross_negative",
    }
    assert expected.issubset(set(names))


def test_dqn_feature_set_is_supported() -> None:
    assert normalize_feature_set("dqn") == "dqn"
    dqn_builder = get_strategy_feature_builder("dqn")
    base_builder = get_strategy_feature_builder("feature2")
    assert dqn_builder.names() == base_builder.names()


def test_ema_feature_set_is_supported() -> None:
    assert normalize_feature_set("many-ema") == "ema"
    builder = get_strategy_feature_builder("ema")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema_stack_bullish",
        "ema_stack_bearish",
        "ema3_slope",
        "ema9_slope",
        "ema21_slope",
    }
    assert expected.issubset(set(names))


def test_bollinger_bands_feature_set_is_supported() -> None:
    assert normalize_feature_set("bollinger") == "bollinger_bands"
    builder = get_strategy_feature_builder("bollinger_bands")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_percent_b",
        "price_to_bb_mid",
    }
    assert expected.issubset(set(names))


def test_vwap_anchor_feature_set_is_supported() -> None:
    assert normalize_feature_set("anchored-vwap") == "vwap_anchor"
    builder = get_strategy_feature_builder("vwap_anchor")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "vwap_anchor_high",
        "vwap_anchor_low",
        "vwap_anchor_spread",
        "price_vs_vwap_high",
        "price_vs_vwap_low",
        "vwap_anchor_mid_bias",
    }
    assert expected.issubset(set(names))


def test_vwap_intraday_reversion_feature_set_is_supported() -> None:
    assert normalize_feature_set("intraday-vwap-reversion") == "vwap_intraday_reversion"
    builder = get_strategy_feature_builder("vwap_intraday_reversion")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "vwap_anchor_high",
        "vwap_anchor_low",
        "vwap_anchor_spread",
        "price_vs_vwap_mid",
        "distance_to_vwap_low",
        "distance_to_vwap_high",
        "zscore_vwap_mid",
        "mean_revert_long_bias",
        "mean_revert_short_bias",
    }
    assert expected.issubset(set(names))


def test_vwap_intraday_momentum_feature_set_is_supported() -> None:
    assert normalize_feature_set("intraday-vwap-momentum") == "vwap_intraday_momentum"
    builder = get_strategy_feature_builder("vwap_intraday_momentum")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema3_slope",
        "ema9_slope",
        "macd_hist",
        "macd_hist_delta",
        "vwap_anchor_high",
        "vwap_anchor_low",
        "price_vs_vwap_mid",
        "vwap_breakout_strength",
        "vwap_breakdown_strength",
        "session_trend_pressure",
    }
    assert expected.issubset(set(names))


def test_vwap_breakout_reversion_regime_feature_set_is_supported() -> None:
    assert normalize_feature_set("vwap-breakout-vs-reversion") == "vwap_breakout_reversion_regime"
    builder = get_strategy_feature_builder("vwap_breakout_reversion_regime")
    names = builder.names()
    expected = {
        "vwap_anchor_high",
        "vwap_anchor_low",
        "price_vs_vwap_mid",
        "mean_revert_long_bias",
        "mean_revert_short_bias",
        "vwap_breakout_strength",
        "vwap_breakdown_strength",
        "breakout_pressure",
        "mean_reversion_pressure",
        "vwap_regime_signal",
        "is_breakout_regime",
        "is_reversion_regime",
    }
    assert expected.issubset(set(names))


def test_hybrid_sharpe_core_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-core") == "hybrid_sharpe_core"
    builder = get_strategy_feature_builder("hybrid_sharpe_core")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema_stack_bullish",
        "ema_stack_bearish",
        "ema3_slope",
        "ema9_slope",
        "ema21_slope",
        "macd_hist",
        "macd_hist_delta",
        "macd_green_increasing",
        "macd_red_recovering",
        "macd_green_fading",
        "macd_red_deepening",
        "ema_derivative_1_diff",
        "ema_derivative_1_cross_positive",
        "ema_derivative_1_cross_negative",
    }
    assert expected.issubset(set(names))


def test_hybrid_sharpe_core_no_stack_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-core-no-stack") == "hybrid_sharpe_core_no_stack"
    builder = get_strategy_feature_builder("hybrid_sharpe_core_no_stack")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema3_slope",
        "ema9_slope",
        "ema21_slope",
        "macd_hist",
        "macd_hist_delta",
        "macd_green_increasing",
        "macd_red_recovering",
        "macd_green_fading",
        "macd_red_deepening",
        "ema_derivative_1_diff",
        "ema_derivative_1_cross_positive",
        "ema_derivative_1_cross_negative",
    }
    assert expected.issubset(set(names))
    assert "ema_stack_bullish" not in names
    assert "ema_stack_bearish" not in names


def test_hybrid_sharpe_momentum_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid_momentum") == "hybrid_sharpe_momentum"
    builder = get_strategy_feature_builder("hybrid_sharpe_momentum")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "macd_hist",
        "macd_hist_delta",
        "ema_derivative_2_diff",
        "ema_derivative_3_diff",
        "ema_derivative_2_cross_positive",
        "ema_derivative_2_cross_negative",
        "ema_derivative_3_cross_positive",
        "ema_derivative_3_cross_negative",
        "ema_slope_alignment",
        "ema_spread_balance",
    }
    assert expected.issubset(set(names))


def test_hybrid_sharpe_selective_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-selective") == "hybrid_sharpe_selective"
    builder = get_strategy_feature_builder("hybrid_sharpe_selective")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema3_slope",
        "ema9_slope",
        "macd_hist",
        "macd_green_increasing",
        "macd_red_recovering",
        "ema_derivative_1_diff",
        "ema_derivative_1_cross_positive",
        "ema_derivative_1_cross_negative",
    }
    assert expected.issubset(set(names))
    assert "ema21_slope" not in names
    assert "macd_hist_delta" not in names


def test_hybrid_sharpe_regime_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-regime") == "hybrid_sharpe_regime"
    builder = get_strategy_feature_builder("hybrid_sharpe_regime")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "trend_20",
        "vol_20",
        "ret_1",
        "ret_3",
        "stoch_rsi_norm",
        "stoch_velocity",
        "ema_slope_alignment",
    }
    assert expected.issubset(set(names))


def test_hybrid_sharpe_volume_flow_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-volume-flow") == "hybrid_sharpe_volume_flow"
    builder = get_strategy_feature_builder("hybrid_sharpe_volume_flow")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "macd_hist",
        "macd_hist_delta",
        "volume",
        "volume_ma20",
        "volume_spike_ratio",
        "signed_volume_pressure",
        "volume_volatility_coupling",
        "volume_trend_coupling",
    }
    assert expected.issubset(set(names))
    assert "ema_stack_bullish" not in names
    assert "ema_stack_bearish" not in names


def test_hybrid_sharpe_volume_regime_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-volume-regime") == "hybrid_sharpe_volume_regime"
    builder = get_strategy_feature_builder("hybrid_sharpe_volume_regime")
    names = builder.names()
    expected = {
        "volume",
        "volume_ma20",
        "volume_spike_ratio",
        "high_volume_regime",
        "low_volume_regime",
        "trend_in_high_volume",
        "pullback_in_low_volume",
    }
    assert expected.issubset(set(names))
