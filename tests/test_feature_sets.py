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
