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
