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
