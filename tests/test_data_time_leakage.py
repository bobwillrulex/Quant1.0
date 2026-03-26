from quant.data import compute_strategy_rows_from_prices


def _build_ohlc(n: int = 60):
    highs = [100.0 + i for i in range(n)]
    lows = [99.0 + i for i in range(n)]
    closes = [99.5 + i for i in range(n)]
    return highs, lows, closes


def test_fvg_features_do_not_depend_on_next_candle_values() -> None:
    highs_a, lows_a, closes = _build_ohlc()
    highs_b, lows_b, _ = _build_ohlc()

    i = 20  # Row index in output is i - 3.
    lows_a[i] = 116.0
    lows_b[i] = 116.0

    # Change only candle i+1 values; row i features must remain identical.
    lows_a[i + 1] = 120.0
    highs_a[i + 1] = 130.0
    lows_b[i + 1] = 105.0
    highs_b[i + 1] = 140.0

    rows_a = compute_strategy_rows_from_prices(highs=highs_a, lows=lows_a, closes=closes)
    rows_b = compute_strategy_rows_from_prices(highs=highs_b, lows=lows_b, closes=closes)

    row_index = i - 3
    keys = [
        "fvg_green_size",
        "fvg_red_size",
        "fvg_red_above_green",
        "first_green_fvg_dip",
        "first_red_fvg_touch",
    ]
    for key in keys:
        assert rows_a[row_index][key] == rows_b[row_index][key]


def test_feature2_columns_are_present_in_rows() -> None:
    highs, lows, closes = _build_ohlc(80)
    rows = compute_strategy_rows_from_prices(highs=highs, lows=lows, closes=closes)
    sample = rows[-1]
    for key in [
        "stoch_velocity",
        "stoch_low_zone",
        "stoch_high_zone",
        "macd_delta",
        "ret_1",
        "ret_3",
        "ret_5",
        "trend_20",
        "vol_20",
        "dist_to_bull_fvg",
        "dist_to_bear_fvg",
        "inside_bull_fvg",
        "inside_bear_fvg",
    ]:
        assert key in sample


def test_derivative_columns_are_present_in_rows() -> None:
    highs, lows, closes = _build_ohlc(80)
    rows = compute_strategy_rows_from_prices(highs=highs, lows=lows, closes=closes)
    sample = rows[-1]
    for key in [
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
    ]:
        assert key in sample


def test_derivative_crossovers_use_only_current_and_previous_bar() -> None:
    highs_a, lows_a, closes_a = _build_ohlc(80)
    highs_b, lows_b, closes_b = _build_ohlc(80)

    i = 20  # row index in output is i - 3
    highs_a[i + 1] = highs_a[i + 1] + 15.0
    lows_a[i + 1] = lows_a[i + 1] - 15.0
    closes_a[i + 1] = closes_a[i + 1] + 20.0

    rows_a = compute_strategy_rows_from_prices(highs=highs_a, lows=lows_a, closes=closes_a)
    rows_b = compute_strategy_rows_from_prices(highs=highs_b, lows=lows_b, closes=closes_b)

    row_index = i - 3
    keys = [
        "ema_derivative_1_diff",
        "ema_derivative_1_cross",
        "ema_derivative_1_cross_positive",
        "ema_derivative_1_cross_negative",
        "ema_derivative_2_diff",
        "ema_derivative_2_cross",
        "ema_derivative_2_cross_positive",
        "ema_derivative_2_cross_negative",
        "ema_derivative_3_diff",
        "ema_derivative_3_cross",
        "ema_derivative_3_cross_positive",
        "ema_derivative_3_cross_negative",
    ]
    for key in keys:
        assert rows_a[row_index][key] == rows_b[row_index][key]


def test_ema_bollinger_and_vwap_columns_are_present_and_dynamic() -> None:
    highs, lows, closes = _build_ohlc(120)
    rows = compute_strategy_rows_from_prices(highs=highs, lows=lows, closes=closes)
    sample = rows[-1]
    required_keys = [
        "ema3",
        "ema9",
        "ema21",
        "ema3_derivative_1",
        "ema9_derivative_1",
        "ema21_derivative_1",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_percent_b",
        "vwap_anchor_high",
        "vwap_anchor_low",
    ]
    for key in required_keys:
        assert key in sample

    vwap_high_values = [row["vwap_anchor_high"] for row in rows]
    vwap_low_values = [row["vwap_anchor_low"] for row in rows]
    assert max(vwap_high_values) - min(vwap_high_values) > 0.0
    assert max(vwap_low_values) - min(vwap_low_values) > 0.0
