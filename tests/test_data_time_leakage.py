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
