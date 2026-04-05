from __future__ import annotations

from quant.data import _filter_regular_trading_hours


def test_filter_regular_trading_hours_removes_pre_post_and_weekends_for_intraday() -> None:
    highs = [1.0, 2.0, 3.0, 4.0, 5.0]
    lows = [0.5, 1.5, 2.5, 3.5, 4.5]
    closes = [0.8, 1.8, 2.8, 3.8, 4.8]
    timestamps = [
        "2026-04-03T13:25:00+00:00",  # 09:25 ET, pre-market
        "2026-04-03T13:30:00+00:00",  # 09:30 ET, regular session
        "2026-04-03T20:00:00+00:00",  # 16:00 ET, regular session close
        "2026-04-03T20:05:00+00:00",  # 16:05 ET, post-market
        "2026-04-04T14:00:00+00:00",  # Saturday
    ]

    kept_highs, kept_lows, kept_closes, kept_timestamps = _filter_regular_trading_hours(
        highs=highs,
        lows=lows,
        closes=closes,
        timestamps=timestamps,
        interval="5m",
    )

    assert kept_highs == [2.0, 3.0]
    assert kept_lows == [1.5, 2.5]
    assert kept_closes == [1.8, 2.8]
    assert kept_timestamps == [timestamps[1], timestamps[2]]


def test_filter_regular_trading_hours_does_not_filter_daily_interval() -> None:
    highs = [10.0, 11.0, 12.0]
    lows = [9.0, 10.0, 11.0]
    closes = [9.5, 10.5, 11.5]
    timestamps = ["2026-04-01", "2026-04-02", "2026-04-03"]

    kept_highs, kept_lows, kept_closes, kept_timestamps = _filter_regular_trading_hours(
        highs=highs,
        lows=lows,
        closes=closes,
        timestamps=timestamps,
        interval="1d",
    )

    assert kept_highs == highs
    assert kept_lows == lows
    assert kept_closes == closes
    assert kept_timestamps == timestamps
