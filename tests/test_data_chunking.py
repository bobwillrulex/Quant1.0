from quant.data import _parse_vendor_datetime, _target_lookback_days


def test_target_lookback_days_intraday_has_two_year_floor() -> None:
    assert _target_lookback_days("5m", 100) >= 730
    assert _target_lookback_days("15m", 100) >= 730
    assert _target_lookback_days("1h", 100) >= 730


def test_parse_vendor_datetime_supports_datetime_and_date() -> None:
    assert _parse_vendor_datetime("2026-01-02 15:30:00").year == 2026
    assert _parse_vendor_datetime("2026-01-02").month == 1
