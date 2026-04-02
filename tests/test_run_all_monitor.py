from datetime import datetime
from zoneinfo import ZoneInfo

from main import is_us_market_open, seconds_until_next_aligned_five_minute


def test_is_us_market_open_during_session() -> None:
    ts = datetime(2026, 3, 20, 10, 0, tzinfo=ZoneInfo("America/New_York"))  # Friday
    assert is_us_market_open(ts)


def test_is_us_market_open_outside_session() -> None:
    pre = datetime(2026, 3, 20, 9, 0, tzinfo=ZoneInfo("America/New_York"))
    post = datetime(2026, 3, 20, 16, 1, tzinfo=ZoneInfo("America/New_York"))
    weekend = datetime(2026, 3, 21, 11, 0, tzinfo=ZoneInfo("America/New_York"))
    assert not is_us_market_open(pre)
    assert not is_us_market_open(post)
    assert not is_us_market_open(weekend)


def test_next_five_minute_alignment_waits_until_next_boundary() -> None:
    ts = datetime(2026, 3, 20, 10, 4, 15)
    assert seconds_until_next_aligned_five_minute(ts) == 45.0


def test_next_five_minute_alignment_rolls_hour() -> None:
    ts = datetime(2026, 3, 20, 10, 59, 59)
    assert seconds_until_next_aligned_five_minute(ts) == 1.0
