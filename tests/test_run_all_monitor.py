from datetime import datetime
from zoneinfo import ZoneInfo

from main import is_us_market_open


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
