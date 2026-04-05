from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from quant.data import fetch_massive_rows


class _MockResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_MockResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_fetch_massive_rows_prefers_latest_intraday_window() -> None:
    newest = datetime(2026, 4, 3, 20, 0, tzinfo=timezone.utc)
    results = []
    # Polygon returns descending results when sort=desc.
    for offset in range(160):
        ts = newest - timedelta(minutes=5 * offset)
        base_price = 200.0 - (offset * 0.1)
        results.append(
            {
                "t": int(ts.timestamp() * 1000),
                "h": base_price + 0.2,
                "l": base_price - 0.2,
                "c": base_price,
            }
        )

    payload = {"status": "OK", "results": results}
    requested_urls: list[str] = []

    def _fake_urlopen(url: str, timeout: int = 20) -> _MockResponse:
        requested_urls.append(url)
        return _MockResponse(payload)

    with patch("quant.data.urlopen", side_effect=_fake_urlopen):
        rows = fetch_massive_rows("AAPL", "5m", row_count=50, api_key="test-key")

    assert rows, "Expected intraday rows."
    assert "sort=desc" in requested_urls[0]
    # 50 rows should be recent bars from the provided descending payload,
    # not stale bars from the oldest edge of a long lookback window.
    assert str(rows[-1]["timestamp"]).startswith("2026-04-03")
    assert str(rows[0]["timestamp"]).startswith("2026-04-03")
