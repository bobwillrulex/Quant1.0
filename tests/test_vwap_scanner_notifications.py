import os
import unittest
from unittest.mock import patch

from quant.vwap_scanner import VwapScannerService, _sort_rows_by_market_cap


class TestVwapScannerNotifications(unittest.TestCase):
    def test_sort_rows_by_market_cap_desc_then_symbol(self) -> None:
        rows = [
            {"symbol": "BBB", "market_cap": 1_000_000_000.0},
            {"symbol": "AAA", "market_cap": 2_000_000_000.0},
            {"symbol": "AAC", "market_cap": 2_000_000_000.0},
        ]
        _sort_rows_by_market_cap(rows)
        self.assertEqual([row["symbol"] for row in rows], ["AAA", "AAC", "BBB"])

    @patch.dict(
        os.environ,
        {
            "VWAP_DISCORD_WEBHOOK_COLUMN_ONE": "https://discord.test/one",
            "VWAP_DISCORD_WEBHOOK_COLUMN_TWO": "https://discord.test/two",
            "VWAP_DISCORD_WEBHOOK_COLUMN_THREE": "https://discord.test/three",
        },
        clear=False,
    )
    @patch("quant.vwap_scanner.send_discord_webhook")
    def test_maybe_push_discord_updates_sends_three_messages(self, send_mock) -> None:
        scanner = VwapScannerService()
        snapshot = {
            "column_one": [{"symbol": "AAA", "price": 10.0, "market_cap": 100.0}],
            "column_two": [{"symbol": "BBB", "price": 20.0, "market_cap": 200.0}],
            "column_three": [{"symbol": "CCC", "price": 30.0, "market_cap": 300.0}],
        }

        scanner._maybe_push_discord_updates(snapshot)
        self.assertEqual(send_mock.call_count, 3)

        scanner._maybe_push_discord_updates(snapshot)
        self.assertEqual(send_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
