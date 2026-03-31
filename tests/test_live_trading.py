from __future__ import annotations

import pytest

from quant.live_trading.trading import place_order


def test_place_order_blocks_live_without_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    with pytest.raises(RuntimeError):
        place_order(
            {
                "accountId": "123",
                "symbol": "AAPL",
                "quantity": 1,
                "action": "Buy",
                "orderType": "Market",
                "isPaper": False,
            }
        )


def test_place_order_paper_mode_returns_stub() -> None:
    response = place_order(
        {
            "accountId": "123",
            "symbol": "AAPL",
            "quantity": 1,
            "action": "Buy",
            "orderType": "Market",
            "isPaper": True,
        }
    )
    assert response["status"] == "paper"


def test_place_order_rejects_invalid_quantity() -> None:
    with pytest.raises(ValueError):
        place_order(
            {
                "accountId": "123",
                "symbol": "AAPL",
                "quantity": 0,
                "action": "Buy",
                "orderType": "Market",
                "isPaper": True,
            }
        )
