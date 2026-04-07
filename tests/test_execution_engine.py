from __future__ import annotations

from types import SimpleNamespace

import pytest

from quant.execution_engine import ExecutionEngine


def _make_bot(*, trade_size: float = 10.0, cash: float = 10000.0) -> SimpleNamespace:
    return SimpleNamespace(
        trade_size=trade_size,
        position_size=0.0,
        average_entry_price=0.0,
        realized_pnl=0.0,
        cash=cash,
        trades=[],
    )


def test_market_buy_uses_ask_and_updates_state() -> None:
    bot = _make_bot()
    engine = ExecutionEngine(enable_slippage=False)

    trade = engine.execute_market_buy(bot, {"bid": 99.0, "ask": 100.0, "timestamp": "2026-04-07T00:00:00Z"})

    assert trade["side"] == "BUY"
    assert trade["price"] == 100.0
    assert trade["size"] == 10.0
    assert bot.position_size == 10.0
    assert bot.average_entry_price == 100.0
    assert bot.realized_pnl == 0.0
    assert bot.cash == 9000.0
    assert bot.trades[-1] == trade


def test_market_sell_uses_bid_and_realizes_pnl() -> None:
    bot = _make_bot()
    engine = ExecutionEngine(enable_slippage=False)

    engine.execute_market_buy(bot, {"bid": 99.0, "ask": 100.0, "timestamp": "2026-04-07T00:00:00Z"})
    trade = engine.execute_market_sell(bot, {"bid": 105.0, "ask": 106.0, "timestamp": "2026-04-07T00:01:00Z"})

    assert trade["side"] == "SELL"
    assert trade["price"] == 105.0
    assert bot.position_size == 0.0
    assert bot.average_entry_price == 0.0
    assert bot.realized_pnl == 50.0
    assert bot.cash == 10050.0
    assert trade["pnl"] == 50.0


def test_slippage_is_deterministic_with_seeded_rng() -> None:
    bot = _make_bot(trade_size=1.0)
    engine = ExecutionEngine(enable_slippage=True)

    trade = engine.execute_market_buy(bot, {"bid": 99.0, "ask": 100.0, "timestamp": "2026-04-07T00:00:00Z"})

    assert trade["price"] == pytest.approx(100.0168884370305)
    assert bot.cash == pytest.approx(9899.98311156297)
