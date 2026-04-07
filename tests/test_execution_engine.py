from __future__ import annotations

import random
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


def test_latency_simulation_sleeps_within_configured_window(monkeypatch) -> None:
    bot = _make_bot(trade_size=1.0)
    engine = ExecutionEngine(
        enable_latency_simulation=True,
        min_latency_ms=50.0,
        max_latency_ms=200.0,
        rng=random.Random(0),
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr("quant.execution_engine.time.sleep", lambda value: sleep_calls.append(value))

    engine.execute_market_buy(bot, {"bid": 99.0, "ask": 100.0, "timestamp": "2026-04-07T00:00:00Z"})

    assert len(sleep_calls) == 1
    assert 0.05 <= sleep_calls[0] <= 0.2


def test_spread_widens_only_when_volatility_is_high() -> None:
    low_bot = _make_bot(trade_size=1.0)
    high_bot = _make_bot(trade_size=1.0)
    engine = ExecutionEngine(enable_spread_widening=True, volatility_threshold=0.02, spread_widening_factor=2.0)

    low_trade = engine.execute_market_buy(
        low_bot,
        {"bid": 99.0, "ask": 101.0, "volatility": 0.01, "timestamp": "2026-04-07T00:00:00Z"},
    )
    high_trade = engine.execute_market_buy(
        high_bot,
        {"bid": 99.0, "ask": 101.0, "volatility": 0.03, "timestamp": "2026-04-07T00:01:00Z"},
    )

    assert low_trade["price"] == 101.0
    assert high_trade["price"] == 102.0
