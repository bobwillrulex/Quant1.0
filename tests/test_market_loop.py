from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime

from bot import TradingBot
from market_loop import MarketLoopConfig, MarketLoopScheduler


def _running_bot(bot_id: str = "b1") -> TradingBot:
    bot = TradingBot(
        id=bot_id,
        name="Bot",
        model_name="noop",
        ticker="AAPL",
        timeframe="1m",
        cash=1000,
    )
    bot.status = "running"
    return bot


def test_updates_pnl_from_quote_outside_market(monkeypatch):
    bot = _running_bot()
    bot.position_size = 10
    bot.average_entry_price = 100.0

    quote_calls = {"n": 0}

    def quote_fetcher(_):
        quote_calls["n"] += 1
        return {"bid": 101.0, "ask": 101.2, "timestamp": "2026-04-06T20:30:00Z"}

    scheduler = MarketLoopScheduler(
        bots_provider=lambda: [bot],
        quote_fetcher=quote_fetcher,
        candle_fetcher=lambda _: {"close": 101.1, "timestamp": "2026-04-06T20:30:00Z"},
        config=MarketLoopConfig(min_quote_interval_seconds=0.0),
    )

    monkeypatch.setattr("market_loop.is_market_open", lambda now=None: False)

    calls = {"on_new_candle": 0}

    def no_trade(_):
        calls["on_new_candle"] += 1
        return {}

    bot.on_new_candle = no_trade  # type: ignore[assignment]

    scheduler.run_once(now=datetime.fromisoformat("2026-04-06T20:30:00+00:00"))

    assert quote_calls["n"] == 1
    assert calls["on_new_candle"] == 0
    assert bot.total_pnl > 0.0


def test_calls_on_new_candle_once_per_bucket(monkeypatch):
    bot = _running_bot()

    quotes = [
        {"bid": 100.0, "ask": 100.2, "timestamp": "2026-04-06T14:00:02Z"},
        {"bid": 100.1, "ask": 100.3, "timestamp": "2026-04-06T14:00:40Z"},
        {"bid": 100.2, "ask": 100.4, "timestamp": "2026-04-06T14:01:01Z"},
    ]

    def quote_fetcher(_):
        return quotes.pop(0)

    scheduler = MarketLoopScheduler(
        bots_provider=lambda: [bot],
        quote_fetcher=quote_fetcher,
        candle_fetcher=lambda _: {"close": 100.2, "timestamp": "2026-04-06T14:00:00Z"},
        config=MarketLoopConfig(min_quote_interval_seconds=0.0),
    )

    monkeypatch.setattr("market_loop.is_market_open", lambda now=None: True)

    calls = {"n": 0}

    def on_new_candle(_):
        calls["n"] += 1
        return {"action": "HOLD"}

    bot.on_new_candle = on_new_candle  # type: ignore[assignment]

    scheduler.run_once(now=datetime.fromisoformat("2026-04-06T14:00:02+00:00"))
    scheduler.run_once(now=datetime.fromisoformat("2026-04-06T14:00:40+00:00"))
    scheduler.run_once(now=datetime.fromisoformat("2026-04-06T14:01:01+00:00"))

    assert calls["n"] == 2
