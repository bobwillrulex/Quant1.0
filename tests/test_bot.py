from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot import TradingBot


def test_long_only_prevents_short_entries():
    bot = TradingBot(
        id="long-only",
        name="LongOnly",
        model_name="noop",
        ticker="AAPL",
        timeframe="1m",
        cash=1000,
        long_only=True,
    )
    bot.status = "running"
    bot.model = lambda _row: 0.1
    result = bot.on_new_candle({"close": 100.0, "timestamp": "2026-04-07T14:00:00+00:00"})
    assert result["action"] == "HOLD"
    assert bot.position_size == 0.0


def test_daily_buy_timing_blocks_midday_buys():
    bot = TradingBot(
        id="daily-buy",
        name="DailyBuy",
        model_name="noop",
        ticker="AAPL",
        timeframe="1d",
        cash=1000,
        daily_buy_timing="end_of_day",
    )
    bot.status = "running"
    bot.buy_threshold = 0.6
    bot.sell_threshold = 0.4
    bot.model = lambda _row: 0.9

    midday_result = bot.on_new_candle({"close": 100.0, "timestamp": "2026-04-07T17:00:00+00:00"})
    assert midday_result["action"] == "HOLD"
    assert bot.position_size == 0.0

    eod_result = bot.on_new_candle({"close": 100.0, "timestamp": "2026-04-07T19:59:00+00:00"})
    assert eod_result["action"] == "BUY"
    assert bot.position_size > 0.0


def test_intraday_trade_interval_enforces_anchored_10_minute_boundaries():
    bot = TradingBot(
        id="intraday-gate",
        name="IntradayGate",
        model_name="noop",
        ticker="AAPL",
        timeframe="1m",
        cash=1000,
        intraday_trade_interval="10m",
    )
    bot.status = "running"
    bot.model = lambda _row: 0.9

    blocked = bot.on_new_candle({"close": 100.0, "timestamp": "2026-04-07T14:12:00+00:00"})
    assert blocked["action"] == "HOLD"
    assert bot.position_size == 0.0

    allowed = bot.on_new_candle({"close": 100.0, "timestamp": "2026-04-07T14:10:00+00:00"})
    assert allowed["action"] == "BUY"
    assert bot.position_size > 0.0


def test_tracks_last_polled_bid_ask_spread():
    bot = TradingBot(
        id="spread",
        name="Spread",
        model_name="noop",
        ticker="AAPL",
        timeframe="1m",
        cash=1000,
    )
    bot.update_pnl({"bid": 100.10, "ask": 100.30, "timestamp": "2026-04-08T14:35:00+00:00"})
    assert bot.last_polled_bid == 100.10
    assert bot.last_polled_ask == 100.30
    assert bot.last_polled_spread == pytest.approx(0.20)
    assert bot.last_polled_timestamp == "2026-04-08T14:35:00+00:00"


def test_tracks_last_p_up_from_model_signal():
    bot = TradingBot(
        id="p-up",
        name="PUp",
        model_name="noop",
        ticker="AAPL",
        timeframe="1m",
        cash=1000,
    )
    bot.status = "running"
    bot.model = lambda _row: 0.73
    bot.on_new_candle({"close": 100.0, "timestamp": "2026-04-08T14:40:00+00:00"})
    assert bot.last_p_up == pytest.approx(0.73)
