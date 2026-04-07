from __future__ import annotations

import time
from datetime import datetime
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bot_manager


@pytest.fixture(autouse=True)
def reset_bot_manager_state():
    # Ensure clean state between tests.
    for bot_id in list(bot_manager.bots.keys()):
        try:
            bot_manager.stop_bot(bot_id)
        except KeyError:
            pass
        bot_manager.bots.pop(bot_id, None)
    bot_manager._BOT_THREADS.clear()
    bot_manager._BOT_STOP_EVENTS.clear()
    yield
    for bot_id in list(bot_manager.bots.keys()):
        try:
            bot_manager.stop_bot(bot_id)
        except KeyError:
            pass
        bot_manager.bots.pop(bot_id, None)


def _config(bot_id: str, fetcher):
    return {
        "id": bot_id,
        "name": "Test Bot",
        "model_name": "noop",
        "ticker": "AAPL",
        "timeframe": "1m",
        "cash": 10_000,
        "market_data_fetcher": fetcher,
        "poll_interval": 0.05,
    }


def test_market_hours_window():
    open_time = datetime.fromisoformat("2026-04-06T13:30:00+00:00")  # 09:30 ET Monday
    closed_time = datetime.fromisoformat("2026-04-06T20:00:00+00:00")  # 16:00 ET Monday

    assert bot_manager.is_market_open(open_time)
    assert not bot_manager.is_market_open(closed_time)


def test_create_get_start_stop_bot(monkeypatch):
    calls = {"n": 0}

    def fetcher(_):
        calls["n"] += 1
        return {"bid": 100.0, "ask": 100.2, "timestamp": "2026-04-06T14:00:00Z"}

    monkeypatch.setattr(bot_manager, "is_market_open", lambda now=None: True)

    created = bot_manager.create_bot(_config("bot-1", fetcher))
    assert bot_manager.get_bot("bot-1") is created
    assert len(bot_manager.get_all_bots()) == 1

    bot_manager.start_bot("bot-1")
    time.sleep(0.18)
    bot_manager.stop_bot("bot-1")

    assert created.status == "stopped"
    assert calls["n"] >= 1


def test_start_missing_bot_raises():
    with pytest.raises(KeyError):
        bot_manager.start_bot("does-not-exist")


def test_create_bot_applies_execution_settings():
    def fetcher(_):
        return None

    bot = bot_manager.create_bot(
        {
            **_config("bot-settings", fetcher),
            "execution_settings": {
                "enable_latency_simulation": True,
                "min_latency_ms": 50.0,
                "max_latency_ms": 200.0,
                "enable_spread_widening": True,
                "volatility_threshold": 0.02,
                "spread_widening_factor": 2.0,
            },
        }
    )

    assert bot.execution_engine.enable_latency_simulation is True
    assert bot.execution_engine.min_latency_ms == 50.0
    assert bot.execution_engine.max_latency_ms == 200.0
    assert bot.execution_engine.enable_spread_widening is True
