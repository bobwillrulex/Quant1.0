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
    bot_manager.clear_persisted_bots()
    yield
    for bot_id in list(bot_manager.bots.keys()):
        try:
            bot_manager.stop_bot(bot_id)
        except KeyError:
            pass
        bot_manager.bots.pop(bot_id, None)
    bot_manager.clear_persisted_bots()


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


def test_seconds_until_next_aligned_poll_tick_anchors_to_five_second_boundaries():
    ts = datetime.fromisoformat("2026-04-06T14:00:02+00:00")
    assert bot_manager._seconds_until_next_aligned_poll_tick(5.0, now=ts) == pytest.approx(3.0)

    ts_on_boundary = datetime.fromisoformat("2026-04-06T14:00:05+00:00")
    assert bot_manager._seconds_until_next_aligned_poll_tick(5.0, now=ts_on_boundary) == pytest.approx(5.0)


def test_seconds_until_market_wakeup_targets_five_seconds_before_open():
    monday_pre_open = datetime.fromisoformat("2026-04-06T09:00:00-04:00")
    wait = bot_manager._seconds_until_market_wakeup(now=monday_pre_open)
    assert wait == pytest.approx((29 * 60) + 55)

    monday_after_close = datetime.fromisoformat("2026-04-06T16:01:00-04:00")
    wait_after_close = bot_manager._seconds_until_market_wakeup(now=monday_after_close)
    assert wait_after_close == pytest.approx((17 * 3600) + (28 * 60) + 55)


def test_create_get_start_stop_bot(monkeypatch):
    calls = {"n": 0}

    def fetcher(_):
        calls["n"] += 1
        return {"bid": 100.0, "ask": 100.2, "timestamp": "2026-04-06T14:00:00Z"}

    monkeypatch.setattr(bot_manager, "is_market_open", lambda now=None: True)
    monkeypatch.setattr(bot_manager, "_seconds_until_next_aligned_poll_tick", lambda *_args, **_kwargs: 0.01)

    created = bot_manager.create_bot(_config("bot-1", fetcher))
    assert bot_manager.get_bot("bot-1") is created
    assert len(bot_manager.get_all_bots()) == 1

    bot_manager.start_bot("bot-1")
    time.sleep(0.18)
    bot_manager.stop_bot("bot-1")

    assert created.status == "stopped"
    assert calls["n"] >= 1


def test_running_bot_updates_quote_even_when_market_closed(monkeypatch):
    calls = {"n": 0}

    def fetcher(_):
        calls["n"] += 1
        return {"bid": 100.0, "ask": 100.2, "timestamp": "2026-04-06T12:00:00Z"}

    monkeypatch.setattr(bot_manager, "is_market_open", lambda now=None: False)
    monkeypatch.setattr(bot_manager, "_seconds_until_next_aligned_poll_tick", lambda *_args, **_kwargs: 0.01)

    created = bot_manager.create_bot(_config("bot-closed", fetcher))
    bot_manager.start_bot("bot-closed")
    time.sleep(0.08)
    bot_manager.stop_bot("bot-closed")

    assert calls["n"] >= 1
    assert created.last_polled_bid == pytest.approx(100.0)
    assert created.last_polled_ask == pytest.approx(100.2)


def test_running_bot_loop_survives_fetch_errors(monkeypatch):
    calls = {"n": 0}

    def fetcher(_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary quote failure")
        return {"bid": 101.0, "ask": 101.3, "timestamp": "2026-04-06T14:05:00Z"}

    monkeypatch.setattr(bot_manager, "is_market_open", lambda now=None: True)
    monkeypatch.setattr(bot_manager, "_seconds_until_next_aligned_poll_tick", lambda *_args, **_kwargs: 0.01)

    created = bot_manager.create_bot(_config("bot-errors", fetcher))
    bot_manager.start_bot("bot-errors")
    time.sleep(0.12)
    bot_manager.stop_bot("bot-errors")

    assert calls["n"] >= 2
    assert created.last_polled_bid == pytest.approx(101.0)
    assert created.last_polled_ask == pytest.approx(101.3)
    assert getattr(created, "last_error", "") == "temporary quote failure"


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


def test_bot_state_persists_to_database_and_reloads():
    def fetcher(_):
        return None

    created = bot_manager.create_bot(_config("persist-1", fetcher))
    created.trades.append({"timestamp": "2026-04-06T14:00:00Z", "side": "BUY", "price": 100.0, "size": 1.0, "pnl": 0.0})
    created.total_pnl = 12.5
    created.status = "running"
    bot_manager.persist_bot(created)

    bot_manager.bots.clear()
    bot_manager._load_persisted_bots()

    reloaded = bot_manager.get_bot("persist-1")
    assert reloaded is not None
    assert reloaded.status == "running"
    assert reloaded.total_pnl == pytest.approx(12.5)
    assert len(reloaded.trades) == 1
