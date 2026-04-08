from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bot_manager
from main import create_app


@pytest.fixture(autouse=True)
def reset_bot_manager_state():
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


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_create_and_list_and_get_bot(client):
    response = client.post(
        "/bots/create",
        json={
            "model": "demo-model",
            "ticker": "AAPL",
            "timeframe": "1m",
            "mode": "spot",
            "starting_money": 10000,
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "name": "Alpha Bot",
        },
    )
    assert response.status_code == 201
    created = response.get_json()
    assert created["name"] == "Alpha Bot"
    assert created["status"] == "stopped"

    list_response = client.get("/api/bots")
    assert list_response.status_code == 200
    bots = list_response.get_json()
    assert len(bots) == 1
    assert bots[0]["id"] == created["id"]

    get_response = client.get(f"/bots/{created['id']}")
    assert get_response.status_code == 200
    detail_payload = get_response.get_json()
    assert detail_payload["id"] == created["id"]
    assert detail_payload["long_only"] is True
    assert "metrics" in detail_payload
    assert "trades" in detail_payload


def test_get_bot_details_includes_trade_metrics(client):
    create_response = client.post(
        "/bots/create",
        json={
            "model": "demo-model",
            "ticker": "AAPL",
            "timeframe": "1m",
            "starting_money": 10000,
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "name": "Metrics Bot",
        },
    )
    bot_id = create_response.get_json()["id"]
    bot = bot_manager.get_bot(bot_id)
    assert bot is not None
    bot.trades.extend(
        [
            {"timestamp": "2026-04-06T15:30:00+00:00", "side": "BUY", "price": 100.0, "size": 1.0, "pnl": -20.0},
            {"timestamp": "2026-04-07T15:30:00+00:00", "side": "SELL", "price": 102.0, "size": 1.0, "pnl": 5.0},
        ]
    )

    response = client.get(f"/bots/{bot_id}")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["metrics"]["trade_count"] == 2
    assert payload["metrics"]["max_drawdown"] >= 0.0
    assert isinstance(payload["trades"], list)


def test_update_bot_settings(client):
    create_response = client.post(
        "/bots/create",
        json={
            "model": "demo-model",
            "ticker": "AAPL",
            "timeframe": "1m",
            "starting_money": 10000,
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "name": "Editable Bot",
        },
    )
    bot_id = create_response.get_json()["id"]
    response = client.patch(
        f"/bots/{bot_id}/settings",
        json={
            "name": "Edited Bot",
            "ticker": "MSFT",
            "timeframe": "5m",
            "buy_threshold": 0.7,
            "sell_threshold": 0.3,
            "trade_size": 2,
            "fixed_stop_pct": 1.5,
            "take_profit": 0.08,
            "stop_loss_strategy": "fixed_percentage",
        },
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["name"] == "Edited Bot"
    updated_bot = bot_manager.get_bot(bot_id)
    assert updated_bot is not None
    assert updated_bot.ticker == "MSFT"
    assert updated_bot.stop_loss == pytest.approx(0.015)


def test_start_and_stop_bot(client, monkeypatch):
    monkeypatch.setattr(bot_manager, "is_market_open", lambda now=None: False)
    create_response = client.post(
        "/bots/create",
        json={
            "model": "demo-model",
            "ticker": "AAPL",
            "timeframe": "1m",
            "starting_money": 10000,
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "name": "Runner",
        },
    )
    bot_id = create_response.get_json()["id"]

    start_response = client.post(f"/bots/start/{bot_id}")
    assert start_response.status_code == 200
    assert start_response.get_json()["status"] == "running"

    stop_response = client.post(f"/bots/stop/{bot_id}")
    assert stop_response.status_code == 200
    assert stop_response.get_json()["status"] == "stopped"


def test_create_bot_validation_error(client):
    response = client.post("/bots/create", json={"model": "missing-fields"})
    assert response.status_code == 400
    payload = response.get_json()
    assert "error" in payload


def test_create_bot_accepts_execution_settings(client):
    response = client.post(
        "/bots/create",
        json={
            "model": "demo-model",
            "ticker": "AAPL",
            "timeframe": "1m",
            "starting_money": 10000,
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "name": "Settings Bot",
            "execution_settings": {
                "enable_latency_simulation": True,
                "min_latency_ms": 50.0,
                "max_latency_ms": 200.0,
                "enable_spread_widening": True,
                "volatility_threshold": 0.02,
                "spread_widening_factor": 2.0,
            },
        },
    )
    assert response.status_code == 201


def test_create_bot_accepts_fixed_or_trailing_stop_strategy(client):
    trailing_response = client.post(
        "/bots/create",
        json={
            "model": "demo-model",
            "ticker": "AAPL",
            "timeframe": "1m",
            "starting_money": 10000,
            "buy_threshold": 0.65,
            "sell_threshold": 0.35,
            "stop_loss_strategy": "trailing_stop",
            "fixed_stop_pct": 2.5,
            "take_profit": 0.05,
            "name": "Trailing Bot",
        },
    )
    assert trailing_response.status_code == 201
    trailing_payload = trailing_response.get_json()
    assert trailing_payload["id"]
    created_bot = bot_manager.get_bot(trailing_payload["id"])
    assert created_bot is not None
    assert created_bot.stop_loss == pytest.approx(0.025)


def test_not_found_errors(client):
    get_response = client.get("/bots/not-real")
    assert get_response.status_code == 404
    start_response = client.post("/bots/start/not-real")
    assert start_response.status_code == 404
    stop_response = client.post("/bots/stop/not-real")
    assert stop_response.status_code == 404


def test_bots_dashboard_page(client):
    response = client.get("/bots")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Trading Bots Dashboard" in body
    assert "/api/bots" in body
    assert 'id="searchInput"' in body
    assert 'id="botStopLossStrategy"' in body
    assert 'id="botFixedStopPctWrap"' in body
    assert "Right-click a bot row to edit settings" in body
    assert 'id="botDetailModal"' in body
    assert 'id="editBotModal"' in body
    assert 'id="botContextMenu"' in body
    assert "searchInput.addEventListener(\"input\", renderRows)" in body
    assert "search_name: String(bot.name || \"\").toLowerCase()" in body
    assert "row.addEventListener(\"contextmenu\"" in body
    assert "fetch(`/bots/${selectedBotId}/settings`" in body
    assert "syncBotStopLossFields" in body


def test_spot_bots_dashboard_page(client):
    response = client.get("/spot/bots")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Trading Bots Dashboard" in body
    assert 'href="/spot"' in body
    assert 'href="/spot/manage-models"' in body
    assert 'href="/spot/run-models"' in body
    assert 'id="botDailyBuyTiming"' in body


def test_bot_form_options_include_daily_buy_timing_choices(client):
    response = client.get("/api/bots/form-options?mode=spot")
    assert response.status_code == 200
    payload = response.get_json()
    choices = payload.get("daily_buy_timing_options", [])
    assert {"value": "start_of_day", "label": "Beginning of day"} in choices
    assert {"value": "end_of_day", "label": "Last minute (end of day)"} in choices
