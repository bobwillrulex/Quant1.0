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
    yield
    for bot_id in list(bot_manager.bots.keys()):
        try:
            bot_manager.stop_bot(bot_id)
        except KeyError:
            pass
        bot_manager.bots.pop(bot_id, None)


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
    assert get_response.get_json()["id"] == created["id"]


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
    assert "searchInput.addEventListener(\"input\", renderRows)" in body
    assert "search_name: String(bot.name || \"\").toLowerCase()" in body


def test_spot_bots_dashboard_page(client):
    response = client.get("/spot/bots")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Trading Bots Dashboard" in body
    assert 'href="/spot"' in body
    assert 'href="/spot/manage-models"' in body
    assert 'href="/spot/run-models"' in body
