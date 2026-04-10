from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, time, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from bot import TradingBot
from quant.constants import OPTIONS_MODE, SPOT_MODE
from quant.data import compute_strategy_rows_from_prices
from quant.execution_engine import ExecutionEngine
from quant.env_bootstrap import load_local_env_files
from quant.live_trading.auth import is_terminal_auth_failure
from quant.ml import predict_signal
from quant.storage import db_path, ensure_db
from quant.storage import load_model_bundle


MarketDataFetcher = Callable[[TradingBot], dict[str, Any] | None]


bots: dict[str, TradingBot] = {}


_LOCK = threading.RLock()
_BOT_THREADS: dict[str, threading.Thread] = {}
_BOT_STOP_EVENTS: dict[str, threading.Event] = {}
_DEFAULT_POLL_SECONDS = 5.0
_POLL_ALIGNMENT_SECONDS = 5.0
_PRE_OPEN_WAKE_BUFFER_SECONDS = 5.0
_EST_TZ = ZoneInfo("America/New_York")
_LOGGER = logging.getLogger(__name__)
_QUESTRADE_CLIENTS_BY_REFRESH_TOKEN: dict[str, Any] = {}


def _normalize_timeframe(value: object) -> str:
    token = str(value or "").strip().lower()
    if not token:
        return "1m"

    compact = token.replace(" ", "")
    aliases = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "60min": "1h",
        "1hr": "1h",
        "1hour": "1h",
        "daily": "1d",
        "day": "1d",
        "d": "1d",
    }
    return aliases.get(compact, compact)


def _default_poll_interval_for_timeframe(timeframe: object) -> float:
    normalized = _normalize_timeframe(timeframe)
    if normalized.endswith("m"):
        try:
            minutes = float(normalized[:-1])
            return max(_POLL_ALIGNMENT_SECONDS, minutes * 60.0)
        except ValueError:
            return _DEFAULT_POLL_SECONDS
    if normalized.endswith("h"):
        try:
            hours = float(normalized[:-1])
            return max(_POLL_ALIGNMENT_SECONDS, hours * 3600.0)
        except ValueError:
            return _DEFAULT_POLL_SECONDS
    if normalized in {"1d"}:
        return 24.0 * 3600.0
    return _DEFAULT_POLL_SECONDS


def create_bot(config: dict[str, Any], *, persist: bool = True) -> TradingBot:
    """Create and register a bot from config.

    Expected config keys for ``TradingBot``:
      id, name, model_name, ticker, timeframe, cash

    Optional runtime keys:
      market_data_fetcher: Callable[[TradingBot], dict[str, Any] | None]
      poll_interval: float (seconds)
    """
    runtime_config = dict(config)
    runtime_config["timeframe"] = _normalize_timeframe(runtime_config.get("timeframe", "1m"))
    mode = str(runtime_config.pop("mode", SPOT_MODE if bool(runtime_config.get("long_only", False)) else OPTIONS_MODE))
    prediction_horizon = int(runtime_config.pop("prediction_horizon", 5) or 5)
    market_data_fetcher = runtime_config.pop("market_data_fetcher", None)
    if market_data_fetcher is None:
        market_data_fetcher = _build_default_fetcher({**runtime_config, "prediction_horizon": prediction_horizon})
    model_predictor = runtime_config.pop("model_predictor", None)
    if model_predictor is None:
        model_predictor = _build_default_model_predictor({**runtime_config, "mode": mode})
    poll_interval = runtime_config.pop(
        "poll_interval",
        _default_poll_interval_for_timeframe(runtime_config.get("timeframe", "1m")),
    )
    execution_settings = runtime_config.pop("execution_settings", None)
    if execution_settings is not None:
        if not isinstance(execution_settings, dict):
            raise ValueError("execution_settings must be an object.")
        runtime_config["execution_engine"] = ExecutionEngine(**execution_settings)

    bot = TradingBot(**runtime_config)
    if market_data_fetcher is not None:
        setattr(bot, "market_data_fetcher", market_data_fetcher)
    if callable(model_predictor):
        setattr(bot, "model", model_predictor)
    setattr(bot, "poll_interval", poll_interval)
    setattr(bot, "mode", mode)
    setattr(bot, "prediction_horizon", prediction_horizon)
    with _LOCK:
        if bot.id in bots:
            raise ValueError(f"Bot with id '{bot.id}' already exists")
        bots[bot.id] = bot
    if persist:
        _save_bot_state(bot)
    return bot


def _build_default_fetcher(config: dict[str, Any]) -> MarketDataFetcher:
    ticker = str(config.get("ticker", "")).strip().upper()
    if not ticker:
        return lambda _bot: None
    load_local_env_files()
    refresh_token = str(config.get("questrade_refresh_token") or "").strip() or str(os.environ.get("QUESTRADE_REFRESH_TOKEN", "")).strip()
    client = None
    questrade_error_type: type[Exception] = Exception
    if refresh_token:
        try:
            from questrade_client import QuestradeClient, QuestradeError
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing optional dependency 'requests' required by questrade_client. "
                "Install requests or configure a custom market_data_fetcher."
            ) from exc
        with _LOCK:
            shared_client = _QUESTRADE_CLIENTS_BY_REFRESH_TOKEN.get(refresh_token)
            if shared_client is None:
                shared_client = QuestradeClient(refresh_token=refresh_token)
                _QUESTRADE_CLIENTS_BY_REFRESH_TOKEN[refresh_token] = shared_client
            client = shared_client
        questrade_error_type = QuestradeError

    timeframe = _normalize_timeframe(config.get("timeframe", "1m"))
    prediction_horizon = int(config.get("prediction_horizon", 5) or 5)
    interval_by_timeframe = {
        "1m": "OneMinute",
        "5m": "FiveMinutes",
        "15m": "FifteenMinutes",
        "30m": "ThirtyMinutes",
        "1h": "OneHour",
        "60m": "OneHour",
        "1d": "OneDay",
        "d": "OneDay",
        "day": "OneDay",
        "daily": "OneDay",
    }
    candle_interval = interval_by_timeframe.get(timeframe, "OneMinute")

    def _fetch_quote(_: TradingBot) -> dict[str, Any] | None:
        if client is None:
            raise RuntimeError("Missing Questrade refresh token. Set QUESTRADE_REFRESH_TOKEN before starting bots.")
        latest_row: dict[str, Any] | None = None
        try:
            candles = client.get_candles(ticker, candle_interval)
            if candles:
                highs = [float(candle.get("high", 0.0)) for candle in candles]
                lows = [float(candle.get("low", 0.0)) for candle in candles]
                closes = [float(candle.get("close", 0.0)) for candle in candles]
                timestamps = [candle.get("timestamp") for candle in candles]
                rows = compute_strategy_rows_from_prices(
                    highs,
                    lows,
                    closes,
                    prediction_horizon=max(1, prediction_horizon),
                    timestamps=timestamps,
                )
                if rows:
                    latest_row = dict(rows[-1])
        except Exception:
            latest_row = None
        try:
            quote = client.get_quote(ticker)
        except questrade_error_type as exc:
            raise RuntimeError(f"Questrade quote fetch failed for {ticker}: {exc}") from exc
        bid = float(quote.get("bid", 0.0))
        ask = float(quote.get("ask", 0.0))
        last = float(quote.get("last", 0.0)) or ((bid + ask) / 2.0 if (bid or ask) else 0.0)
        timestamp = quote.get("timestamp")
        payload = dict(latest_row or {})
        payload.update(
            {
                "bid": bid,
                "ask": ask,
                "last": last,
                "close": float(payload.get("close", last)),
                "timestamp": str(timestamp or datetime.now(tz=timezone.utc).isoformat()),
            }
        )
        return payload

    return _fetch_quote


def _build_default_model_predictor(config: dict[str, Any]) -> Callable[[dict[str, Any]], float] | None:
    model_name = str(config.get("model_name", "")).strip()
    if not model_name:
        return None
    mode_text = str(config.get("mode", "")).strip().lower()
    if mode_text not in {"spot", "options"}:
        mode_text = "spot" if bool(config.get("long_only", False)) else "options"
    mode = SPOT_MODE if mode_text == "spot" else OPTIONS_MODE
    long_only = mode == SPOT_MODE
    try:
        bundle = load_model_bundle(mode, model_name)
    except Exception:
        return None

    def _predict(row: dict[str, Any]) -> float:
        try:
            prediction = predict_signal(bundle, row, long_only=long_only)
            return float(prediction.get("p_up", 0.5))
        except Exception:
            return 0.5

    return _predict


def start_bot(bot_id: str) -> TradingBot:
    """Start a bot loop in a dedicated thread."""
    with _LOCK:
        bot = _require_bot(bot_id)

        existing = _BOT_THREADS.get(bot_id)
        if existing is not None and existing.is_alive():
            return bot

        stop_event = threading.Event()
        _BOT_STOP_EVENTS[bot_id] = stop_event
        bot.status = "running"

        thread = threading.Thread(
            target=_bot_loop,
            args=(bot, stop_event),
            name=f"bot-loop-{bot_id}",
            daemon=True,
        )
        _BOT_THREADS[bot_id] = thread
        thread.start()
        _save_bot_state(bot)
        return bot


def stop_bot(bot_id: str) -> TradingBot:
    """Stop a running bot thread safely."""
    with _LOCK:
        bot = _require_bot(bot_id)
        bot.status = "stopped"
        stop_event = _BOT_STOP_EVENTS.get(bot_id)
        thread = _BOT_THREADS.get(bot_id)

    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)

    with _LOCK:
        _BOT_STOP_EVENTS.pop(bot_id, None)
        _BOT_THREADS.pop(bot_id, None)
        _save_bot_state(bot)
    return bot


def get_all_bots() -> list[TradingBot]:
    """Return all registered bots as a snapshot list."""
    with _LOCK:
        return list(bots.values())


def get_bot(bot_id: str) -> TradingBot | None:
    """Return a bot by id."""
    with _LOCK:
        return bots.get(bot_id)


def delete_bot(bot_id: str) -> None:
    """Delete a bot from memory and persisted storage."""
    with _LOCK:
        if bot_id not in bots:
            raise KeyError(f"Bot '{bot_id}' not found")
    stop_bot(bot_id)
    with _LOCK:
        bots.pop(bot_id, None)
    _delete_persisted_bot(bot_id)


def is_market_open(now: datetime | None = None) -> bool:
    """Check U.S. equity regular market hours (9:30–16:00 ET, weekdays)."""
    current = now or datetime.now(tz=_EST_TZ)
    et = current.astimezone(_EST_TZ)

    if et.weekday() >= 5:  # Sat/Sun
        return False

    minutes = et.hour * 60 + et.minute
    market_open = 9 * 60 + 30
    market_close = 16 * 60
    return market_open <= minutes < market_close


def _bot_loop(bot: TradingBot, stop_event: threading.Event) -> None:
    fetcher = _extract_fetcher(bot)
    poll_interval = _extract_poll_interval(bot)

    while not stop_event.is_set():
        if bot.status != "running":
            wait_seconds = _seconds_until_market_wakeup()
            stop_event.wait(timeout=wait_seconds)
            continue

        wait_seconds = _seconds_until_next_aligned_poll_tick(poll_interval)
        try:
            payload = fetcher(bot)
            if payload is not None:
                bot.on_market_data(payload, allow_trades=is_market_open())
                _save_bot_state(bot)
        except Exception as exc:  # noqa: BLE001
            setattr(bot, "last_error", str(exc))
            if is_terminal_auth_failure(exc):
                bot.status = "stopped"
                _save_bot_state(bot)
                _LOGGER.error("Bot %s stopped: terminal Questrade auth failure. Manual token refresh required.", bot.id)
            _LOGGER.exception("Bot loop error for %s: %s", bot.id, exc)
        stop_event.wait(timeout=wait_seconds)


def _extract_fetcher(bot: TradingBot) -> MarketDataFetcher:
    fetcher = getattr(bot, "market_data_fetcher", None)
    if callable(fetcher):
        return fetcher

    def _default_fetcher(_: TradingBot) -> dict[str, Any] | None:
        return None

    return _default_fetcher


def _extract_poll_interval(bot: TradingBot) -> float:
    raw = getattr(bot, "poll_interval", _DEFAULT_POLL_SECONDS)
    try:
        interval = float(raw)
    except (TypeError, ValueError):
        interval = _DEFAULT_POLL_SECONDS
    return max(_POLL_ALIGNMENT_SECONDS, interval)


def _seconds_until_next_aligned_poll_tick(poll_interval: float, now: datetime | None = None) -> float:
    current = now or datetime.now(tz=timezone.utc)
    aligned_interval = max(_POLL_ALIGNMENT_SECONDS, float(poll_interval))
    epoch = current.timestamp()
    remainder = epoch % aligned_interval
    wait_seconds = aligned_interval - remainder
    if wait_seconds <= 1e-9:
        wait_seconds = aligned_interval
    return wait_seconds


def _seconds_until_market_wakeup(now: datetime | None = None) -> float:
    current = now or datetime.now(tz=_EST_TZ)
    et_now = current.astimezone(_EST_TZ)

    market_open_today = datetime.combine(et_now.date(), time(hour=9, minute=30), tzinfo=_EST_TZ)
    wake_target = market_open_today - timedelta(seconds=_PRE_OPEN_WAKE_BUFFER_SECONDS)

    if et_now < wake_target:
        return max(1.0, (wake_target - et_now).total_seconds())

    next_day = et_now.date() + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    next_open = datetime.combine(next_day, time(hour=9, minute=30), tzinfo=_EST_TZ)
    next_wake = next_open - timedelta(seconds=_PRE_OPEN_WAKE_BUFFER_SECONDS)
    return max(1.0, (next_wake - et_now).total_seconds())


def _require_bot(bot_id: str) -> TradingBot:
    bot = bots.get(bot_id)
    if bot is None:
        raise KeyError(f"Bot '{bot_id}' not found")
    return bot


def persist_bot(bot: TradingBot) -> None:
    """Persist the current bot state."""
    _save_bot_state(bot)


def clear_persisted_bots() -> None:
    """Delete all persisted paper-trading bots."""
    _ensure_bots_table()
    with sqlite3.connect(db_path()) as conn:
        conn.execute("DELETE FROM paper_trading_bots")


def _ensure_bots_table() -> None:
    ensure_db()
    with sqlite3.connect(db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trading_bots (
                id TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


def _save_bot_state(bot: TradingBot) -> None:
    _ensure_bots_table()
    now_iso = datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()
    payload_json = json.dumps(_serialize_bot(bot))
    with sqlite3.connect(db_path()) as conn:
        conn.execute(
            """
            INSERT INTO paper_trading_bots (id, payload_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET payload_json = excluded.payload_json, updated_at = excluded.updated_at
            """,
            (bot.id, payload_json, now_iso),
        )


def _delete_persisted_bot(bot_id: str) -> None:
    _ensure_bots_table()
    with sqlite3.connect(db_path()) as conn:
        conn.execute("DELETE FROM paper_trading_bots WHERE id = ?", (bot_id,))


def _serialize_bot(bot: TradingBot) -> dict[str, Any]:
    engine = bot.execution_engine
    return {
        "id": bot.id,
        "name": bot.name,
        "model_name": bot.model_name,
        "ticker": bot.ticker,
        "timeframe": bot.timeframe,
        "cash": float(bot.cash),
        "position": float(bot.position),
        "avg_entry_price": float(bot.avg_entry_price),
        "total_pnl": float(bot.total_pnl),
        "day_pnl": float(bot.day_pnl),
        "status": bot.status,
        "buy_threshold": float(bot.buy_threshold),
        "sell_threshold": float(bot.sell_threshold),
        "long_only": bool(bot.long_only),
        "mode": SPOT_MODE if bool(bot.long_only) else OPTIONS_MODE,
        "daily_buy_timing": str(bot.daily_buy_timing),
        "intraday_trade_interval": str(getattr(bot, "intraday_trade_interval", "unlimited")),
        "stop_loss": float(bot.stop_loss),
        "take_profit": float(bot.take_profit),
        "prediction_horizon": int(getattr(bot, "prediction_horizon", 5)),
        "trade_size": float(bot.trade_size),
        "trades": list(bot.trades),
        "realized_pnl": float(bot.realized_pnl),
        "position_size": float(bot.position_size),
        "average_entry_price": float(bot.average_entry_price),
        "last_polled_bid": bot.last_polled_bid,
        "last_polled_ask": bot.last_polled_ask,
        "last_polled_spread": bot.last_polled_spread,
        "last_polled_timestamp": bot.last_polled_timestamp,
        "last_p_up": float(getattr(bot, "last_p_up", 0.5)),
        "execution_settings": {
            "enable_slippage": bool(engine.enable_slippage),
            "max_slippage_pct": float(engine.max_slippage_pct),
            "enable_latency_simulation": bool(engine.enable_latency_simulation),
            "min_latency_ms": float(engine.min_latency_ms),
            "max_latency_ms": float(engine.max_latency_ms),
            "enable_spread_widening": bool(engine.enable_spread_widening),
            "volatility_threshold": float(engine.volatility_threshold),
            "spread_widening_factor": float(engine.spread_widening_factor),
        },
    }


def _load_persisted_bots() -> None:
    _ensure_bots_table()
    with sqlite3.connect(db_path()) as conn:
        rows = conn.execute("SELECT payload_json FROM paper_trading_bots ORDER BY updated_at ASC").fetchall()

    for row in rows:
        try:
            payload = json.loads(str(row[0]))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        try:
            _create_bot_from_payload(payload)
        except Exception:
            continue
    _resume_running_bots()


def _resume_running_bots() -> None:
    for bot in list(get_all_bots()):
        if str(getattr(bot, "status", "")).strip().lower() != "running":
            continue
        try:
            start_bot(bot.id)
        except Exception:
            _LOGGER.exception("Failed to resume running bot %s during startup.", bot.id)


def _create_bot_from_payload(payload: dict[str, Any]) -> None:
    bot = create_bot(
        {
            "id": str(payload.get("id")),
            "name": str(payload.get("name", "Paper Bot")),
            "model_name": str(payload.get("model_name", "demo-model")),
            "ticker": str(payload.get("ticker", "AAPL")),
            "timeframe": str(payload.get("timeframe", "1m")),
            "cash": float(payload.get("cash", 0.0)),
            "position": float(payload.get("position", 0.0)),
            "avg_entry_price": float(payload.get("avg_entry_price", 0.0)),
            "total_pnl": float(payload.get("total_pnl", 0.0)),
            "day_pnl": float(payload.get("day_pnl", 0.0)),
            "status": str(payload.get("status", "stopped")),
            "buy_threshold": float(payload.get("buy_threshold", 0.6)),
            "sell_threshold": float(payload.get("sell_threshold", 0.4)),
            "long_only": bool(payload.get("long_only", False)),
            "mode": str(payload.get("mode", SPOT_MODE if bool(payload.get("long_only", False)) else OPTIONS_MODE)),
            "daily_buy_timing": str(payload.get("daily_buy_timing", "start_of_day")),
            "intraday_trade_interval": str(payload.get("intraday_trade_interval", "unlimited")),
            "stop_loss": float(payload.get("stop_loss", 0.02)),
            "take_profit": float(payload.get("take_profit", 0.04)),
            "prediction_horizon": int(payload.get("prediction_horizon", 5)),
            "trade_size": float(payload.get("trade_size", 1.0)),
            "execution_settings": payload.get("execution_settings", {}),
        },
        persist=False,
    )
    trades = payload.get("trades", [])
    if isinstance(trades, list):
        bot.trades = [trade for trade in trades if isinstance(trade, dict)]
    bot.realized_pnl = float(payload.get("realized_pnl", bot.realized_pnl))
    bot.position_size = float(payload.get("position_size", bot.position))
    bot.average_entry_price = float(payload.get("average_entry_price", bot.avg_entry_price))
    raw_bid = payload.get("last_polled_bid")
    raw_ask = payload.get("last_polled_ask")
    raw_spread = payload.get("last_polled_spread")
    bot.last_polled_bid = float(raw_bid) if raw_bid is not None else None
    bot.last_polled_ask = float(raw_ask) if raw_ask is not None else None
    bot.last_polled_spread = float(raw_spread) if raw_spread is not None else None
    raw_ts = payload.get("last_polled_timestamp")
    bot.last_polled_timestamp = str(raw_ts) if raw_ts is not None else None
    bot.last_p_up = float(payload.get("last_p_up", getattr(bot, "last_p_up", 0.5)))
    bot._sync_public_state()


_load_persisted_bots()
