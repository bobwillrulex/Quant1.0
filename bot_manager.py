from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from bot import TradingBot
from quant.execution_engine import ExecutionEngine
from quant.live_trading.market import get_quote
from quant.storage import db_path, ensure_db


MarketDataFetcher = Callable[[TradingBot], dict[str, Any] | None]


bots: dict[str, TradingBot] = {}


_LOCK = threading.RLock()
_BOT_THREADS: dict[str, threading.Thread] = {}
_BOT_STOP_EVENTS: dict[str, threading.Event] = {}
_DEFAULT_POLL_SECONDS = 5.0
_EST_TZ = ZoneInfo("America/New_York")


def create_bot(config: dict[str, Any], *, persist: bool = True) -> TradingBot:
    """Create and register a bot from config.

    Expected config keys for ``TradingBot``:
      id, name, model_name, ticker, timeframe, cash

    Optional runtime keys:
      market_data_fetcher: Callable[[TradingBot], dict[str, Any] | None]
      poll_interval: float (seconds)
    """
    runtime_config = dict(config)
    market_data_fetcher = runtime_config.pop("market_data_fetcher", None)
    if market_data_fetcher is None:
        market_data_fetcher = _build_default_fetcher(runtime_config)
    poll_interval = runtime_config.pop("poll_interval", _DEFAULT_POLL_SECONDS)
    execution_settings = runtime_config.pop("execution_settings", None)
    if execution_settings is not None:
        if not isinstance(execution_settings, dict):
            raise ValueError("execution_settings must be an object.")
        runtime_config["execution_engine"] = ExecutionEngine(**execution_settings)

    bot = TradingBot(**runtime_config)
    if market_data_fetcher is not None:
        setattr(bot, "market_data_fetcher", market_data_fetcher)
    setattr(bot, "poll_interval", poll_interval)
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

    def _fetch_quote(_: TradingBot) -> dict[str, Any] | None:
        quote = get_quote(ticker)
        bid = quote.get("bidPrice", quote.get("bid"))
        ask = quote.get("askPrice", quote.get("ask"))
        if bid is None or ask is None:
            return None
        return {
            "bid": float(bid),
            "ask": float(ask),
            "timestamp": str(quote.get("lastTradeTime") or datetime.now(tz=timezone.utc).isoformat()),
            "last": float(quote.get("lastTradePrice") or quote.get("last") or (float(bid) + float(ask)) / 2.0),
        }

    return _fetch_quote


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
        if bot.status == "running" and is_market_open():
            payload = fetcher(bot)
            if payload is not None:
                bot.on_new_candle(payload)
                _save_bot_state(bot)

        stop_event.wait(timeout=poll_interval)


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
    return max(0.1, interval)


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
        "daily_buy_timing": str(bot.daily_buy_timing),
        "intraday_trade_interval": str(getattr(bot, "intraday_trade_interval", "unlimited")),
        "stop_loss": float(bot.stop_loss),
        "take_profit": float(bot.take_profit),
        "trade_size": float(bot.trade_size),
        "trades": list(bot.trades),
        "realized_pnl": float(bot.realized_pnl),
        "position_size": float(bot.position_size),
        "average_entry_price": float(bot.average_entry_price),
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
            "daily_buy_timing": str(payload.get("daily_buy_timing", "start_of_day")),
            "intraday_trade_interval": str(payload.get("intraday_trade_interval", "unlimited")),
            "stop_loss": float(payload.get("stop_loss", 0.02)),
            "take_profit": float(payload.get("take_profit", 0.04)),
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
    bot._sync_public_state()


_load_persisted_bots()
