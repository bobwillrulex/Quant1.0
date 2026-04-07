from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable
from zoneinfo import ZoneInfo

from bot import TradingBot


MarketDataFetcher = Callable[[TradingBot], dict[str, Any] | None]


bots: dict[str, TradingBot] = {}


_LOCK = threading.RLock()
_BOT_THREADS: dict[str, threading.Thread] = {}
_BOT_STOP_EVENTS: dict[str, threading.Event] = {}
_DEFAULT_POLL_SECONDS = 5.0
_EST_TZ = ZoneInfo("America/New_York")


def create_bot(config: dict[str, Any]) -> TradingBot:
    """Create and register a bot from config.

    Expected config keys for ``TradingBot``:
      id, name, model_name, ticker, timeframe, cash

    Optional runtime keys:
      market_data_fetcher: Callable[[TradingBot], dict[str, Any] | None]
      poll_interval: float (seconds)
    """
    runtime_config = dict(config)
    market_data_fetcher = runtime_config.pop("market_data_fetcher", None)
    poll_interval = runtime_config.pop("poll_interval", _DEFAULT_POLL_SECONDS)

    bot = TradingBot(**runtime_config)
    if market_data_fetcher is not None:
        setattr(bot, "market_data_fetcher", market_data_fetcher)
    setattr(bot, "poll_interval", poll_interval)
    with _LOCK:
        if bot.id in bots:
            raise ValueError(f"Bot with id '{bot.id}' already exists")
        bots[bot.id] = bot
    return bot


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
