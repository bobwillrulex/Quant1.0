from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from bot import TradingBot
from bot_manager import is_market_open


QuoteFetcher = Callable[[TradingBot], dict[str, Any] | None]
CandleFetcher = Callable[[TradingBot], dict[str, Any] | None]
BotsProvider = Callable[[], Iterable[TradingBot]]


@dataclass
class MarketLoopConfig:
    """Runtime controls for the scheduler loop."""

    poll_interval_seconds: float = 0.25
    closed_poll_interval_seconds: float = 5.0
    min_quote_interval_seconds: float = 0.25


@dataclass
class _BotRuntimeState:
    last_quote_at: float = 0.0
    last_candle_bucket: int | None = None


class MarketLoopScheduler:
    """Central scheduler for live bot processing.

    Responsibilities:
    - runs continuously in one background loop;
    - updates every running bot's mark-to-market PnL with latest bid/ask;
    - calls ``bot.on_new_candle`` only when a fresh candle is due;
    - enforces market hours by idling decision logic while closed.
    """

    def __init__(
        self,
        *,
        bots_provider: BotsProvider,
        quote_fetcher: QuoteFetcher,
        candle_fetcher: CandleFetcher | None = None,
        config: MarketLoopConfig | None = None,
    ) -> None:
        self._bots_provider = bots_provider
        self._quote_fetcher = quote_fetcher
        self._candle_fetcher = candle_fetcher
        self._config = config or MarketLoopConfig()

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state: dict[str, _BotRuntimeState] = {}

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_forever, name="market-loop", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def run_once(self, now: datetime | None = None) -> None:
        epoch_now = time.monotonic()
        for bot in self._bots_provider():
            if bot.status != "running":
                continue
            self._tick_bot(bot=bot, now=now, epoch_now=epoch_now)

    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            now = datetime.now(tz=timezone.utc)
            self.run_once(now=now)
            is_open = is_market_open(now)
            timeout = (
                self._config.poll_interval_seconds
                if is_open
                else self._config.closed_poll_interval_seconds
            )
            self._stop_event.wait(timeout=max(0.05, timeout))

    def _tick_bot(self, *, bot: TradingBot, now: datetime | None, epoch_now: float) -> None:
        state = self._state.setdefault(bot.id, _BotRuntimeState())

        if (epoch_now - state.last_quote_at) >= self._config.min_quote_interval_seconds:
            quote = self._quote_fetcher(bot)
            state.last_quote_at = epoch_now
            if quote is not None:
                bot.update_pnl(quote)
            else:
                return
        else:
            return

        if not is_market_open(now):
            return

        if not self._is_candle_due(bot=bot, quote=quote, state=state):
            return

        candle_payload = self._fetch_candle(bot=bot)
        if candle_payload is None:
            return

        merged = dict(candle_payload)
        merged.setdefault("bid", quote.get("bid"))
        merged.setdefault("ask", quote.get("ask"))
        merged.setdefault("timestamp", candle_payload.get("timestamp") or quote.get("timestamp"))
        bot.on_new_candle(merged)

    def _fetch_candle(self, *, bot: TradingBot) -> dict[str, Any] | None:
        if self._candle_fetcher is not None:
            return self._candle_fetcher(bot)
        per_bot = getattr(bot, "candle_fetcher", None)
        if callable(per_bot):
            return per_bot(bot)
        return None

    def _is_candle_due(self, *, bot: TradingBot, quote: dict[str, Any], state: _BotRuntimeState) -> bool:
        seconds = _timeframe_to_seconds(getattr(bot, "timeframe", "1m"))
        ts = quote.get("timestamp")
        bucket = _to_epoch_bucket(ts, seconds)
        if bucket is None:
            return True
        if state.last_candle_bucket == bucket:
            return False
        state.last_candle_bucket = bucket
        return True


def _timeframe_to_seconds(timeframe: str) -> int:
    raw = (timeframe or "1m").strip().lower()
    if raw.endswith("m"):
        return max(60, int(raw[:-1] or 1) * 60)
    if raw.endswith("h"):
        return max(3600, int(raw[:-1] or 1) * 3600)
    if raw.endswith("d"):
        return max(86400, int(raw[:-1] or 1) * 86400)
    return 60


def _to_epoch_bucket(timestamp: Any, seconds: int) -> int | None:
    if timestamp is None:
        return None
    if isinstance(timestamp, datetime):
        epoch = timestamp.timestamp()
    elif isinstance(timestamp, (int, float)):
        epoch = float(timestamp)
    else:
        try:
            epoch = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return int(epoch // seconds)
