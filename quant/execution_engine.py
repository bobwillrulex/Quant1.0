from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any


class ExecutionEngine:
    """Simple market execution simulator with optional slippage."""

    def __init__(
        self,
        *,
        enable_slippage: bool = False,
        max_slippage_pct: float = 0.0002,
        enable_latency_simulation: bool = False,
        min_latency_ms: float = 50.0,
        max_latency_ms: float = 200.0,
        enable_spread_widening: bool = False,
        volatility_threshold: float = 0.02,
        spread_widening_factor: float = 2.0,
        rng: random.Random | None = None,
    ) -> None:
        if max_slippage_pct < 0:
            raise ValueError("max_slippage_pct must be non-negative.")
        if min_latency_ms < 0 or max_latency_ms < 0:
            raise ValueError("min_latency_ms and max_latency_ms must be non-negative.")
        if min_latency_ms > max_latency_ms:
            raise ValueError("min_latency_ms cannot be greater than max_latency_ms.")
        if volatility_threshold < 0:
            raise ValueError("volatility_threshold must be non-negative.")
        if spread_widening_factor < 1.0:
            raise ValueError("spread_widening_factor must be >= 1.0.")
        self.enable_slippage = enable_slippage
        self.max_slippage_pct = max_slippage_pct
        self.enable_latency_simulation = enable_latency_simulation
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.enable_spread_widening = enable_spread_widening
        self.volatility_threshold = volatility_threshold
        self.spread_widening_factor = spread_widening_factor
        self._rng = rng or random.Random(0)

    def execute_market_buy(self, bot: Any, quote: dict[str, Any]) -> dict[str, Any]:
        """Execute a market buy at ask with optional positive slippage."""
        self._simulate_latency()
        _, ask = self._effective_bid_ask(quote)
        size = self._get_trade_size(bot)
        fill_price = ask * (1.0 + self._slippage())
        pnl = self._apply_buy(bot, size=size, fill_price=fill_price)
        trade = self._record_trade(bot, quote=quote, side="BUY", price=fill_price, size=size, pnl=pnl)
        return trade

    def execute_market_sell(self, bot: Any, quote: dict[str, Any]) -> dict[str, Any]:
        """Execute a market sell at bid with optional negative slippage."""
        self._simulate_latency()
        bid, _ = self._effective_bid_ask(quote)
        size = self._get_trade_size(bot)
        fill_price = bid * (1.0 - self._slippage())
        pnl = self._apply_sell(bot, size=size, fill_price=fill_price)
        trade = self._record_trade(bot, quote=quote, side="SELL", price=fill_price, size=size, pnl=pnl)
        return trade

    def _slippage(self) -> float:
        if not self.enable_slippage or self.max_slippage_pct == 0:
            return 0.0
        return self._rng.uniform(0.0, self.max_slippage_pct)

    def _simulate_latency(self) -> None:
        if not self.enable_latency_simulation:
            return
        if self.max_latency_ms == 0:
            return
        latency_seconds = self._rng.uniform(self.min_latency_ms, self.max_latency_ms) / 1000.0
        if latency_seconds > 0:
            time.sleep(latency_seconds)

    def _effective_bid_ask(self, quote: dict[str, Any]) -> tuple[float, float]:
        bid = float(quote["bid"])
        ask = float(quote["ask"])
        if not self.enable_spread_widening:
            return bid, ask

        volatility = self._extract_volatility(quote)
        if volatility < self.volatility_threshold:
            return bid, ask

        spread = max(0.0, ask - bid)
        widened_spread = spread * self.spread_widening_factor
        midpoint = (bid + ask) / 2.0
        return midpoint - (widened_spread / 2.0), midpoint + (widened_spread / 2.0)

    @staticmethod
    def _extract_volatility(quote: dict[str, Any]) -> float:
        for key in ("volatility", "realized_volatility", "atr_pct", "volatility_score"):
            if key in quote:
                try:
                    return max(0.0, float(quote[key]))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    @staticmethod
    def _get_trade_size(bot: Any) -> float:
        for field in ("trade_size", "order_size", "size"):
            value = getattr(bot, field, None)
            if value is not None:
                size = float(value)
                if size <= 0:
                    raise ValueError(f"{field} must be positive.")
                return size
        raise AttributeError("Bot must define one of: trade_size, order_size, or size.")

    @staticmethod
    def _ensure_bot_state(bot: Any) -> None:
        if not hasattr(bot, "position_size"):
            bot.position_size = 0.0
        if not hasattr(bot, "average_entry_price"):
            bot.average_entry_price = 0.0
        if not hasattr(bot, "realized_pnl"):
            bot.realized_pnl = 0.0
        if not hasattr(bot, "cash"):
            bot.cash = 0.0
        if not hasattr(bot, "trades"):
            bot.trades = []

    def _apply_buy(self, bot: Any, *, size: float, fill_price: float) -> float:
        self._ensure_bot_state(bot)
        position_size = float(bot.position_size)
        avg = float(bot.average_entry_price)
        realized = 0.0

        if position_size < 0:
            cover = min(size, abs(position_size))
            realized = (avg - fill_price) * cover
            remaining_short = abs(position_size) - cover
            if remaining_short > 0:
                new_position = -remaining_short
                new_avg = avg
            else:
                new_position = size - cover
                new_avg = fill_price if new_position > 0 else 0.0
        else:
            new_position = position_size + size
            prev_cost = position_size * avg
            new_avg = ((prev_cost + (size * fill_price)) / new_position) if new_position > 0 else 0.0

        bot.position_size = new_position
        bot.average_entry_price = new_avg
        bot.realized_pnl = float(bot.realized_pnl) + realized
        bot.cash = float(bot.cash) - (size * fill_price)
        return realized

    def _apply_sell(self, bot: Any, *, size: float, fill_price: float) -> float:
        self._ensure_bot_state(bot)
        position_size = float(bot.position_size)
        avg = float(bot.average_entry_price)
        realized = 0.0

        if position_size > 0:
            close = min(size, position_size)
            realized = (fill_price - avg) * close
            remaining_long = position_size - close
            if remaining_long > 0:
                new_position = remaining_long
                new_avg = avg
            else:
                new_position = -(size - close)
                new_avg = fill_price if new_position < 0 else 0.0
        else:
            short_size = abs(position_size)
            new_position = -(short_size + size)
            prev_proceeds = short_size * avg
            new_avg = ((prev_proceeds + (size * fill_price)) / abs(new_position)) if new_position < 0 else 0.0

        bot.position_size = new_position
        bot.average_entry_price = new_avg
        bot.realized_pnl = float(bot.realized_pnl) + realized
        bot.cash = float(bot.cash) + (size * fill_price)
        return realized

    @staticmethod
    def _record_trade(bot: Any, *, quote: dict[str, Any], side: str, price: float, size: float, pnl: float) -> dict[str, Any]:
        timestamp = quote.get("timestamp") or datetime.now(timezone.utc).isoformat()
        trade = {
            "timestamp": timestamp,
            "side": side,
            "price": float(price),
            "size": float(size),
            "pnl": float(pnl),
        }
        bot.trades.append(trade)
        return trade
