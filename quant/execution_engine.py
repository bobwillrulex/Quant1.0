from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any


class ExecutionEngine:
    """Simple market execution simulator with optional slippage."""

    def __init__(self, *, enable_slippage: bool = False, max_slippage_pct: float = 0.0002, rng: random.Random | None = None) -> None:
        if max_slippage_pct < 0:
            raise ValueError("max_slippage_pct must be non-negative.")
        self.enable_slippage = enable_slippage
        self.max_slippage_pct = max_slippage_pct
        self._rng = rng or random.Random(0)

    def execute_market_buy(self, bot: Any, quote: dict[str, Any]) -> dict[str, Any]:
        """Execute a market buy at ask with optional positive slippage."""
        ask = float(quote["ask"])
        size = self._get_trade_size(bot)
        fill_price = ask * (1.0 + self._slippage())
        pnl = self._apply_buy(bot, size=size, fill_price=fill_price)
        trade = self._record_trade(bot, quote=quote, side="BUY", price=fill_price, size=size, pnl=pnl)
        return trade

    def execute_market_sell(self, bot: Any, quote: dict[str, Any]) -> dict[str, Any]:
        """Execute a market sell at bid with optional negative slippage."""
        bid = float(quote["bid"])
        size = self._get_trade_size(bot)
        fill_price = bid * (1.0 - self._slippage())
        pnl = self._apply_sell(bot, size=size, fill_price=fill_price)
        trade = self._record_trade(bot, quote=quote, side="SELL", price=fill_price, size=size, pnl=pnl)
        return trade

    def _slippage(self) -> float:
        if not self.enable_slippage or self.max_slippage_pct == 0:
            return 0.0
        return self._rng.uniform(0.0, self.max_slippage_pct)

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
