from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol
from zoneinfo import ZoneInfo

from quant.execution_engine import ExecutionEngine


class SupportsPredict(Protocol):
    def predict(self, data: dict[str, Any]) -> float: ...


@dataclass
class TradingBot:
    """Reusable trading bot with model-driven entries and SL/TP risk controls."""

    id: str
    name: str
    model_name: str
    ticker: str
    timeframe: str
    cash: float
    position: float = 0.0
    avg_entry_price: float = 0.0
    total_pnl: float = 0.0
    day_pnl: float = 0.0
    status: str = "stopped"
    buy_threshold: float = 0.6
    sell_threshold: float = 0.4
    stop_loss: float = 0.02
    take_profit: float = 0.04
    trade_size: float = 1.0
    long_only: bool = False
    daily_buy_timing: str = "start_of_day"
    model: SupportsPredict | Callable[[dict[str, Any]], float] | None = None
    execution_engine: ExecutionEngine = field(default_factory=ExecutionEngine)

    # Engine-compatible state
    position_size: float = field(init=False, default=0.0)
    average_entry_price: float = field(init=False, default=0.0)
    realized_pnl: float = field(init=False, default=0.0)
    trades: list[dict[str, Any]] = field(init=False, default_factory=list)

    _day_start_total_pnl: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.position_size = float(self.position)
        self.average_entry_price = float(self.avg_entry_price)
        self._sync_public_state()
        self._day_start_total_pnl = float(self.total_pnl)

    def on_new_candle(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process a new candle/quote, place trades, and refresh PnL."""
        if self.status != "running":
            return {"status": self.status, "action": "HOLD", "trade": None}

        quote = self._as_quote(data)

        # Risk-first: SL/TP exits take precedence over model entries.
        risk_trade = self._apply_risk_controls(quote)
        if risk_trade is not None:
            self.update_pnl(quote)
            return {"status": self.status, "action": "RISK_EXIT", "trade": risk_trade}

        signal = self._predict_signal(data)
        action = self.decide_action(signal)
        if action == "BUY" and not self._allow_buy_now(data):
            action = "HOLD"
        trade = self._execute_action(action, quote)
        self.update_pnl(quote)
        return {"status": self.status, "action": action, "trade": trade, "signal": signal}

    def decide_action(self, signal: float) -> str:
        """Map model signal to BUY/SELL/HOLD decisions."""
        if signal >= self.buy_threshold:
            return "BUY"
        if signal <= self.sell_threshold:
            if self.long_only and self.position_size <= 0:
                return "HOLD"
            return "SELL"
        return "HOLD"

    def update_pnl(self, current_quote: dict[str, Any]) -> float:
        """Continuously update total and day PnL using realized + unrealized."""
        mark = self._mark_price(current_quote)
        unrealized_pnl = 0.0
        if self.position_size != 0 and self.average_entry_price > 0:
            unrealized_pnl = (mark - self.average_entry_price) * self.position_size

        self.total_pnl = float(self.realized_pnl) + unrealized_pnl
        self.day_pnl = self.total_pnl - self._day_start_total_pnl
        self._sync_public_state()
        return self.total_pnl

    def reset_day_pnl(self) -> None:
        """Reset day baseline for day PnL calculation."""
        self._day_start_total_pnl = float(self.total_pnl)
        self.day_pnl = 0.0

    def _predict_signal(self, data: dict[str, Any]) -> float:
        if self.model is None:
            return 0.5

        if hasattr(self.model, "predict"):
            raw_signal = self.model.predict(data)
        else:
            raw_signal = self.model(data)

        return self._normalize_signal(raw_signal)

    def _normalize_signal(self, signal: Any) -> float:
        if isinstance(signal, str):
            token = signal.strip().upper()
            if token in {"BUY", "LONG"}:
                return 1.0
            if token in {"SELL", "SHORT"}:
                return 0.0
            return 0.5
        return float(signal)

    def _execute_action(self, action: str, quote: dict[str, Any]) -> dict[str, Any] | None:
        if action == "BUY":
            trade = self.execution_engine.execute_market_buy(self, quote)
            self._sync_public_state()
            return trade
        if action == "SELL":
            trade = self.execution_engine.execute_market_sell(self, quote)
            self._sync_public_state()
            return trade
        return None

    def _apply_risk_controls(self, quote: dict[str, Any]) -> dict[str, Any] | None:
        if self.position_size == 0 or self.average_entry_price <= 0:
            return None

        mark = self._mark_price(quote)
        entry = self.average_entry_price

        if self.position_size > 0:
            stop_hit = mark <= entry * (1.0 - self.stop_loss)
            target_hit = mark >= entry * (1.0 + self.take_profit)
            if stop_hit or target_hit:
                return self.execution_engine.execute_market_sell(self, quote)
            return None

        stop_hit = mark >= entry * (1.0 + self.stop_loss)
        target_hit = mark <= entry * (1.0 - self.take_profit)
        if stop_hit or target_hit:
            return self.execution_engine.execute_market_buy(self, quote)
        return None

    @staticmethod
    def _as_quote(data: dict[str, Any]) -> dict[str, Any]:
        if "bid" in data and "ask" in data:
            return data

        close = float(data.get("close") or data.get("price") or data.get("last"))
        spread = float(data.get("spread", 0.0))
        half_spread = spread / 2.0
        return {
            "bid": close - half_spread,
            "ask": close + half_spread,
            "timestamp": data.get("timestamp"),
        }

    @staticmethod
    def _mark_price(quote: dict[str, Any]) -> float:
        bid = quote.get("bid")
        ask = quote.get("ask")
        if bid is not None and ask is not None:
            return (float(bid) + float(ask)) / 2.0
        if "close" in quote:
            return float(quote["close"])
        if "last" in quote:
            return float(quote["last"])
        raise KeyError("Quote must include bid/ask, close, or last.")

    def _sync_public_state(self) -> None:
        self.position = float(self.position_size)
        self.avg_entry_price = float(self.average_entry_price)

    def _allow_buy_now(self, data: dict[str, Any]) -> bool:
        timeframe = self.timeframe.strip().lower()
        if timeframe not in {"1d", "d", "day", "daily"}:
            return True
        timing = self.daily_buy_timing.strip().lower()
        if timing not in {"start_of_day", "end_of_day"}:
            timing = "start_of_day"
        timestamp = data.get("timestamp")
        if timestamp is None:
            return timing == "start_of_day"
        parsed = self._parse_timestamp(timestamp)
        if parsed is None:
            return timing == "start_of_day"
        et = parsed.astimezone(ZoneInfo("America/New_York"))
        minutes = (et.hour * 60) + et.minute
        if timing == "end_of_day":
            return minutes >= (15 * 60 + 59)
        return 9 * 60 + 30 <= minutes <= 9 * 60 + 31

    @staticmethod
    def _parse_timestamp(timestamp: Any) -> datetime | None:
        text = str(timestamp or "").strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed
