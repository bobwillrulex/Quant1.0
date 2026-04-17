from __future__ import annotations

import csv
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

from .discord_notify import send_discord_webhook
from .storage import load_vwap_scan_universe, save_vwap_scan_universe

UNIVERSE_SOURCE_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote?symbols="


@dataclass
class VwapScanResult:
    symbol: str
    price: float
    vwap: float
    upper_1: float
    upper_2: float
    candle_open: float
    candle_close: float


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fetch_sp500_symbols(timeout_seconds: float = 8.0) -> list[str]:
    req = urlopen(UNIVERSE_SOURCE_URL, timeout=timeout_seconds)
    payload = req.read().decode("utf-8")
    reader = csv.DictReader(StringIO(payload))
    symbols: list[str] = []
    for row in reader:
        symbol = str(row.get("Symbol", "")).strip().upper().replace(".", "-")
        if symbol:
            symbols.append(symbol)
    return sorted(set(symbols))


def _chunks(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _fetch_market_caps(symbols: list[str], timeout_seconds: float = 8.0) -> dict[str, float]:
    market_caps: dict[str, float] = {}
    if not symbols:
        return market_caps

    for symbol_batch in _chunks(symbols, 150):
        joined = ",".join(symbol_batch)
        if not joined:
            continue
        url = f"{YAHOO_QUOTE_URL}{quote(joined)}"
        try:
            req = urlopen(url, timeout=timeout_seconds)
            payload = req.read().decode("utf-8")
            parsed = json.loads(payload)
            results = ((parsed.get("quoteResponse") or {}).get("result")) or []
        except Exception:
            continue

        for row in results:
            symbol = str(row.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            cap = _safe_float(row.get("marketCap"))
            market_caps[symbol] = cap if cap > 0 else 0.0

    return market_caps


def _sort_rows_by_market_cap(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: (-_safe_float(row.get("market_cap")), str(row.get("symbol", ""))))


def ensure_universe_symbols() -> list[str]:
    existing = load_vwap_scan_universe()
    if existing:
        symbols = sorted(set(existing + ["SPY"]))
        if symbols != existing:
            save_vwap_scan_universe(symbols, source="local_cache")
        return symbols

    fetched = _fetch_sp500_symbols()
    symbols = sorted(set(fetched + ["SPY"]))
    save_vwap_scan_universe(symbols, source=f"{UNIVERSE_SOURCE_URL} @ {_now_iso()}")
    return symbols


def _session_vwap_and_std(candles: list[dict[str, Any]]) -> tuple[float, float, float, float, float]:
    cumulative_weighted_price = 0.0
    cumulative_volume = 0.0
    prices: list[float] = []
    for candle in candles:
        high = _safe_float(candle.get("high"))
        low = _safe_float(candle.get("low"))
        close = _safe_float(candle.get("close"))
        volume = max(1.0, _safe_float(candle.get("volume")))
        hlc3 = (high + low + close) / 3.0
        prices.append(hlc3)
        cumulative_weighted_price += hlc3 * volume
        cumulative_volume += volume

    if not prices or cumulative_volume <= 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)

    vwap = cumulative_weighted_price / cumulative_volume
    mean_price = sum(prices) / len(prices)
    variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
    std_1 = variance ** 0.5
    upper_1 = vwap + std_1
    upper_2 = vwap + (2.0 * std_1)
    last_close = _safe_float(candles[-1].get("close"))
    return (vwap, std_1, upper_1, upper_2, last_close)


def _symbol_scan(client: Any, symbol: str) -> tuple[bool, bool, bool, VwapScanResult | None]:
    candles = client.get_candles(symbol, "FiveMinutes")
    if len(candles) < 2:
        return (False, False, False, None)

    session_candles = candles[-78:]
    vwap, std_1, upper_1, upper_2, last_close = _session_vwap_and_std(session_candles)
    if last_close <= 0.0 or std_1 <= 0.0:
        return (False, False, False, None)

    prev = session_candles[-2]
    curr = session_candles[-1]
    prev_open = _safe_float(prev.get("open"))
    prev_close = _safe_float(prev.get("close"))
    curr_open = _safe_float(curr.get("open"))
    curr_close = _safe_float(curr.get("close"))

    band_width_pct = ((upper_2 - upper_1) / last_close) * 100.0
    first = band_width_pct > 0.5
    second = (last_close >= upper_1) and (last_close <= upper_2)
    prev_in_band = (prev_open >= upper_1 and prev_open <= upper_2) and (prev_close >= upper_1 and prev_close <= upper_2)
    curr_in_band = (curr_open >= upper_1 and curr_open <= upper_2) and (curr_close >= upper_1 and curr_close <= upper_2)
    third = (prev_close < prev_open) and (curr_close > curr_open) and prev_in_band and curr_in_band

    result = VwapScanResult(
        symbol=symbol,
        price=last_close,
        vwap=vwap,
        upper_1=upper_1,
        upper_2=upper_2,
        candle_open=curr_open,
        candle_close=curr_close,
    )
    return (first, second, third, result)


class VwapScannerService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last_refresh_unix = 0.0
        self._market_caps: dict[str, float] = {}
        self._market_caps_last_refresh_unix = 0.0
        self._last_discord_push_unix = 0.0
        self._cache: dict[str, Any] = {
            "timestamp": "",
            "universe_size": 0,
            "errors": [],
            "column_one": [],
            "column_two": [],
            "column_three": [],
        }

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            if (time.time() - self._last_refresh_unix) < 60 and self._cache.get("timestamp"):
                return dict(self._cache)

        snapshot = self._scan_now()
        with self._lock:
            self._cache = snapshot
            self._last_refresh_unix = time.time()
            return dict(self._cache)

    def _scan_now(self) -> dict[str, Any]:
        symbols = ensure_universe_symbols()
        market_caps = self._get_market_caps(symbols)
        refresh_token = str(os.environ.get("QUESTRADE_REFRESH_TOKEN", "")).strip()
        if not refresh_token:
            return {
                "timestamp": _now_iso(),
                "universe_size": len(symbols),
                "errors": ["Missing QUESTRADE_REFRESH_TOKEN."],
                "column_one": [],
                "column_two": [],
                "column_three": [],
            }

        try:
            from questrade_client import QuestradeClient

            client = QuestradeClient(refresh_token=refresh_token)
        except Exception as exc:
            return {
                "timestamp": _now_iso(),
                "universe_size": len(symbols),
                "errors": [f"Failed to initialize Questrade client: {exc}"],
                "column_one": [],
                "column_two": [],
                "column_three": [],
            }

        col_one: list[dict[str, Any]] = []
        col_two: list[dict[str, Any]] = []
        col_three: list[dict[str, Any]] = []
        errors: list[str] = []

        for symbol in symbols:
            try:
                first, second, third, result = _symbol_scan(client, symbol)
            except (ValueError, URLError, RuntimeError, OSError) as exc:
                errors.append(f"{symbol}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"{symbol}: {exc}")
                continue
            if result is None:
                continue

            payload = {
                "symbol": result.symbol,
                "price": round(result.price, 4),
                "market_cap": round(market_caps.get(result.symbol, 0.0), 2),
                "vwap": round(result.vwap, 4),
                "upper_1": round(result.upper_1, 4),
                "upper_2": round(result.upper_2, 4),
                "candle_open": round(result.candle_open, 4),
                "candle_close": round(result.candle_close, 4),
            }
            if first:
                col_one.append(payload)
            if second:
                col_two.append(payload)
            if third:
                col_three.append(payload)

        _sort_rows_by_market_cap(col_one)
        _sort_rows_by_market_cap(col_two)
        _sort_rows_by_market_cap(col_three)

        snapshot = {
            "timestamp": _now_iso(),
            "universe_size": len(symbols),
            "errors": errors[:25],
            "column_one": col_one,
            "column_two": col_two,
            "column_three": col_three,
        }
        self._maybe_push_discord_updates(snapshot)
        return snapshot

    def _get_market_caps(self, symbols: list[str]) -> dict[str, float]:
        with self._lock:
            has_cache = bool(self._market_caps)
            stale = (time.time() - self._market_caps_last_refresh_unix) >= (6 * 60 * 60)
            if has_cache and not stale:
                return dict(self._market_caps)
        fresh = _fetch_market_caps(symbols)
        if fresh:
            with self._lock:
                self._market_caps = dict(fresh)
                self._market_caps_last_refresh_unix = time.time()
                return dict(self._market_caps)
        with self._lock:
            return dict(self._market_caps)

    def _maybe_push_discord_updates(self, snapshot: dict[str, Any]) -> None:
        now = time.time()
        with self._lock:
            if (now - self._last_discord_push_unix) < 600:
                return
            self._last_discord_push_unix = now

        webhook_one = str(os.environ.get("VWAP_DISCORD_WEBHOOK_COLUMN_ONE", "")).strip()
        webhook_two = str(os.environ.get("VWAP_DISCORD_WEBHOOK_COLUMN_TWO", "")).strip()
        webhook_three = str(os.environ.get("VWAP_DISCORD_WEBHOOK_COLUMN_THREE", "")).strip()

        self._send_top_10(webhook_one, "Column 1 • Band Width > 0.5% of price", snapshot.get("column_one") or [])
        self._send_top_10(webhook_two, "Column 2 • Price between Upper 1σ and Upper 2σ", snapshot.get("column_two") or [])
        self._send_top_10(
            webhook_three,
            "Column 3 • Red candle then Green candle (Upper 1σ..2σ)",
            snapshot.get("column_three") or [],
        )

    def _send_top_10(self, webhook_url: str, title: str, rows: list[dict[str, Any]]) -> None:
        if not webhook_url:
            return
        top_rows = rows[:10]
        lines = [f"**{title}**", f"Top 10 by market cap • {_now_iso()}"]
        if not top_rows:
            lines.append("No symbols matched this scan.")
        else:
            for idx, row in enumerate(top_rows, start=1):
                symbol = str(row.get("symbol", ""))
                price = _safe_float(row.get("price"))
                market_cap = _safe_float(row.get("market_cap"))
                lines.append(f"{idx}. {symbol} | price=${price:.2f} | market cap=${market_cap:,.0f}")
        try:
            send_discord_webhook(webhook_url, "\n".join(lines))
        except Exception:
            return


VWAP_SCANNER_SERVICE = VwapScannerService()
