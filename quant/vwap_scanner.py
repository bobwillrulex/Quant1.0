from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .storage import load_vwap_scan_universe, save_vwap_scan_universe

UNIVERSE_SOURCE_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"


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

        key = lambda row: (row["symbol"])
        col_one.sort(key=key)
        col_two.sort(key=key)
        col_three.sort(key=key)

        return {
            "timestamp": _now_iso(),
            "universe_size": len(symbols),
            "errors": errors[:25],
            "column_one": col_one,
            "column_two": col_two,
            "column_three": col_three,
        }


VWAP_SCANNER_SERVICE = VwapScannerService()
