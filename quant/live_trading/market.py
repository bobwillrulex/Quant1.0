from __future__ import annotations

from typing import Any

from .auth import QuestradeAuthClient


def get_quote(symbol: str, *, auth_client: QuestradeAuthClient | None = None) -> dict[str, Any]:
    """Fetch a quote snapshot for one symbol."""
    clean_symbol = symbol.strip().upper()
    if not clean_symbol:
        raise ValueError("Symbol is required.")

    client = auth_client or QuestradeAuthClient()
    payload = client.authorized_request("GET", "/v1/markets/quotes", query={"symbols": clean_symbol})
    quotes = payload.get("quotes", [])
    if not quotes:
        raise ValueError(f"No quote data returned for symbol '{clean_symbol}'.")
    return quotes[0]


def get_candles(symbol: str, interval: str, *, auth_client: QuestradeAuthClient | None = None) -> list[dict[str, Any]]:
    """Fetch historical OHLCV candles for a symbol."""
    clean_symbol = symbol.strip().upper()
    clean_interval = interval.strip().capitalize()
    if not clean_symbol:
        raise ValueError("Symbol is required.")
    if clean_interval not in {"OneMinute", "FiveMinutes", "FifteenMinutes", "ThirtyMinutes", "OneHour", "OneDay", "OneWeek"}:
        raise ValueError("Unsupported interval for Questrade candles endpoint.")

    client = auth_client or QuestradeAuthClient()
    lookup = client.authorized_request("GET", "/v1/symbols/search", query={"prefix": clean_symbol})
    symbols = lookup.get("symbols", [])
    symbol_id = next((item.get("symbolId") for item in symbols if str(item.get("symbol", "")).upper() == clean_symbol), None)
    if symbol_id is None:
        raise ValueError(f"Could not resolve symbol id for '{clean_symbol}'.")

    candle_response = client.authorized_request(
        "GET",
        f"/v1/markets/candles/{symbol_id}",
        query={"interval": clean_interval},
    )
    return list(candle_response.get("candles", []))
