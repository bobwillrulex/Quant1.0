"""Thread-safe Questrade API client with automatic OAuth token refresh.

This module wraps common Questrade endpoints and provides normalized outputs
for quote and candle data.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests import Response, Session


class QuestradeError(Exception):
    """Base class for Questrade client errors."""


class QuestradeAuthError(QuestradeError):
    """Raised when authentication or token refresh fails."""


class QuestradeRequestError(QuestradeError):
    """Raised when a Questrade API request fails."""


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    last: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bid": self.bid,
            "ask": self.ask,
            "last": self.last,
            "timestamp": self.timestamp,
        }


class QuestradeClient:
    """Production-ready client for Questrade's REST API.

    Notes:
    - Uses OAuth refresh tokens to automatically renew expired access tokens.
    - Applies conservative retry/backoff for transient failures and rate limits.
    - Caches the latest quote per ticker and falls back to cache on failures.
    - Thread-safe for concurrent callers.
    """

    TOKEN_URL = "https://login.questrade.com/oauth2/token"

    def __init__(
        self,
        refresh_token: str,
        *,
        timeout: float = 10.0,
        max_retries: int = 3,
        token_refresh_buffer_seconds: int = 60,
        session: Optional[Session] = None,
        refresh_token_file: str | Path | None = None,
    ) -> None:
        if not refresh_token:
            raise ValueError("refresh_token is required")

        self._refresh_token = refresh_token
        self._timeout = timeout
        self._max_retries = max_retries
        self._token_refresh_buffer = timedelta(seconds=token_refresh_buffer_seconds)

        self._session = session or requests.Session()
        self._refresh_token_file = Path(refresh_token_file).expanduser() if refresh_token_file else None

        self._lock = threading.RLock()
        self._access_token: Optional[str] = None
        self._api_server: Optional[str] = None
        self._token_expiry: Optional[datetime] = None

        self._symbol_cache: Dict[str, int] = {}
        self._quote_cache: Dict[str, Quote] = {}


    def get_quote(self, ticker: str) -> Dict[str, Any]:
        """Fetch a quote and normalize to a clean dict.

        Returns:
            {
                "bid": float,
                "ask": float,
                "last": float,
                "timestamp": datetime
            }
        """
        ticker_key = ticker.strip().upper()
        if not ticker_key:
            raise ValueError("ticker is required")

        try:
            symbol_id = self._resolve_symbol_id(ticker_key)
            payload = self._request(
                "GET",
                f"/v1/markets/quotes/{symbol_id}",
            )

            quote_data = payload.get("quote") or payload.get("quotes", [{}])[0]
            quote = self._normalize_quote(quote_data)
            with self._lock:
                self._quote_cache[ticker_key] = quote
            return quote.to_dict()
        except Exception as exc:
            with self._lock:
                cached = self._quote_cache.get(ticker_key)
            if cached is not None:
                return cached.to_dict()
            raise QuestradeRequestError(f"Unable to fetch quote for {ticker_key}: {exc}") from exc

    def get_candles(self, ticker: str, interval: str) -> List[Dict[str, Any]]:
        """Fetch OHLCV candles and normalize output."""
        ticker_key = ticker.strip().upper()
        if not ticker_key:
            raise ValueError("ticker is required")
        if not interval or not interval.strip():
            raise ValueError("interval is required")

        symbol_id = self._resolve_symbol_id(ticker_key)
        payload = self._request(
            "GET",
            f"/v1/markets/candles/{symbol_id}",
            params={"interval": interval.strip()},
        )

        candles = payload.get("candles", [])
        normalized: List[Dict[str, Any]] = []
        for candle in candles:
            ts = self._parse_timestamp(candle.get("start") or candle.get("end"))
            normalized.append(
                {
                    "timestamp": ts,
                    "open": float(candle.get("open") or 0.0),
                    "high": float(candle.get("high") or 0.0),
                    "low": float(candle.get("low") or 0.0),
                    "close": float(candle.get("close") or 0.0),
                    "volume": float(candle.get("volume") or 0.0),
                }
            )
        return normalized

    def _resolve_symbol_id(self, ticker: str) -> int:
        with self._lock:
            cached = self._symbol_cache.get(ticker)
        if cached is not None:
            return cached

        payload = self._request("GET", "/v1/symbols/search", params={"prefix": ticker})
        symbols = payload.get("symbols", [])
        exact_match = next((s for s in symbols if str(s.get("symbol", "")).upper() == ticker), None)
        symbol = exact_match or (symbols[0] if symbols else None)
        if not symbol or "symbolId" not in symbol:
            raise QuestradeRequestError(f"Symbol not found for ticker '{ticker}'")

        symbol_id = int(symbol["symbolId"])
        with self._lock:
            self._symbol_cache[ticker] = symbol_id
        return symbol_id

    def _normalize_quote(self, quote_data: Dict[str, Any]) -> Quote:
        bid = float(quote_data.get("bidPrice") or 0.0)
        ask = float(quote_data.get("askPrice") or 0.0)
        last = float(quote_data.get("lastTradePrice") or quote_data.get("lastPrice") or 0.0)

        timestamp = self._parse_timestamp(
            quote_data.get("lastTradeTime")
            or quote_data.get("lastTradeTick")
            or quote_data.get("delay")
        )

        return Quote(bid=bid, ask=ask, last=last, timestamp=timestamp)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_valid_token()

        for attempt in range(self._max_retries + 1):
            url = f"{self._get_api_server().rstrip('/')}{path}"
            headers = {
                "Authorization": f"Bearer {self._get_access_token()}",
                "Accept": "application/json",
            }

            try:
                response = self._session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                if attempt >= self._max_retries:
                    raise QuestradeRequestError(f"Network error calling {path}: {exc}") from exc
                self._sleep_with_backoff(attempt)
                continue

            if response.status_code == 401:
                self._refresh_access_token()
                if attempt >= self._max_retries:
                    raise QuestradeAuthError("Authentication failed after token refresh")
                continue

            if response.status_code == 429:
                if attempt >= self._max_retries:
                    raise QuestradeRequestError("Rate limit exceeded and retries exhausted")
                self._handle_rate_limit(response, attempt)
                continue

            if 500 <= response.status_code < 600:
                if attempt >= self._max_retries:
                    raise QuestradeRequestError(
                        f"Server error {response.status_code}: {response.text}"
                    )
                self._sleep_with_backoff(attempt)
                continue

            if response.status_code >= 400:
                raise QuestradeRequestError(
                    f"Request failed ({response.status_code}) for {path}: {response.text}"
                )

            try:
                return response.json()
            except ValueError as exc:
                raise QuestradeRequestError(f"Invalid JSON response from {path}") from exc

        raise QuestradeRequestError(f"Failed request to {path} after retries")

    def _ensure_valid_token(self) -> None:
        with self._lock:
            expiry = self._token_expiry
            should_refresh = (
                self._access_token is None
                or self._api_server is None
                or expiry is None
                or datetime.now(timezone.utc) >= (expiry - self._token_refresh_buffer)
            )

        if should_refresh:
            self._refresh_access_token()

    def _refresh_access_token(self) -> None:
        with self._lock:
            params = {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            }

            try:
                response = self._session.get(self.TOKEN_URL, params=params, timeout=self._timeout)
            except requests.RequestException as exc:
                raise QuestradeAuthError(f"Token refresh network error: {exc}") from exc

            if response.status_code >= 400:
                raise QuestradeAuthError(
                    f"Token refresh failed ({response.status_code}): {response.text}"
                )

            data = response.json()
            access_token = data.get("access_token")
            api_server = data.get("api_server")
            expires_in = data.get("expires_in")

            if not access_token or not api_server or not expires_in:
                raise QuestradeAuthError("Token refresh response missing required fields")

            self._access_token = str(access_token)
            self._api_server = str(api_server)
            self._token_expiry = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))

            new_refresh = data.get("refresh_token")
            if new_refresh:
                self._refresh_token = str(new_refresh)
                self._persist_refresh_token()


    def _persist_refresh_token(self) -> None:
        if self._refresh_token_file is None:
            return
        self._refresh_token_file.parent.mkdir(parents=True, exist_ok=True)
        self._refresh_token_file.write_text(self._refresh_token, encoding="utf-8")

    def _get_access_token(self) -> str:
        with self._lock:
            if not self._access_token:
                raise QuestradeAuthError("No access token available")
            return self._access_token

    def _get_api_server(self) -> str:
        with self._lock:
            if not self._api_server:
                raise QuestradeAuthError("No API server available")
            return self._api_server

    def _handle_rate_limit(self, response: Response, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                delay = max(float(retry_after), 0.0)
            except ValueError:
                delay = 0.0
        else:
            delay = (2**attempt) + random.uniform(0.0, 0.25)
        time.sleep(delay)

    def _sleep_with_backoff(self, attempt: int) -> None:
        delay = (2**attempt) + random.uniform(0.0, 0.25)
        time.sleep(delay)

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        if isinstance(value, str) and value.strip():
            candidate = value.strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(candidate)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return datetime.now(timezone.utc)
