from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

TOKEN_FILE = Path(os.environ.get("QUESTRADE_TOKEN_FILE", Path.home() / ".quant1_data" / "questrade_tokens.json"))
_ENV_FILES = (Path(".env"), Path(".env.example"))


class QuestradeApiError(RuntimeError):
    """Raised when a Questrade API call fails."""


_TERMINAL_AUTH_ERROR_MARKER = "questrade authentication failed (403/1010)"


def is_terminal_auth_failure(exc: Exception) -> bool:
    """Return True when an error requires manual Questrade re-authentication."""
    return _TERMINAL_AUTH_ERROR_MARKER in str(exc).lower()


@dataclass
class TokenState:
    access_token: str
    refresh_token: str
    api_server: str
    expires_at: float

    @property
    def is_expired(self) -> bool:
        # Refresh one minute before expiration to avoid race conditions.
        return time.time() >= (self.expires_at - 60)


class QuestradeAuthClient:
    """OAuth2 helper for Questrade API that auto-refreshes and persists tokens."""

    def __init__(self) -> None:
        self._token_state: TokenState | None = self._load_token_state()

    def get_access_token(self) -> str:
        return self._ensure_token_state().access_token

    def get_api_server(self) -> str:
        return self._ensure_token_state().api_server.rstrip("/")

    def authorized_request(self, method: str, path: str, *, query: dict[str, Any] | None = None, body: dict[str, Any] | None = None) -> dict[str, Any]:
        token_state = self._ensure_token_state()
        url = f"{token_state.api_server.rstrip('/')}/{path.lstrip('/')}"
        if query:
            url += f"?{urlencode(query, doseq=True)}"

        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")

        request = Request(url, method=method.upper(), data=payload)
        request.add_header("Authorization", f"Bearer {token_state.access_token}")
        request.add_header("Content-Type", "application/json")

        return self._request_with_retry(request=request, retry_on_auth=True)

    def _ensure_token_state(self) -> TokenState:
        state = self._token_state
        if state is None or state.is_expired:
            state = self.refresh_access_token()
            self._token_state = state
        return state

    def refresh_access_token(self) -> TokenState:
        refresh_token = self._resolve_refresh_token()
        url = f"https://login.questrade.com/oauth2/token?grant_type=refresh_token&refresh_token={refresh_token}"
        request = Request(url, method="GET")
        payload = self._request_with_retry(request=request, retry_on_auth=False)

        try:
            access_token = str(payload["access_token"])
            new_refresh_token = str(payload["refresh_token"])
            api_server = str(payload["api_server"])
            expires_in = int(payload.get("expires_in", 0))
        except (KeyError, ValueError, TypeError) as exc:
            raise QuestradeApiError(f"Malformed OAuth2 token response: {payload}") from exc

        if expires_in <= 0:
            raise QuestradeApiError("Received non-positive token expiry from Questrade OAuth2 response.")

        state = TokenState(
            access_token=access_token,
            refresh_token=new_refresh_token,
            api_server=api_server,
            expires_at=time.time() + expires_in,
        )
        self._persist_token_state(state)
        return state

    def _resolve_refresh_token(self) -> str:
        env_refresh = _clean_refresh_token_candidate(os.environ.get("QUESTRADE_REFRESH_TOKEN", ""))
        if env_refresh:
            return env_refresh
        _load_env_tokens_from_files()
        env_refresh = _clean_refresh_token_candidate(os.environ.get("QUESTRADE_REFRESH_TOKEN", ""))
        if env_refresh:
            return env_refresh
        if self._token_state and self._token_state.refresh_token:
            state_refresh = _clean_refresh_token_candidate(self._token_state.refresh_token)
            if state_refresh:
                return state_refresh
        raise QuestradeApiError("Missing Questrade refresh token. Set QUESTRADE_REFRESH_TOKEN in environment.")

    def _request_with_retry(self, *, request: Request, retry_on_auth: bool) -> dict[str, Any]:
        # Minimal retry policy for transient rate/infra failures.
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                with urlopen(request, timeout=20) as response:
                    raw = response.read().decode("utf-8")
                    parsed = json.loads(raw) if raw else {}
                    if isinstance(parsed, dict):
                        return parsed
                    raise QuestradeApiError(f"Unexpected non-object response: {parsed!r}")
            except HTTPError as exc:
                status = exc.code
                body = exc.read().decode("utf-8", errors="ignore")
                if status == 403 and "1010" in body:
                    raise QuestradeApiError(
                        "Questrade authentication failed (403/1010): refresh token is invalid or expired. "
                        "Generate a new token and update QUESTRADE_REFRESH_TOKEN."
                    ) from exc
                if retry_on_auth and status == 401 and attempt == 1:
                    # Access token expired/revoked; refresh then retry once.
                    self._token_state = self.refresh_access_token()
                    request.headers["Authorization"] = f"Bearer {self._token_state.access_token}"
                    continue
                if status in (429, 500, 502, 503, 504) and attempt < max_attempts:
                    time.sleep(0.6 * attempt)
                    continue
                raise QuestradeApiError(f"Questrade request failed [{status}]: {body}") from exc
            except URLError as exc:
                if attempt < max_attempts:
                    time.sleep(0.6 * attempt)
                    continue
                raise QuestradeApiError(f"Network error calling Questrade API: {exc}") from exc

        raise QuestradeApiError("Questrade request failed after retries.")

    def _load_token_state(self) -> TokenState | None:
        if not TOKEN_FILE.exists():
            return None
        try:
            payload = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
            return TokenState(
                access_token=str(payload["access_token"]),
                refresh_token=str(payload["refresh_token"]),
                api_server=str(payload["api_server"]),
                expires_at=float(payload["expires_at"]),
            )
        except Exception:
            return None

    def _persist_token_state(self, state: TokenState) -> None:
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(
            json.dumps(
                {
                    "access_token": state.access_token,
                    "refresh_token": state.refresh_token,
                    "api_server": state.api_server,
                    "expires_at": state.expires_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _load_env_tokens_from_files() -> None:
    for env_file in _ENV_FILES:
        if not env_file.exists():
            continue
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            clean_key = key.strip()
            if not clean_key or clean_key in os.environ:
                continue
            os.environ[clean_key] = value.strip().strip("'\"")


def _clean_refresh_token_candidate(raw_value: str | None) -> str:
    if not raw_value:
        return ""
    value = str(raw_value).strip().strip("'\"")
    prefix = "QUESTRADE_REFRESH_TOKEN="
    if value.upper().startswith(prefix):
        value = value[len(prefix) :].strip()
    value = value.replace("\r", "").replace("\n", "")
    while value.endswith("\\n") or value.endswith("\\r"):
        value = value[:-2]
    return value.strip()
