from __future__ import annotations

from typing import Any

from .auth import QuestradeAuthClient


def get_accounts(*, auth_client: QuestradeAuthClient | None = None) -> list[dict[str, Any]]:
    client = auth_client or QuestradeAuthClient()
    response = client.authorized_request("GET", "/v1/accounts")
    return list(response.get("accounts", []))


def get_positions(account_id: str, *, auth_client: QuestradeAuthClient | None = None) -> list[dict[str, Any]]:
    clean_account_id = account_id.strip()
    if not clean_account_id:
        raise ValueError("accountId is required.")
    client = auth_client or QuestradeAuthClient()
    response = client.authorized_request("GET", f"/v1/accounts/{clean_account_id}/positions")
    return list(response.get("positions", []))


def get_balances(account_id: str, *, auth_client: QuestradeAuthClient | None = None) -> list[dict[str, Any]]:
    clean_account_id = account_id.strip()
    if not clean_account_id:
        raise ValueError("accountId is required.")
    client = auth_client or QuestradeAuthClient()
    response = client.authorized_request("GET", f"/v1/accounts/{clean_account_id}/balances")
    return list(response.get("perCurrencyBalances", []))
