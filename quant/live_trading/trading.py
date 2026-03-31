from __future__ import annotations

import logging
import os
from typing import Any, Literal, TypedDict

from .auth import QuestradeAuthClient

logger = logging.getLogger(__name__)


class OrderRequest(TypedDict, total=False):
    accountId: str
    symbol: str
    quantity: int
    action: Literal["Buy", "Sell"]
    orderType: Literal["Market", "Limit"]
    limitPrice: float
    isPaper: bool


def _is_live_enabled() -> bool:
    return os.environ.get("ENABLE_LIVE_TRADING", "false").strip().lower() == "true"


def _validate_order_input(order: OrderRequest) -> None:
    if not str(order.get("accountId", "")).strip():
        raise ValueError("accountId is required.")
    symbol = str(order.get("symbol", "")).strip().upper()
    if not symbol or not symbol.isalnum():
        raise ValueError("symbol must be alphanumeric and non-empty.")

    quantity = int(order.get("quantity", 0))
    if quantity <= 0:
        raise ValueError("quantity must be positive.")

    action = order.get("action")
    if action not in {"Buy", "Sell"}:
        raise ValueError("action must be Buy or Sell.")

    order_type = order.get("orderType")
    if order_type not in {"Market", "Limit"}:
        raise ValueError("orderType must be Market or Limit.")

    if order_type == "Limit":
        limit_price = float(order.get("limitPrice", 0.0))
        if limit_price <= 0:
            raise ValueError("limitPrice must be positive for limit orders.")


def place_order(order: OrderRequest, *, auth_client: QuestradeAuthClient | None = None) -> dict[str, Any]:
    """Place market or limit order with paper/live safety gates."""
    _validate_order_input(order)

    if order.get("isPaper", True):
        return {"status": "paper", "message": "Paper trading mode enabled. No real order sent.", "order": dict(order)}

    if not _is_live_enabled():
        raise RuntimeError("Live order blocked. Set ENABLE_LIVE_TRADING=true to place real trades.")

    client = auth_client or QuestradeAuthClient()
    order_type = str(order["orderType"])
    payload: dict[str, Any] = {
        "symbolId": None,
        "quantity": int(order["quantity"]),
        "isAllOrNone": False,
        "isAnonymous": False,
        "action": order["action"].upper(),
        "orderType": "Market" if order_type == "Market" else "Limit",
        "timeInForce": "Day",
    }

    symbol_search = client.authorized_request("GET", "/v1/symbols/search", query={"prefix": str(order["symbol"]).upper()})
    matches = symbol_search.get("symbols", [])
    symbol_id = next((item.get("symbolId") for item in matches if str(item.get("symbol", "")).upper() == str(order["symbol"]).upper()), None)
    if symbol_id is None:
        raise ValueError(f"Unknown symbol: {order['symbol']}")
    payload["symbolId"] = int(symbol_id)

    if order_type == "Limit":
        payload["limitPrice"] = float(order["limitPrice"])

    try:
        response = client.authorized_request("POST", f"/v1/accounts/{order['accountId']}/orders", body=payload)
        return response
    except Exception as exc:
        logger.exception("Failed to place order")
        raise RuntimeError(f"Failed to place trade: {exc}") from exc


def get_order_status(account_id: str, order_id: str, *, auth_client: QuestradeAuthClient | None = None) -> dict[str, Any]:
    client = auth_client or QuestradeAuthClient()
    return client.authorized_request("GET", f"/v1/accounts/{account_id}/orders/{order_id}")


def cancel_order(account_id: str, order_id: str, *, auth_client: QuestradeAuthClient | None = None) -> dict[str, Any]:
    client = auth_client or QuestradeAuthClient()
    return client.authorized_request("DELETE", f"/v1/accounts/{account_id}/orders/{order_id}")


def get_order_history(account_id: str, *, auth_client: QuestradeAuthClient | None = None) -> list[dict[str, Any]]:
    client = auth_client or QuestradeAuthClient()
    response = client.authorized_request("GET", f"/v1/accounts/{account_id}/orders")
    return list(response.get("orders", []))
