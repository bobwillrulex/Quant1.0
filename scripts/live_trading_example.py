"""Example script for live/paper trading workflow with Questrade."""

from __future__ import annotations

from quant.live_trading import QuestradeAuthClient, get_quote, place_order


def main() -> None:
    auth = QuestradeAuthClient()
    auth.refresh_access_token()

    symbol = "AAPL"
    account_id = "YOUR_ACCOUNT_ID"

    quote = get_quote(symbol, auth_client=auth)
    print("AAPL quote:", quote)

    buy_result = place_order(
        {
            "accountId": account_id,
            "symbol": symbol,
            "quantity": 1,
            "action": "Buy",
            "orderType": "Market",
            "isPaper": False,
        },
        auth_client=auth,
    )
    print("Buy result:", buy_result)

    sell_result = place_order(
        {
            "accountId": account_id,
            "symbol": symbol,
            "quantity": 1,
            "action": "Sell",
            "orderType": "Market",
            "isPaper": False,
        },
        auth_client=auth,
    )
    print("Sell result:", sell_result)


if __name__ == "__main__":
    main()
