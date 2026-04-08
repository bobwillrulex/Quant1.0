"""Quick smoke test script to fetch and print the current MSFT quote from Questrade.

Usage:
    py msft.py
    # or
    python msft.py

Required environment variable:
    QUESTRADE_REFRESH_TOKEN
"""

from __future__ import annotations

import os
import sys

from questrade_client import QuestradeClient, QuestradeError


def main() -> int:
    refresh_token = os.environ.get("QUESTRADE_REFRESH_TOKEN", "").strip()
    if not refresh_token:
        print("ERROR: QUESTRADE_REFRESH_TOKEN is not set.")
        print("Set it first, then run: py msft.py")
        return 1

    client = QuestradeClient(refresh_token=refresh_token)

    try:
        quote = client.get_quote("MSFT")
    except QuestradeError as exc:
        print(f"ERROR: Failed to fetch MSFT quote from Questrade: {exc}")
        return 1

    last = float(quote.get("last", 0.0))
    bid = float(quote.get("bid", 0.0))
    ask = float(quote.get("ask", 0.0))
    timestamp = quote.get("timestamp")

    print(f"MSFT quote OK | last={last:.4f} bid={bid:.4f} ask={ask:.4f} timestamp={timestamp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
