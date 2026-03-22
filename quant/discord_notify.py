from __future__ import annotations

import json
import urllib.request


def send_discord_webhook(webhook_url: str, content: str, timeout: float = 10.0) -> None:
    clean_url = webhook_url.strip()
    if not clean_url:
        raise ValueError("Discord webhook URL is required.")
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        clean_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if response.status >= 400:
            raise RuntimeError(f"Discord webhook failed with status {response.status}.")
