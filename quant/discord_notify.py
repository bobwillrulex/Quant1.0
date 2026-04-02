from __future__ import annotations

import json
import urllib.request
from urllib.error import HTTPError, URLError


def send_discord_webhook(webhook_url: str, content: str, timeout: float = 10.0) -> None:
    clean_url = webhook_url.strip()
    if not clean_url:
        raise ValueError("Discord webhook URL is required.")
    payload = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        clean_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Quant1.0-DiscordWebhook/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status >= 400:
                raise RuntimeError(f"Discord webhook failed with status {response.status}.")
    except HTTPError as exc:
        details = ""
        try:
            response_body = exc.read().decode("utf-8", errors="replace").strip()
            if response_body:
                details = f" Response: {response_body}"
        except Exception:
            details = ""
        raise RuntimeError(f"Discord webhook failed with status {exc.code}.{details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Discord webhook request failed: {exc.reason}.") from exc
