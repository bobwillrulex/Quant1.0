from __future__ import annotations

from pathlib import Path

from quant.live_trading import auth


def test_load_env_tokens_from_example_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QUESTRADE_REFRESH_TOKEN", raising=False)
    Path(".env.example").write_text("QUESTRADE_REFRESH_TOKEN=test_refresh_token\n", encoding="utf-8")

    auth._load_env_tokens_from_files()

    assert auth.os.environ.get("QUESTRADE_REFRESH_TOKEN") == "test_refresh_token"
