from __future__ import annotations

from pathlib import Path

from quant.live_trading import auth


def test_load_env_tokens_from_example_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QUESTRADE_REFRESH_TOKEN", raising=False)
    Path(".env.example").write_text("QUESTRADE_REFRESH_TOKEN=test_refresh_token\n", encoding="utf-8")

    auth._load_env_tokens_from_files()

    assert auth.os.environ.get("QUESTRADE_REFRESH_TOKEN") == "test_refresh_token"


def test_resolve_refresh_token_strips_escaped_newline(monkeypatch):
    monkeypatch.setenv("QUESTRADE_REFRESH_TOKEN", "test_refresh_token\\n")
    client = auth.QuestradeAuthClient()

    assert client._resolve_refresh_token() == "test_refresh_token"


def test_resolve_refresh_token_accepts_full_assignment_value(monkeypatch):
    monkeypatch.setenv("QUESTRADE_REFRESH_TOKEN", "QUESTRADE_REFRESH_TOKEN=test_refresh_token")
    client = auth.QuestradeAuthClient()

    assert client._resolve_refresh_token() == "test_refresh_token"


def test_resolve_refresh_token_prefers_env_over_persisted_state(monkeypatch):
    monkeypatch.setenv("QUESTRADE_REFRESH_TOKEN", "env_refresh")
    client = auth.QuestradeAuthClient()
    client._token_state = auth.TokenState(
        access_token="access",
        refresh_token="stale_state_refresh",
        api_server="https://api01.iq.questrade.com",
        expires_at=0.0,
    )

    assert client._resolve_refresh_token() == "env_refresh"


def test_load_env_tokens_from_project_root_when_cwd_differs(tmp_path, monkeypatch):
    fake_project_root = tmp_path / "repo_root"
    fake_project_root.mkdir()
    (fake_project_root / ".env").write_text("QUESTRADE_REFRESH_TOKEN=project_root_token\n", encoding="utf-8")

    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    monkeypatch.delenv("QUESTRADE_REFRESH_TOKEN", raising=False)
    monkeypatch.setattr(auth, "_PROJECT_ROOT", fake_project_root)

    auth._load_env_tokens_from_files()

    assert auth.os.environ.get("QUESTRADE_REFRESH_TOKEN") == "project_root_token"
