from __future__ import annotations

import os
from pathlib import Path


def load_local_env_files(*, override: bool = False) -> None:
    """Load key/value pairs from local .env files into ``os.environ``.

    This is intentionally dependency-free so the app behaves the same on
    PowerShell, cmd, bash, and zsh without requiring shell-specific export
    syntax.
    """

    for env_path in _candidate_env_files():
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            if line.lower().startswith("export "):
                line = line[7:].strip()
                if "=" not in line:
                    continue

            key, value = line.split("=", 1)
            clean_key = key.strip()
            if not clean_key:
                continue

            if not override and clean_key in os.environ:
                continue

            os.environ[clean_key] = value.strip().strip("'\"")


def _candidate_env_files() -> tuple[Path, ...]:
    cwd = Path.cwd()
    return (
        cwd / ".env",
        cwd / ".env.local",
    )
