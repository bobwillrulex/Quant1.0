from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List

from .constants import APP_DATA_DIR, DB_FILE_NAME, LEGACY_MODEL_CONFIGS_FILE, LEGACY_MODEL_DIR, MODEL_DIR_NAME, OPTIONS_MODE, SPOT_MODE


def app_data_dir() -> Path:
    path = Path.home() / APP_DATA_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return app_data_dir() / DB_FILE_NAME


def ensure_db() -> None:
    with sqlite3.connect(db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_configs (
                mode TEXT NOT NULL,
                model_name TEXT NOT NULL,
                ticker TEXT NOT NULL,
                interval TEXT NOT NULL,
                rows INTEGER NOT NULL,
                include_in_run_all INTEGER NOT NULL,
                buy_threshold REAL NOT NULL,
                sell_threshold REAL NOT NULL,
                stop_loss_strategy TEXT NOT NULL DEFAULT 'none',
                fixed_stop_pct REAL NOT NULL DEFAULT 2.0,
                PRIMARY KEY (mode, model_name)
            )
            """
        )
        _ensure_column(conn, "model_configs", "stop_loss_strategy", "TEXT NOT NULL DEFAULT 'none'")
        _ensure_column(conn, "model_configs", "fixed_stop_pct", "REAL NOT NULL DEFAULT 2.0")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def ensure_model_dir() -> None:
    mode_model_dir(OPTIONS_MODE)
    mode_model_dir(SPOT_MODE)


def mode_model_dir(mode: str) -> str:
    if mode not in (OPTIONS_MODE, SPOT_MODE):
        raise ValueError("Invalid mode.")
    path = app_data_dir() / MODEL_DIR_NAME / mode
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _legacy_configs_path(mode: str) -> Path:
    return Path(LEGACY_MODEL_DIR) / mode / LEGACY_MODEL_CONFIGS_FILE


def _legacy_model_dirs(mode: str) -> List[Path]:
    # Old versions stored model JSON files in ./saved/<mode>.
    # Newer versions used ./saved_models/<mode>.
    return [Path("saved") / mode, Path(LEGACY_MODEL_DIR) / mode]


def _model_bundle_path(mode: str, model_name: str) -> Path:
    primary = Path(mode_model_dir(mode)) / f"{model_name}.json"
    if primary.exists():
        return primary
    for legacy_dir in _legacy_model_dirs(mode):
        candidate = legacy_dir / f"{model_name}.json"
        if candidate.exists():
            return candidate
    return primary


def load_model_configs(mode: str) -> Dict[str, Dict[str, object]]:
    ensure_db()
    out: Dict[str, Dict[str, object]] = {}
    with sqlite3.connect(db_path()) as conn:
        rows = conn.execute(
            "SELECT model_name, ticker, interval, rows, include_in_run_all, buy_threshold, sell_threshold, stop_loss_strategy, fixed_stop_pct FROM model_configs WHERE mode = ?",
            (mode,),
        ).fetchall()
    for model_name, ticker, interval, row_count, include_in_run_all, buy_threshold, sell_threshold, stop_loss_strategy, fixed_stop_pct in rows:
        out[str(model_name)] = {
            "ticker": str(ticker),
            "interval": str(interval),
            "rows": int(row_count),
            "include_in_run_all": bool(include_in_run_all),
            "buy_threshold": float(buy_threshold),
            "sell_threshold": float(sell_threshold),
            "stop_loss_strategy": str(stop_loss_strategy or "none"),
            "fixed_stop_pct": float(fixed_stop_pct if fixed_stop_pct is not None else 2.0),
        }

    if out:
        return out

    legacy_path = _legacy_configs_path(mode)
    if legacy_path.exists():
        with legacy_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    return {}


def save_model_configs(mode: str, configs: Dict[str, Dict[str, object]]) -> None:
    ensure_db()
    with sqlite3.connect(db_path()) as conn:
        conn.execute("DELETE FROM model_configs WHERE mode = ?", (mode,))
        for model_name, cfg in configs.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO model_configs
                (mode, model_name, ticker, interval, rows, include_in_run_all, buy_threshold, sell_threshold, stop_loss_strategy, fixed_stop_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    mode,
                    model_name,
                    str(cfg.get("ticker", "AAPL")),
                    str(cfg.get("interval", "1d")),
                    int(cfg.get("rows", 250)),
                    1 if bool(cfg.get("include_in_run_all", True)) else 0,
                    float(cfg.get("buy_threshold", 0.6)),
                    float(cfg.get("sell_threshold", 0.4)),
                    str(cfg.get("stop_loss_strategy", "none")),
                    float(cfg.get("fixed_stop_pct", 2.0)),
                ),
            )


def sanitize_model_name(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name).strip("_")


def save_model_bundle(mode: str, model_name: str, bundle: Dict[str, object]) -> str:
    safe_name = sanitize_model_name(model_name)
    if not safe_name:
        raise ValueError("Model name must include letters or numbers.")
    path = Path(mode_model_dir(mode)) / f"{safe_name}.json"
    payload = {k: bundle[k] for k in ["feature_names", "means", "stds", "lin_weights", "lin_bias", "logit_weights", "logit_bias"]}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return str(path)


def list_saved_models(mode: str) -> List[str]:
    seen: set[str] = set()
    model_names: List[str] = []
    all_dirs = [Path(mode_model_dir(mode)), *_legacy_model_dirs(mode)]
    for model_dir in all_dirs:
        if not model_dir.exists():
            continue
        for model_file in model_dir.glob("*.json"):
            name = model_file.stem
            if name in seen:
                continue
            seen.add(name)
            model_names.append(name)
    return sorted(model_names)


def load_model_bundle(mode: str, model_name: str) -> Dict[str, object]:
    path = _model_bundle_path(mode, model_name)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
