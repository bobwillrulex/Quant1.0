#!/usr/bin/env python3
"""
Quant probability model for a momentum/reversal strategy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from html import escape
from typing import TYPE_CHECKING, Dict
from urllib.error import URLError
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from quant.constants import OPTIONS_MODE, SPOT_MODE
from quant.data import fetch_market_rows, load_csv, synthetic_data
from quant.discord_notify import send_discord_webhook
from quant.env_bootstrap import load_local_env_files
from quant.ml import (
    evaluate_bundle,
    get_strategy_feature_builder,
    infer_bundle_feature_set,
    normalize_feature_set,
    parse_thresholds,
    predict_signal,
    run_model,
    train_strategy_models,
    train_test_split,
)
from quant.stop_loss import (
    MODEL_MAE_DEFAULT,
    StopLossConfig,
    StopLossStrategy,
    parse_stop_loss_strategy,
    stop_loss_price,
    validate_fixed_stop_pct,
    validate_max_hold_bars,
    validate_take_profit_pct,
)
from quant.storage import (
    get_app_setting,
    list_saved_models,
    list_evaluation_snapshots,
    load_evaluation_snapshot,
    load_model_bundle,
    load_model_configs,
    mode_model_dir,
    sanitize_model_name,
    save_evaluation_snapshot,
    save_model_bundle,
    save_model_configs,
    set_app_setting,
    delete_evaluation_snapshot,
)
from bot_manager import create_bot, delete_bot, get_all_bots, get_bot, persist_bot, start_bot, stop_bot

if TYPE_CHECKING:
    from flask import Flask

load_local_env_files()

DISPLAY_TIMEZONE = ZoneInfo("America/Vancouver")


def format_display_time(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "n/a"
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    localized = parsed.astimezone(DISPLAY_TIMEZONE)
    return localized.strftime("%Y-%m-%d %H:%M:%S %Z")


def default_model_config() -> Dict[str, object]:
    return {
        "ticker": "AAPL",
        "interval": "1d",
        "rows": 250,
        "include_in_run_all": True,
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
        "stop_loss_strategy": StopLossStrategy.NONE.value,
        "fixed_stop_pct": 2.0,
        "take_profit_pct": 0.0,
        "max_hold_bars": 0,
        "prediction_horizon": 5,
    }


def get_model_config(model_name: str, configs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    merged = dict(default_model_config())
    stored = configs.get(model_name, {})
    if isinstance(stored, dict):
        merged.update(stored)
    return merged


def parse_csv_values(raw: str, *, uppercase: bool = False) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if uppercase:
        return [value.upper() for value in values]
    return values


def extract_timestamp_range(rows: list[dict[str, object]]) -> tuple[str, str]:
    if not rows:
        return ("n/a", "n/a")

    def _label(row: dict[str, object]) -> str:
        timestamp = row.get("timestamp")
        if timestamp is None:
            return "n/a"
        text = str(timestamp).strip()
        return text if text else "n/a"

    return (_label(rows[0]), _label(rows[-1]))


def build_default_model_name(*, ticker: str, interval: str, row_count: int, feature_set: str, prediction_horizon: int) -> str:
    return sanitize_model_name(f"{ticker}_{interval}_{row_count}_{prediction_horizon}")


def wait_for_ngrok_url(*, timeout_seconds: float = 10.0) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urlopen("http://127.0.0.1:4040/api/tunnels", timeout=1.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            tunnels = payload.get("tunnels", [])
            https_url = ""
            http_url = ""
            for tunnel in tunnels:
                if not isinstance(tunnel, dict):
                    continue
                public_url = str(tunnel.get("public_url", ""))
                if public_url.startswith("https://"):
                    https_url = public_url
                elif public_url.startswith("http://"):
                    http_url = public_url
            if https_url:
                return https_url
            if http_url:
                return http_url
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise RuntimeError("Timed out waiting for ngrok tunnel URL. Is ngrok installed and running?")


def start_ngrok_tunnel(*, host: str, port: int, authtoken: str = "") -> tuple[subprocess.Popen[bytes], str]:
    ngrok_bin = os.getenv("NGROK_BIN", "ngrok")
    if authtoken:
        subprocess.run(
            [ngrok_bin, "config", "add-authtoken", authtoken],
            check=True,
            capture_output=True,
            text=True,
        )
    process = subprocess.Popen(
        [ngrok_bin, "http", f"{host}:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    public_url = wait_for_ngrok_url()
    return process, public_url


def parse_manual_feature_weights(raw: str, expected_feature_count: int) -> list[float]:
    if expected_feature_count <= 0:
        raise ValueError("Feature set must include at least one feature for manual weighting.")
    if not raw:
        raise ValueError("Manual feature weights were not provided.")
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("Manual feature weights must be provided as a list.")
    if len(parsed) != expected_feature_count:
        raise ValueError("Manual feature weight count does not match the selected feature set.")
    weights: list[float] = []
    for weight in parsed:
        value = float(weight)
        weights.append(value)
    total = sum(weights)
    if abs(total) <= 1e-12:
        raise ValueError("Manual feature weights total cannot be zero.")
    return [weight / total for weight in weights]


def _mc_quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * frac)


def build_forward_monte_carlo_projection(
    bundle: Dict[str, object],
    *,
    expected_return_per_bar: float,
    simulations: int = 300,
    seed: int = 7,
) -> dict[str, float | int | list[float]] | None:
    historical_mc = bundle.get("historical_monte_carlo")
    if not isinstance(historical_mc, dict):
        return None
    historical_summary = historical_mc.get("summary", {})
    if not isinstance(historical_summary, dict):
        return None
    historical_std_total = float(historical_summary.get("std_return", 0.0))
    historical_raw = historical_mc.get("raw_results", [])
    horizon_bars = 20
    if isinstance(historical_raw, list) and historical_raw:
        horizon_bars = max(1, min(252, len(historical_raw)))
    per_bar_vol = max(0.0, historical_std_total / math.sqrt(float(horizon_bars)))
    rng = random.Random(seed)
    forward_total_returns: list[float] = []
    for _ in range(max(10, simulations)):
        equity = 1.0
        for _step in range(horizon_bars):
            sampled_ret = rng.gauss(expected_return_per_bar, per_bar_vol)
            sampled_ret = max(-0.95, min(0.95, sampled_ret))
            equity *= 1.0 + sampled_ret
        forward_total_returns.append(equity - 1.0)
    if not forward_total_returns:
        return None
    return {
        "simulations": len(forward_total_returns),
        "horizon_bars": horizon_bars,
        "expected_return": sum(forward_total_returns) / len(forward_total_returns),
        "median_return": _mc_quantile(forward_total_returns, 0.5),
        "p5_return": _mc_quantile(forward_total_returns, 0.05),
        "p95_return": _mc_quantile(forward_total_returns, 0.95),
        "worst_return": min(forward_total_returns),
        "best_return": max(forward_total_returns),
        "probability_profit": sum(1 for value in forward_total_returns if value > 0.0) / len(forward_total_returns),
        "probability_loss": sum(1 for value in forward_total_returns if value < 0.0) / len(forward_total_returns),
        "distribution": forward_total_returns,
    }


def evaluate_run_all_models(saved_models, model_configs, *, mode: str, long_only: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data_provider = str(model_configs.get("__ui_data_provider__", "yfinance")).strip().lower()
    twelve_api_key = str(model_configs.get("__ui_twelve_api_key__", "")).strip()
    massive_api_key = str(model_configs.get("__ui_massive_api_key__", "")).strip()
    for model_name in saved_models:
        cfg = get_model_config(model_name, model_configs)
        if not cfg.get("include_in_run_all", True):
            continue
        try:
            dataset, provider_notice = fetch_market_rows(
                ticker=str(cfg.get("ticker", "AAPL")),
                interval=str(cfg.get("interval", "1d")),
                row_count=int(cfg.get("rows", 250)),
                provider=data_provider,
                twelve_api_key=twelve_api_key,
                massive_api_key=massive_api_key,
                prediction_horizon=int(cfg.get("prediction_horizon", 5)),
            )
            latest_row = dataset[-1]
            provider_notice_html = f"<br><span class='muted'>{provider_notice}</span>" if provider_notice else ""
            bundle = load_model_bundle(mode, model_name)
            buy_threshold = float(cfg.get("buy_threshold", 0.6))
            sell_threshold = float(cfg.get("sell_threshold", 0.4))
            prediction = predict_signal(bundle, latest_row, buy_threshold=buy_threshold, sell_threshold=sell_threshold, long_only=long_only)
            stop_strategy = parse_stop_loss_strategy(str(cfg.get("stop_loss_strategy", StopLossStrategy.NONE.value)))
            stop_price_value = stop_loss_price(
                strategy=stop_strategy,
                action=str(prediction["action"]),
                reference_price=float(latest_row.get("close", 0.0)),
                expected_return=float(prediction["expected_return"]),
                fixed_pct=float(cfg.get("fixed_stop_pct", 2.0)),
                atr_fraction=float(latest_row.get("atr_frac", 0.0)),
                model_mae=MODEL_MAE_DEFAULT,
            )
            rows.append(
                {
                    "model_name": model_name,
                    "ticker": str(cfg.get("ticker", "AAPL")),
                    "interval": str(cfg.get("interval", "1d")),
                    "row_count": int(cfg.get("rows", 250)),
                    "buy_threshold": buy_threshold,
                    "sell_threshold": sell_threshold,
                    "stop_strategy": stop_strategy.value,
                    "expected_return": float(prediction["expected_return"]),
                    "p_up": float(prediction["p_up"]),
                    "stop_price": stop_price_value,
                    "action": str(prediction["action"]),
                    "forward_monte_carlo": build_forward_monte_carlo_projection(
                        bundle,
                        expected_return_per_bar=float(prediction["expected_return"]),
                        seed=abs(hash(model_name)) % 100_000,
                    ),
                    "provider_notice": provider_notice_html,
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model_name": model_name,
                    "ticker": str(cfg.get("ticker", "AAPL")),
                    "interval": str(cfg.get("interval", "1d")),
                    "row_count": int(cfg.get("rows", 250)),
                    "buy_threshold": float(cfg.get("buy_threshold", 0.6)),
                    "sell_threshold": float(cfg.get("sell_threshold", 0.4)),
                    "stop_strategy": str(cfg.get("stop_loss_strategy", StopLossStrategy.NONE.value)),
                    "expected_return": 0.0,
                    "p_up": 0.0,
                    "stop_price": None,
                    "action": "ERROR",
                    "forward_monte_carlo": None,
                    "provider_notice": "",
                    "error": str(exc),
                }
            )
    return rows


def build_run_all_rows(saved_models, model_configs, *, mode: str, long_only: bool) -> str:
    results = evaluate_run_all_models(saved_models, model_configs, mode=mode, long_only=long_only)
    return build_run_all_rows_from_results(results)


def build_run_all_rows_from_results(results: list[dict[str, object]]) -> str:
    run_all_rows = ""
    interval_order = {"5m": 0, "15m": 1, "1h": 2, "1d": 3}
    interval_labels = {"5m": "5 min presets", "15m": "15 min presets", "1h": "1h presets", "1d": "1d presets"}

    def normalize_interval(value: object) -> str:
        return str(value or "1d").strip().lower()

    sorted_results = sorted(
        results,
        key=lambda item: (
            interval_order.get(normalize_interval(item.get("interval", "1d")), 99),
            str(item.get("ticker", "")),
            str(item.get("model_name", "")),
        ),
    )

    current_interval = None
    for item in sorted_results:
        item_interval = normalize_interval(item.get("interval", "1d"))
        if item_interval != current_interval:
            current_interval = item_interval
            run_all_rows += (
                "<tr class='run-all-group-row'>"
                f"<td colspan='10'><strong>{interval_labels.get(item_interval, f'{item_interval} presets')}</strong></td>"
                "</tr>"
            )
            run_all_rows += "<tr class='run-all-group-divider'><td colspan='10'></td></tr>"
        if item["error"]:
            run_all_rows += (
                "<tr>"
                f"<td>{item['model_name']}</td>"
                f"<td>{item['ticker']}</td>"
                f"<td>{item_interval}</td>"
                f"<td>{int(item['row_count'])}</td>"
                f"<td>{float(item['buy_threshold']):.2f} / {float(item['sell_threshold']):.2f}</td>"
                f"<td>{item['stop_strategy']}</td>"
                "<td colspan='4' style='color:#ff7b7b;'>"
                f"Run failed: {item['error']}"
                "</td>"
                "</tr>"
            )
            continue
        stop_price = item["stop_price"]
        stop_price_display = f"{float(stop_price):.4f}" if stop_price is not None else "n/a"
        forward_mc = item.get("forward_monte_carlo")
        model_cell = str(item["model_name"])
        if isinstance(forward_mc, dict):
            distribution_values = forward_mc.get("distribution", [])
            distribution_chart_html = ""
            if isinstance(distribution_values, list):
                distribution_chart_html = render_distribution_histogram(
                    [float(value) for value in distribution_values],
                    stroke="#7f94b7",
                    fill="#9eb2d2",
                )
            model_cell = (
                "<details>"
                f"<summary>{item['model_name']}</summary>"
                "<div class='muted' style='margin-top:.4rem;'>"
                f"Forward MC · sims {int(forward_mc.get('simulations', 0))} · horizon {int(forward_mc.get('horizon_bars', 0))} bars"
                f"<br>Expected {float(forward_mc.get('expected_return', 0.0)):+.2%} | Median {float(forward_mc.get('median_return', 0.0)):+.2%}"
                f"<br>P5/P95 {float(forward_mc.get('p5_return', 0.0)):+.2%} / {float(forward_mc.get('p95_return', 0.0)):+.2%}"
                f"<br>P(Profit) {float(forward_mc.get('probability_profit', 0.0)):.1%} | P(Loss) {float(forward_mc.get('probability_loss', 0.0)):.1%}"
                f"{distribution_chart_html}"
                "</div>"
                "</details>"
            )
        run_all_rows += (
            "<tr>"
            f"<td>{model_cell}</td>"
            f"<td>{item['ticker']}</td>"
            f"<td>{item_interval}</td>"
            f"<td>{int(item['row_count'])}</td>"
            f"<td>{float(item['buy_threshold']):.2f} / {float(item['sell_threshold']):.2f}</td>"
            f"<td>{item['stop_strategy']}</td>"
            f"<td>{float(item['expected_return']):+.4%}</td>"
            f"<td>{float(item['p_up']):.2%}</td>"
            f"<td>{stop_price_display}</td>"
            f"<td><strong>{item['action']}</strong>{item['provider_notice']}</td>"
            "</tr>"
        )
    return run_all_rows


def is_us_market_open(now: datetime | None = None) -> bool:
    ts = now or datetime.now(tz=ZoneInfo("America/New_York"))
    if ts.weekday() >= 5:
        return False
    minute_of_day = ts.hour * 60 + ts.minute
    return (9 * 60 + 30) <= minute_of_day < (16 * 60)


def seconds_until_next_aligned_five_minute(now: datetime | None = None) -> float:
    ts = now or datetime.utcnow()
    minute_bucket = (ts.minute // 5) * 5
    aligned = ts.replace(minute=minute_bucket, second=0, microsecond=0)
    if aligned <= ts:
        aligned += timedelta(minutes=5)
    return max(1.0, (aligned - ts).total_seconds())


class RunAllMonitor:
    def __init__(self) -> None:
        self._state: dict[str, dict[str, object]] = {
            OPTIONS_MODE: {
                "running": False,
                "thread": None,
                "last_actions": {},
                "last_tick": "",
                "started_at": "",
                "last_error": "",
                "last_market_state": "unknown",
                "worker_state": "stopped",
                "last_rows": [],
                "next_tick_at": "",
            },
            SPOT_MODE: {
                "running": False,
                "thread": None,
                "last_actions": {},
                "last_tick": "",
                "started_at": "",
                "last_error": "",
                "last_market_state": "unknown",
                "worker_state": "stopped",
                "last_rows": [],
                "next_tick_at": "",
            },
        }
        self._lock = threading.Lock()

    def status(self, mode: str) -> dict[str, object]:
        with self._lock:
            mode_state = dict(self._state[mode])
            worker = mode_state.get("thread")
            mode_state["worker_alive"] = bool(worker and isinstance(worker, threading.Thread) and worker.is_alive())
            running_requested = bool(mode_state.get("running"))
            if running_requested and not mode_state["worker_alive"] and mode_state.get("worker_state") != "crashed":
                mode_state["worker_state"] = "stopped"
            mode_state["last_actions_count"] = len(dict(mode_state.get("last_actions", {})))
            mode_state.pop("thread", None)
            return mode_state

    def start(self, *, mode: str, long_only: bool, data_provider: str, twelve_api_key: str, massive_api_key: str, webhook_url: str) -> None:
        with self._lock:
            mode_state = self._state[mode]
            if bool(mode_state.get("running")):
                return
            mode_state["running"] = True
            mode_state["last_actions"] = {}
            now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
            mode_state["started_at"] = now_iso
            mode_state["last_tick"] = now_iso
            mode_state["last_error"] = ""
            mode_state["worker_state"] = "running"
            mode_state["last_rows"] = []
            mode_state["next_tick_at"] = ""
            worker = threading.Thread(
                target=self._loop,
                kwargs={
                    "mode": mode,
                    "long_only": long_only,
                    "data_provider": data_provider,
                    "twelve_api_key": twelve_api_key,
                    "massive_api_key": massive_api_key,
                    "webhook_url": webhook_url,
                },
                daemon=True,
            )
            mode_state["thread"] = worker
            worker.start()
        self._send_lifecycle_message(mode=mode, webhook_url=webhook_url, status="up")

    def stop(self, mode: str, webhook_url: str) -> None:
        with self._lock:
            self._state[mode]["running"] = False
            self._state[mode]["worker_state"] = "stopped"
            self._state[mode]["next_tick_at"] = ""
        self._send_lifecycle_message(mode=mode, webhook_url=webhook_url, status="down")

    def _loop(self, *, mode: str, long_only: bool, data_provider: str, twelve_api_key: str, massive_api_key: str, webhook_url: str) -> None:
        try:
            while True:
                with self._lock:
                    if not bool(self._state[mode]["running"]):
                        self._state[mode]["thread"] = None
                        return
                market_open = is_us_market_open()
                with self._lock:
                    self._state[mode]["last_market_state"] = "open" if market_open else "closed"
                try:
                    configs = load_model_configs(mode)
                    configs["__ui_data_provider__"] = data_provider
                    configs["__ui_twelve_api_key__"] = twelve_api_key
                    configs["__ui_massive_api_key__"] = massive_api_key
                    models = list_saved_models(mode)
                    rows = evaluate_run_all_models(models, configs, mode=mode, long_only=long_only)
                    self._notify_action_changes(mode=mode, rows=rows, webhook_url=webhook_url)
                    with self._lock:
                        self._state[mode]["last_error"] = ""
                        self._state[mode]["last_rows"] = rows
                except Exception as exc:
                    with self._lock:
                        self._state[mode]["last_error"] = str(exc)
                wait_seconds = seconds_until_next_aligned_five_minute()
                now_iso = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
                next_tick_iso = datetime.utcfromtimestamp(time.time() + wait_seconds).replace(microsecond=0).isoformat() + "Z"
                with self._lock:
                    self._state[mode]["last_tick"] = now_iso
                    self._state[mode]["next_tick_at"] = next_tick_iso
                time.sleep(wait_seconds)
        except Exception as exc:
            with self._lock:
                self._state[mode]["running"] = False
                self._state[mode]["thread"] = None
                self._state[mode]["worker_state"] = "crashed"
                self._state[mode]["last_error"] = str(exc)
            self._send_lifecycle_message(mode=mode, webhook_url=webhook_url, status="crash")
            return

    def _notify_action_changes(self, *, mode: str, rows: list[dict[str, object]], webhook_url: str) -> None:
        with self._lock:
            known = dict(self._state[mode].get("last_actions", {}))
        new_known = dict(known)
        for row in rows:
            if row["error"]:
                continue
            model_name = str(row["model_name"])
            action = str(row["action"])
            prev = str(known.get(model_name, ""))
            new_known[model_name] = action
            if not prev or prev == action:
                continue
            content = (
                f"📈 Quant1.0 {mode.upper()} action change for {row['ticker']} ({model_name})\n"
                f"{prev} ➜ {action} | Model: {model_name} | P(Up): {float(row['p_up']):.2%} | Linear: {float(row['expected_return']):+.4%}"
            )
            send_discord_webhook(webhook_url, content)
        with self._lock:
            self._state[mode]["last_actions"] = new_known

    def _send_lifecycle_message(self, *, mode: str, webhook_url: str, status: str) -> None:
        if not webhook_url.strip():
            return
        emoji = "🟢" if status == "up" else ("🟠" if status == "down" else "🔴")
        content = f"{emoji} Quant1.0 {mode.upper()} bot {status}."
        send_discord_webhook(webhook_url, content)



RUN_ALL_MONITOR = RunAllMonitor()


def render_hold_time_boxplot(stats: Dict[str, object], *, stroke: str = "#8ca0bf", accent: str = "#cfd8e6") -> str:
    count = int(float(stats.get("count", 0.0)))
    if count <= 0:
        return "<p class='muted'>No active hold streaks in this test window.</p>"
    min_v = float(stats.get("min", 0.0))
    q1 = float(stats.get("q1", 0.0))
    median = float(stats.get("median", 0.0))
    q3 = float(stats.get("q3", 0.0))
    max_v = float(stats.get("max", 0.0))
    span = max(1e-9, max_v - min_v)

    def scale_x(value: float) -> float:
        return 22.0 + ((value - min_v) / span) * 236.0

    min_x = scale_x(min_v)
    q1_x = scale_x(q1)
    med_x = scale_x(median)
    q3_x = scale_x(q3)
    max_x = scale_x(max_v)

    def label(x: float, y: float, text: str) -> str:
        return f"<text x='{x:.1f}' y='{y:.1f}' fill='{accent}' font-size='10' text-anchor='middle'>{text}</text>"

    return (
        "<div style='display:inline-flex; flex-direction:column; gap:0.35rem;'>"
        "<svg width='280' height='96' viewBox='0 0 280 96' xmlns='http://www.w3.org/2000/svg'>"
        f"<line x1='{min_x:.1f}' y1='48' x2='{max_x:.1f}' y2='48' stroke='{stroke}' stroke-width='1.2'/>"
        f"<line x1='{min_x:.1f}' y1='39' x2='{min_x:.1f}' y2='57' stroke='{stroke}' stroke-width='1.2'/>"
        f"<line x1='{max_x:.1f}' y1='39' x2='{max_x:.1f}' y2='57' stroke='{stroke}' stroke-width='1.2'/>"
        f"<rect x='{q1_x:.1f}' y='34' width='{max(2.0, q3_x - q1_x):.1f}' height='28' fill='none' stroke='{stroke}' stroke-width='1.4' rx='4'/>"
        f"<line x1='{med_x:.1f}' y1='31' x2='{med_x:.1f}' y2='65' stroke='{accent}' stroke-width='1.5'/>"
        f"{label(min_x, 23, f'min {min_v:.1f}')}"
        f"{label(q1_x, 18, f'q1 {q1:.1f}')}"
        f"{label(med_x, 76, f'med {median:.1f}')}"
        f"{label(q3_x, 18, f'q3 {q3:.1f}')}"
        f"{label(max_x, 23, f'max {max_v:.1f}')}"
        "</svg>"
        f"<span class='muted' style='font-size:0.82rem;'>Hold streak samples: {count}</span>"
        "</div>"
    )


def render_distribution_histogram(values: list[float], *, stroke: str = "#8ca0bf", fill: str = "#cfd8e6") -> str:
    if not values:
        return "<p class='muted'>No distribution values available.</p>"
    ordered = sorted(float(v) for v in values)
    min_v = ordered[0]
    max_v = ordered[-1]
    n_bins = max(8, min(24, int(math.sqrt(len(ordered)))))
    span = max_v - min_v
    if span <= 1e-12:
        counts = [len(ordered)]
        edges = [min_v - 0.5, max_v + 0.5]
    else:
        width = span / n_bins
        counts = [0] * n_bins
        edges = [min_v + (width * idx) for idx in range(n_bins + 1)]
        for value in ordered:
            idx = min(n_bins - 1, int((value - min_v) / width))
            counts[idx] += 1
    chart_w = 680.0
    chart_h = 220.0
    left = 40.0
    bottom = 175.0
    usable_w = chart_w - left - 20.0
    usable_h = 130.0
    bar_count = len(counts)
    bar_w = usable_w / max(1, bar_count)
    max_count = max(counts) if counts else 1
    bars = []
    for idx, count in enumerate(counts):
        h = 0.0 if max_count <= 0 else (count / max_count) * usable_h
        x = left + idx * bar_w + 1.0
        y = bottom - h
        bars.append(
            f"<rect x='{x:.2f}' y='{y:.2f}' width='{max(1.0, bar_w - 2.0):.2f}' height='{h:.2f}' "
            f"fill='{fill}' fill-opacity='0.45' stroke='{stroke}' stroke-width='1'/>"
        )
    return (
        "<div class='mini-chart' style='overflow:auto;'>"
        f"<svg viewBox='0 0 {chart_w:.0f} {chart_h:.0f}' width='100%' height='220' role='img' aria-label='Monte Carlo return distribution'>"
        f"<line x1='{left:.2f}' y1='{bottom:.2f}' x2='{chart_w - 20.0:.2f}' y2='{bottom:.2f}' stroke='{stroke}' stroke-width='1.2'/>"
        f"<line x1='{left:.2f}' y1='{bottom - usable_h:.2f}' x2='{left:.2f}' y2='{bottom:.2f}' stroke='{stroke}' stroke-width='1.2'/>"
        + "".join(bars)
        + f"<text x='{left:.2f}' y='198' fill='{stroke}' font-size='11'>{min_v:+.1%}</text>"
        + f"<text x='{chart_w - 64.0:.2f}' y='198' fill='{stroke}' font-size='11'>{max_v:+.1%}</text>"
        + f"<text x='{left:.2f}' y='{bottom - usable_h - 6.0:.2f}' fill='{stroke}' font-size='11'>count</text>"
        + "</svg>"
        "</div>"
    )


MOBILE_USER_AGENT_TOKENS = (
    "android",
    "iphone",
    "ipad",
    "ipod",
    "mobile",
    "blackberry",
    "opera mini",
    "iemobile",
    "windows phone",
)


def is_mobile_request(request: object) -> bool:
    explicit_mode = getattr(request, "args", {}).get("ui", "").strip().lower()
    if explicit_mode in {"mobile", "m"}:
        return True
    if explicit_mode in {"desktop", "d"}:
        return False
    ua = str(getattr(request, "headers", {}).get("User-Agent", "")).lower()
    if any(token in ua for token in MOBILE_USER_AGENT_TOKENS):
        return True
    sec_mobile = str(getattr(request, "headers", {}).get("Sec-CH-UA-Mobile", "")).strip()
    return sec_mobile == "?1"


def create_app() -> "Flask":
    from flask import Flask, jsonify, redirect, request, url_for

    app = Flask(__name__)

    def _bot_payload(bot: object) -> dict[str, object]:
        return {
            "id": str(getattr(bot, "id", "")),
            "name": str(getattr(bot, "name", "")),
            "status": str(getattr(bot, "status", "stopped")),
            "day_pnl": float(getattr(bot, "day_pnl", 0.0)),
            "total_pnl": float(getattr(bot, "total_pnl", 0.0)),
            "position": float(getattr(bot, "position", 0.0)),
            "cash": float(getattr(bot, "cash", 0.0)),
            "last_polled_bid": getattr(bot, "last_polled_bid", None),
            "last_polled_ask": getattr(bot, "last_polled_ask", None),
            "last_polled_spread": getattr(bot, "last_polled_spread", None),
            "last_polled_timestamp": getattr(bot, "last_polled_timestamp", None),
            "p_up": float(getattr(bot, "last_p_up", 0.5)),
        }

    def _parse_trade_timestamp(value: object) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        normalized = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo("UTC"))

    def _build_bot_metrics(bot: object) -> dict[str, object]:
        trades = getattr(bot, "trades", [])
        if not isinstance(trades, list):
            trades = []
        now = datetime.now(tz=ZoneInfo("UTC"))
        day_cutoff = now - timedelta(days=1)
        week_cutoff = now - timedelta(days=7)
        month_cutoff = now - timedelta(days=30)

        pnl_day = 0.0
        pnl_week = 0.0
        pnl_month = 0.0
        curve = [0.0]
        cumulative = 0.0

        for trade in trades:
            if not isinstance(trade, dict):
                continue
            trade_pnl = float(trade.get("pnl", 0.0))
            cumulative += trade_pnl
            curve.append(cumulative)
            ts = _parse_trade_timestamp(trade.get("timestamp"))
            if ts is None:
                continue
            if ts >= day_cutoff:
                pnl_day += trade_pnl
            if ts >= week_cutoff:
                pnl_week += trade_pnl
            if ts >= month_cutoff:
                pnl_month += trade_pnl

        peak = curve[0]
        drawdowns: list[float] = []
        for point in curve:
            peak = max(peak, point)
            drawdown = peak - point
            drawdowns.append(drawdown)
        max_drawdown = max(drawdowns) if drawdowns else 0.0
        non_zero_drawdowns = [value for value in drawdowns if value > 0.0]
        avg_drawdown = sum(non_zero_drawdowns) / len(non_zero_drawdowns) if non_zero_drawdowns else 0.0
        closed_trade_pnls = [float(trade.get("pnl", 0.0)) for trade in trades if isinstance(trade, dict) and abs(float(trade.get("pnl", 0.0))) > 1e-12]
        closed_trade_count = len(closed_trade_pnls)
        winning_trade_count = sum(1 for pnl in closed_trade_pnls if pnl > 0.0)
        avg_gain_per_trade = (sum(closed_trade_pnls) / closed_trade_count) if closed_trade_count else 0.0

        equity_curve = [0.0]
        cumulative_closed = 0.0
        for pnl in closed_trade_pnls:
            cumulative_closed += pnl
            equity_curve.append(cumulative_closed)
        returns: list[float] = []
        for idx in range(1, len(equity_curve)):
            prev = max(1.0, abs(equity_curve[idx - 1]))
            returns.append((equity_curve[idx] - equity_curve[idx - 1]) / prev)
        sharpe = 0.0
        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
            if variance > 1e-12:
                sharpe = mean_return / (variance**0.5)

        stop_loss_exits = 0
        hold_bars: list[float] = []
        open_ts: datetime | None = None
        timeframe_text = str(getattr(bot, "timeframe", "1m")).strip().lower()
        minutes_per_bar = 1.0
        if timeframe_text.endswith("m"):
            try:
                minutes_per_bar = max(1.0, float(timeframe_text[:-1]))
            except ValueError:
                minutes_per_bar = 1.0
        elif timeframe_text.endswith("h"):
            try:
                minutes_per_bar = max(60.0, float(timeframe_text[:-1]) * 60.0)
            except ValueError:
                minutes_per_bar = 60.0
        elif timeframe_text in {"1d", "d", "day", "daily"}:
            minutes_per_bar = 390.0
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            ts = _parse_trade_timestamp(trade.get("timestamp"))
            pnl = float(trade.get("pnl", 0.0))
            side = str(trade.get("side", "")).upper()
            if open_ts is None and side in {"BUY", "SELL"}:
                open_ts = ts
            if pnl < -1e-12:
                stop_loss_exits += 1
            if abs(pnl) > 1e-12 and open_ts is not None and ts is not None:
                elapsed_minutes = max(0.0, (ts - open_ts).total_seconds() / 60.0)
                hold_bars.append(max(1.0, elapsed_minutes / minutes_per_bar))
                open_ts = None

        sorted_holds = sorted(hold_bars)

        def _percentile(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            if len(values) == 1:
                return values[0]
            index = (len(values) - 1) * q
            low = int(math.floor(index))
            high = int(math.ceil(index))
            if low == high:
                return values[low]
            ratio = index - low
            return values[low] + (values[high] - values[low]) * ratio

        hold_time_stats = {
            "count": len(sorted_holds),
            "min": float(sorted_holds[0]) if sorted_holds else 0.0,
            "q1": _percentile(sorted_holds, 0.25),
            "median": _percentile(sorted_holds, 0.5),
            "q3": _percentile(sorted_holds, 0.75),
            "max": float(sorted_holds[-1]) if sorted_holds else 0.0,
        }

        return {
            "max_drawdown": max_drawdown,
            "avg_drawdown": avg_drawdown,
            "pnl_day": pnl_day,
            "pnl_week": pnl_week,
            "pnl_month": pnl_month,
            "trade_count": len(trades),
            "closed_trade_count": closed_trade_count,
            "win_rate": (winning_trade_count / closed_trade_count) if closed_trade_count else 0.0,
            "avg_gain_per_trade": avg_gain_per_trade,
            "stop_loss_exits": stop_loss_exits,
            "sharpe": sharpe,
            "hold_time_stats": hold_time_stats,
        }

    def _bot_details_payload(bot: object) -> dict[str, object]:
        raw_trades = getattr(bot, "trades", [])
        trades = raw_trades if isinstance(raw_trades, list) else []
        return {
            **_bot_payload(bot),
            "model_name": str(getattr(bot, "model_name", "")),
            "ticker": str(getattr(bot, "ticker", "")),
            "timeframe": str(getattr(bot, "timeframe", "")),
            "buy_threshold": float(getattr(bot, "buy_threshold", 0.6)),
            "sell_threshold": float(getattr(bot, "sell_threshold", 0.4)),
            "long_only": bool(getattr(bot, "long_only", False)),
            "daily_buy_timing": str(getattr(bot, "daily_buy_timing", "start_of_day")),
            "intraday_trade_interval": str(getattr(bot, "intraday_trade_interval", "unlimited")),
            "stop_loss": float(getattr(bot, "stop_loss", 0.0)),
            "take_profit": float(getattr(bot, "take_profit", 0.0)),
            "trade_size": float(getattr(bot, "trade_size", 0.0)),
            "metrics": _build_bot_metrics(bot),
            "trades": trades,
        }

    def _parse_bot_create_payload(payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        required_fields = (
            "model",
            "ticker",
            "timeframe",
            "starting_money",
            "buy_threshold",
            "sell_threshold",
            "take_profit",
        )
        missing = [field for field in required_fields if field not in payload]
        if "timeframe" in missing and "candle_time" in payload:
            missing.remove("timeframe")
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}.")

        model_name = str(payload["model"]).strip()
        ticker = str(payload["ticker"]).strip().upper()
        timeframe = str(payload.get("timeframe", payload.get("candle_time", ""))).strip()
        name = str(payload.get("name", "")).strip() or model_name
        if not model_name or not ticker or not timeframe:
            raise ValueError("Fields model, ticker, and timeframe cannot be empty.")

        buy_threshold = float(payload["buy_threshold"])
        sell_threshold = float(payload["sell_threshold"])
        if not (0.0 <= buy_threshold <= 1.0 and 0.0 <= sell_threshold <= 1.0):
            raise ValueError("buy_threshold and sell_threshold must be between 0 and 1.")
        mode = str(payload.get("mode", "options")).strip().lower()
        long_only = mode == "spot"
        daily_buy_timing = str(payload.get("daily_buy_timing", "start_of_day")).strip().lower()
        if daily_buy_timing not in {"start_of_day", "end_of_day"}:
            raise ValueError("daily_buy_timing must be start_of_day or end_of_day.")
        intraday_trade_interval = str(payload.get("intraday_trade_interval", "unlimited")).strip().lower()
        if intraday_trade_interval not in {"unlimited", "5m", "10m", "15m", "1h"}:
            raise ValueError("intraday_trade_interval must be one of: unlimited, 5m, 10m, 15m, 1h.")
        stop_loss_strategy = parse_stop_loss_strategy(str(payload.get("stop_loss_strategy", StopLossStrategy.NONE.value)))
        fixed_stop_pct_value = float(payload.get("fixed_stop_pct", payload.get("stop_loss", 0.02) * 100.0))
        if stop_loss_strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
            validated_fixed_stop_pct = validate_fixed_stop_pct(fixed_stop_pct_value)
            stop_loss_value = validated_fixed_stop_pct / 100.0
        else:
            stop_loss_value = float(payload.get("stop_loss", 0.02))

        execution_settings = payload.get("execution_settings")
        if execution_settings is not None and not isinstance(execution_settings, dict):
            raise ValueError("execution_settings must be an object when provided.")

        return {
            "id": str(uuid.uuid4()),
            "name": name,
            "model_name": model_name,
            "ticker": ticker,
            "timeframe": timeframe,
            "cash": float(payload["starting_money"]),
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
            "long_only": long_only,
            "daily_buy_timing": daily_buy_timing,
            "intraday_trade_interval": intraday_trade_interval,
            "stop_loss": stop_loss_value,
            "take_profit": float(payload["take_profit"]),
            "execution_settings": execution_settings,
        }

    def _parse_bot_update_payload(payload: object, *, existing_bot: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        updated: dict[str, object] = {}
        if "name" in payload:
            name = str(payload.get("name", "")).strip()
            if not name:
                raise ValueError("name cannot be empty.")
            updated["name"] = name
        if "ticker" in payload:
            ticker = str(payload.get("ticker", "")).strip().upper()
            if not ticker:
                raise ValueError("ticker cannot be empty.")
            updated["ticker"] = ticker
        if "timeframe" in payload:
            timeframe = str(payload.get("timeframe", "")).strip()
            if not timeframe:
                raise ValueError("timeframe cannot be empty.")
            updated["timeframe"] = timeframe

        for field in ("buy_threshold", "sell_threshold", "take_profit", "trade_size"):
            if field in payload:
                updated[field] = float(payload[field])
        if "daily_buy_timing" in payload:
            daily_buy_timing = str(payload.get("daily_buy_timing", "")).strip().lower()
            if daily_buy_timing not in {"start_of_day", "end_of_day"}:
                raise ValueError("daily_buy_timing must be start_of_day or end_of_day.")
            updated["daily_buy_timing"] = daily_buy_timing
        if "intraday_trade_interval" in payload:
            intraday_trade_interval = str(payload.get("intraday_trade_interval", "")).strip().lower()
            if intraday_trade_interval not in {"unlimited", "5m", "10m", "15m", "1h"}:
                raise ValueError("intraday_trade_interval must be one of: unlimited, 5m, 10m, 15m, 1h.")
            updated["intraday_trade_interval"] = intraday_trade_interval

        if "buy_threshold" in updated and not (0.0 <= float(updated["buy_threshold"]) <= 1.0):
            raise ValueError("buy_threshold must be between 0 and 1.")
        if "sell_threshold" in updated and not (0.0 <= float(updated["sell_threshold"]) <= 1.0):
            raise ValueError("sell_threshold must be between 0 and 1.")
        if "take_profit" in updated and float(updated["take_profit"]) < 0.0:
            raise ValueError("take_profit must be non-negative.")
        if "trade_size" in updated and float(updated["trade_size"]) <= 0.0:
            raise ValueError("trade_size must be positive.")

        if "stop_loss_strategy" in payload or "fixed_stop_pct" in payload or "stop_loss" in payload:
            strategy_raw = payload.get("stop_loss_strategy", "fixed_percentage")
            stop_loss_strategy = parse_stop_loss_strategy(str(strategy_raw))
            fixed_stop_pct_value = float(payload.get("fixed_stop_pct", payload.get("stop_loss", getattr(existing_bot, "stop_loss", 0.02) * 100.0)))
            if stop_loss_strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
                validated_fixed_stop_pct = validate_fixed_stop_pct(fixed_stop_pct_value)
                updated["stop_loss"] = validated_fixed_stop_pct / 100.0
            else:
                updated["stop_loss"] = float(payload.get("stop_loss", getattr(existing_bot, "stop_loss", 0.02)))

        return updated

    @app.route("/api/bots/form-options", methods=["GET"])
    def bot_form_options() -> object:
        mode_key = SPOT_MODE if request.args.get("mode", "").strip().lower() == "spot" else OPTIONS_MODE
        return jsonify(
            {
                "models": list_saved_models(mode_key),
                "timeframes": ["1m", "5m", "15m", "30m", "1h", "1d"],
                "daily_buy_timing_options": [
                    {"value": "start_of_day", "label": "Beginning of day"},
                    {"value": "end_of_day", "label": "Last minute (end of day)"},
                ],
                "intraday_trade_interval_options": [
                    {"value": "unlimited", "label": "Unlimited"},
                    {"value": "5m", "label": "Every 5 minutes"},
                    {"value": "10m", "label": "Every 10 minutes"},
                    {"value": "15m", "label": "Every 15 minutes"},
                    {"value": "1h", "label": "Every hour"},
                ],
                "default_ticker": "AAPL",
            }
        )

    @app.route("/api/bots", methods=["GET"])
    def list_bots() -> object:
        return jsonify([_bot_payload(bot) for bot in get_all_bots()])

    @app.route("/bots", methods=["GET"])
    @app.route("/spot/bots", methods=["GET"])
    def bots_dashboard() -> str:
        is_spot = request.path.startswith("/spot")
        home_href = "/spot" if is_spot else "/"
        manage_href = "/spot/manage-models" if is_spot else "/manage-models"
        run_models_href = "/spot/run-models" if is_spot else "/run-models"
        bots_href = "/spot/bots" if is_spot else "/bots"
        mode_switch_href = "/" if is_spot else "/spot"
        mode_switch_label = "Switch to Options Mode" if is_spot else "Switch to Spot Mode"
        brand_label = "Quant Trader • Spot Mode" if is_spot else "Quant Trader • Options Mode"
        theme_bg = "#100d07" if is_spot else "#090c12"
        theme_text = "#efe0be" if is_spot else "#c6ccd7"
        theme_panel = "#19130b" if is_spot else "#121722"
        theme_panel2 = "#221b10" if is_spot else "#1a202c"
        theme_border = "#4a3a20" if is_spot else "#3a4455"
        theme_muted = "#b79f66" if is_spot else "#98a2b3"
        theme_accent = "#d4af37" if is_spot else "#c0c0c0"
        theme_topbar_bg = "rgba(16, 13, 7, 0.94)" if is_spot else "rgba(9, 12, 18, 0.94)"
        theme_brand = "#f2dd9f" if is_spot else "#d8dde6"
        theme_tab = "#c8ac60" if is_spot else "#aab3c2"
        theme_tab_active = "#f3e2b5" if is_spot else "#e5e7eb"
        theme_tab_hover_bg = "#2a210f" if is_spot else "#1a212f"
        theme_surface = "#130f08" if is_spot else "#0f141e"
        theme_secondary_bg = "#312511" if is_spot else "#2a3342"
        theme_secondary_text = "#f3e2b6" if is_spot else "#e5e7eb"
        template = """
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Trading Bots Dashboard</title>
          <style>
            :root {
              --bg: __THEME_BG__;
              --panel: __THEME_PANEL__;
              --panel-2: __THEME_PANEL2__;
              --text: __THEME_TEXT__;
              --muted: __THEME_MUTED__;
              --border: __THEME_BORDER__;
              --accent: __THEME_ACCENT__;
              --surface: __THEME_SURFACE__;
              --secondary-bg: __THEME_SECONDARY_BG__;
              --secondary-text: __THEME_SECONDARY_TEXT__;
              --success: #82d995;
              --danger: #ff9b9b;
            }
            * { box-sizing: border-box; }
            body {
              margin: 0;
              font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
              background: radial-gradient(circle at 8% -10%, rgba(95,69,20,0.25) 0%, transparent 42%), radial-gradient(circle at 92% -20%, rgba(63,46,13,0.28) 0%, transparent 48%), var(--bg);
              color: var(--text);
            }
            .container {
              max-width: 1100px;
              margin: 0 auto;
              padding: 24px;
            }
            .topbar {
              position: sticky;
              top: 0;
              z-index: 50;
              background: __THEME_TOPBAR_BG__;
              border-bottom: 1px solid var(--border);
              backdrop-filter: blur(6px);
            }
            .topbar-inner {
              max-width: 1100px;
              margin: 0 auto;
              padding: 0.9rem 2rem;
              display: flex;
              align-items: center;
              gap: 1rem;
            }
            .brand {
              font-weight: 700;
              color: __THEME_BRAND__;
              text-decoration: none;
              margin-right: auto;
            }
            .nav-links { display: flex; align-items: center; gap: 1rem; }
            .tab-link {
              color: __THEME_TAB__;
              text-decoration: none;
              padding: 0.4rem 0.65rem;
              border-radius: 8px;
              border: 1px solid transparent;
            }
            .tab-link:hover, .tab-link.active {
              color: __THEME_TAB_ACTIVE__;
              border-color: var(--border);
              background: __THEME_TAB_HOVER_BG__;
            }
            h1 {
              margin: 0 0 18px 0;
              font-size: 1.8rem;
            }
            .toolbar {
              display: flex;
              gap: 12px;
              flex-wrap: wrap;
              align-items: center;
              margin-bottom: 16px;
            }
            .search-input {
              flex: 1 1 300px;
              border: 1px solid var(--border);
              background: var(--surface);
              color: var(--text);
              border-radius: 10px;
              padding: 10px 12px;
              outline: none;
            }
            .search-input:focus {
              border-color: var(--accent);
              box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.2);
            }
            .btn {
              border: 1px solid var(--border);
              border-radius: 10px;
              padding: 10px 14px;
              font-weight: 600;
              cursor: pointer;
            }
            .btn-primary {
              background: var(--accent);
              color: #111;
            }
            .btn-primary:hover { filter: brightness(1.06); }
            .btn-secondary {
              background: var(--secondary-bg);
              color: var(--secondary-text);
            }
            .table-wrap {
              overflow-x: auto;
              border: 1px solid var(--border);
              border-radius: 12px;
            }
            table {
              width: 100%;
              border-collapse: collapse;
              background: var(--panel);
            }
            th, td {
              text-align: left;
              padding: 12px;
              border-bottom: 1px solid var(--border);
              white-space: nowrap;
            }
            th { background: var(--panel-2); font-weight: 600; }
            .status-running { color: var(--success); font-weight: 600; }
            .status-stopped { color: var(--danger); font-weight: 600; }
            .btn-small { padding: 6px 10px; font-size: 0.86rem; margin-right: 6px; }
            .btn-start { background: #1f5533; color: #dcfce7; }
            .btn-stop { background: #6f2121; color: #fee2e2; }
            .muted { color: var(--muted); font-size: 0.9rem; margin-top: 10px; }
            .clickable-row { cursor: pointer; }
            .clickable-row:hover { background: rgba(255,255,255,0.04); }
            .p-up-cell { font-variant-numeric: tabular-nums; }
            .flash-up { animation: flash-up 0.9s ease-out; }
            .flash-down { animation: flash-down 0.9s ease-out; }
            @keyframes flash-up {
              0% { background: rgba(56, 214, 130, 0.35); }
              100% { background: transparent; }
            }
            @keyframes flash-down {
              0% { background: rgba(240, 82, 82, 0.35); }
              100% { background: transparent; }
            }
            .modal-backdrop {
              display: none;
              position: fixed;
              inset: 0;
              background: rgba(0,0,0,0.6);
              z-index: 60;
              align-items: center;
              justify-content: center;
              padding: 18px;
            }
            .modal-panel {
              width: min(980px, 100%);
              background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
              border: 1px solid var(--border);
              border-radius: 16px;
              padding: 20px;
            }
            .modal-grid {
              display: grid;
              grid-template-columns: repeat(2, minmax(220px,1fr));
              gap: 12px 16px;
            }
            .field-label { display: block; color: var(--muted); font-size: 0.95rem; }
            .field-label .search-input { margin-top: 6px; width: 100%; }
            .full-width { grid-column: 1 / span 2; }
            .modal-actions {
              display: grid;
              grid-template-columns: 1fr 1fr;
              gap: 10px;
              margin-top: 14px;
            }
            .metrics-grid {
              display: grid;
              grid-template-columns: repeat(3, minmax(160px, 1fr));
              gap: 8px;
              margin: 12px 0 14px;
            }
            .metric-card {
              border: 1px solid var(--border);
              border-radius: 10px;
              padding: 8px 10px;
              background: var(--surface);
            }
            .metric-label { font-size: 0.78rem; color: var(--muted); margin-bottom: 2px; }
            .metric-value { font-size: 0.96rem; font-weight: 600; }
            .trades-table-wrap {
              max-height: 240px;
              overflow: auto;
              border: 1px solid var(--border);
              border-radius: 10px;
            }
            .context-menu {
              display: none;
              position: fixed;
              z-index: 90;
              background: var(--panel-2);
              border: 1px solid var(--border);
              border-radius: 10px;
              min-width: 150px;
              box-shadow: 0 8px 26px rgba(0,0,0,0.35);
              padding: 6px;
            }
            .context-item {
              display: block;
              width: 100%;
              text-align: left;
              border: 0;
              border-radius: 8px;
              background: transparent;
              color: var(--text);
              padding: 8px 10px;
              cursor: pointer;
            }
            .context-item:hover { background: var(--surface); }
          </style>
        </head>
        <body>
          <nav class="topbar">
            <div class="topbar-inner">
              <a href="__HOME_HREF__" class="brand">__BRAND_LABEL__</a>
              <div class="nav-links">
                <a href="__HOME_HREF__" class="tab-link">Model</a>
                <a href="__MANAGE_HREF__" class="tab-link">Manage Models</a>
                <a href="__RUN_MODELS_HREF__" class="tab-link">Run Models</a>
                <a href="__BOTS_HREF__" class="tab-link active">Bots</a>
                <a href="__MODE_SWITCH_HREF__" class="tab-link">__MODE_SWITCH_LABEL__</a>
              </div>
            </div>
          </nav>
          <div class="container">
            <h1>Trading Bots Dashboard</h1>
            <div class="toolbar">
              <input id="searchInput" class="search-input" type="text" placeholder="Search bots by name..." />
              <button id="createBotButton" class="btn btn-primary">Create New Bot</button>
            </div>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Status (Running/Stopped)</th>
                    <th>Bid/Ask</th>
                    <th>Day PnL</th>
                    <th>Total PnL</th>
                    <th>Position</th>
                    <th>Cash</th>
                    <th>P(Up)</th>
                    <th>Actions (Start/Stop)</th>
                  </tr>
                </thead>
                <tbody id="botsTableBody"></tbody>
              </table>
            </div>
            <p id="botsEmptyState" class="muted" style="display:none;">No bots match that search.</p>
            <div class="muted">Click a bot row for details. Right-click a bot row to edit or delete. Auto-refreshes every 5 seconds.</div>
          </div>
          <div id="createBotModal" class="modal-backdrop">
            <div class="modal-panel">
              <h2 style="margin:0 0 10px 0;">Create Paper Trading Bot</h2>
              <p class="muted" style="margin:0 0 12px;">Preset-style bot configuration using the same app theme.</p>
              <div class="modal-grid">
                <label class="field-label">Model<select id="botModel" class="search-input"></select></label>
                <label class="field-label">Ticker<input id="botTicker" class="search-input" value="AAPL" /></label>
                <label class="field-label">Candle Time<select id="botTimeframe" class="search-input"></select></label>
                <label class="field-label" id="botDailyBuyTimingWrap">Daily Buy Timing
                  <select id="botDailyBuyTiming" class="search-input">
                    <option value="start_of_day">Beginning of day</option>
                    <option value="end_of_day">Last minute (end of day)</option>
                  </select>
                </label>
                <label class="field-label">Intraday Trade Interval
                  <select id="botIntradayTradeInterval" class="search-input">
                    <option value="unlimited">Unlimited</option>
                    <option value="5m">Every 5 minutes</option>
                    <option value="10m">Every 10 minutes</option>
                    <option value="15m">Every 15 minutes</option>
                    <option value="1h">Every hour</option>
                  </select>
                </label>
                <label class="field-label">Starting Money<input id="botStartingMoney" class="search-input" type="number" min="100" step="100" value="10000" /></label>
                <label class="field-label">BUY if P(Up) &gt;<input id="botBuyThreshold" class="search-input" type="number" min="0" max="1" step="0.01" value="0.60" /></label>
                <label class="field-label">SELL if P(Up) &lt;<input id="botSellThreshold" class="search-input" type="number" min="0" max="1" step="0.01" value="0.40" /></label>
                <label class="field-label">Stop Loss Strategy
                  <select id="botStopLossStrategy" class="search-input">
                    <option value="none">None</option>
                    <option value="atr">Volatility Buffer (ATR-Based)</option>
                    <option value="model_invalidation">Model Invalidation (MAE-Linked)</option>
                    <option value="time_decay">Time-Decay (Temporal Exit)</option>
                    <option value="fixed_percentage">Fixed Percentage</option>
                    <option value="trailing_stop">Trailing Stop Loss</option>
                  </select>
                </label>
                <label class="field-label" id="botFixedStopPctWrap">Fixed Stop Loss %<input id="botFixedStopPct" class="search-input" type="number" min="0.01" step="0.1" value="2.0" /></label>
                <label class="field-label">Take Profit %<input id="botTakeProfit" class="search-input" type="number" min="0" max="1" step="0.001" value="0.05" /></label>
                <label class="field-label full-width">Bot Name (optional)<input id="botName" class="search-input" /></label>
              </div>
              <p class="muted" style="margin:10px 0 14px;">Paper mode uses real bid/ask quotes (Questrade API) with zero commissions and places simulated orders only during U.S. market hours.</p>
              <div class="modal-actions">
                <button id="submitCreateBot" class="btn btn-primary">Start Bot</button>
                <button id="cancelCreateBot" class="btn btn-secondary">Cancel</button>
              </div>
            </div>
          </div>
          <div id="botDetailModal" class="modal-backdrop">
            <div class="modal-panel">
              <h2 id="botDetailTitle" style="margin:0;">Bot Details</h2>
              <p id="botDetailMeta" class="muted" style="margin:6px 0 0;"></p>
              <div id="botMetricsGrid" class="metrics-grid"></div>
              <h3 style="margin:8px 0;">Trade History</h3>
              <div class="trades-table-wrap">
                <table>
                  <thead><tr><th>Time</th><th>Side</th><th>Price</th><th>Size</th><th>PnL</th></tr></thead>
                  <tbody id="botTradeHistoryBody"></tbody>
                </table>
              </div>
              <div class="modal-actions" style="margin-top:12px;">
                <button id="closeBotDetail" class="btn btn-secondary">Close</button>
              </div>
            </div>
          </div>
          <div id="editBotModal" class="modal-backdrop">
            <div class="modal-panel">
              <h2 style="margin:0 0 10px 0;">Edit Bot Settings</h2>
              <div class="modal-grid">
                <label class="field-label">Bot Name<input id="editBotName" class="search-input" /></label>
                <label class="field-label">Ticker<input id="editBotTicker" class="search-input" /></label>
                <label class="field-label">Candle Time<input id="editBotTimeframe" class="search-input" /></label>
                <label class="field-label">Trade Size<input id="editBotTradeSize" class="search-input" type="number" min="0.01" step="0.01" /></label>
                <label class="field-label">BUY if P(Up) &gt;<input id="editBotBuyThreshold" class="search-input" type="number" min="0" max="1" step="0.01" /></label>
                <label class="field-label">SELL if P(Up) &lt;<input id="editBotSellThreshold" class="search-input" type="number" min="0" max="1" step="0.01" /></label>
                <label class="field-label">Daily Buy Timing
                  <select id="editBotDailyBuyTiming" class="search-input">
                    <option value="start_of_day">Beginning of day</option>
                    <option value="end_of_day">Last minute (end of day)</option>
                  </select>
                </label>
                <label class="field-label">Intraday Trade Interval
                  <select id="editBotIntradayTradeInterval" class="search-input">
                    <option value="unlimited">Unlimited</option>
                    <option value="5m">Every 5 minutes</option>
                    <option value="10m">Every 10 minutes</option>
                    <option value="15m">Every 15 minutes</option>
                    <option value="1h">Every hour</option>
                  </select>
                </label>
                <label class="field-label">Fixed Stop Loss %<input id="editBotFixedStopPct" class="search-input" type="number" min="0.01" step="0.1" /></label>
                <label class="field-label">Take Profit %<input id="editBotTakeProfit" class="search-input" type="number" min="0" step="0.001" /></label>
              </div>
              <div class="modal-actions">
                <button id="submitEditBot" class="btn btn-primary">Save Changes</button>
                <button id="cancelEditBot" class="btn btn-secondary">Cancel</button>
              </div>
            </div>
          </div>
          <div id="botContextMenu" class="context-menu">
            <button id="contextEditBot" class="context-item">Edit settings</button>
            <button id="contextDeleteBot" class="context-item">Delete bot</button>
          </div>
          <script>
            const tableBody = document.getElementById("botsTableBody");
            const searchInput = document.getElementById("searchInput");
            const botsEmptyState = document.getElementById("botsEmptyState");
            const createBotButton = document.getElementById("createBotButton");
            const createBotModal = document.getElementById("createBotModal");
            const cancelCreateBot = document.getElementById("cancelCreateBot");
            const submitCreateBot = document.getElementById("submitCreateBot");
            const botDetailModal = document.getElementById("botDetailModal");
            const botDetailTitle = document.getElementById("botDetailTitle");
            const botDetailMeta = document.getElementById("botDetailMeta");
            const botMetricsGrid = document.getElementById("botMetricsGrid");
            const botTradeHistoryBody = document.getElementById("botTradeHistoryBody");
            const closeBotDetail = document.getElementById("closeBotDetail");
            const editBotModal = document.getElementById("editBotModal");
            const submitEditBot = document.getElementById("submitEditBot");
            const cancelEditBot = document.getElementById("cancelEditBot");
            const botContextMenu = document.getElementById("botContextMenu");
            const contextEditBot = document.getElementById("contextEditBot");
            const contextDeleteBot = document.getElementById("contextDeleteBot");
            let allBots = [];
            let selectedBotId = null;
            let previousPUpByBotId = {};

            const formatCurrency = (value) => {
              const number = Number(value || 0);
              return number.toLocaleString(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 2 });
            };
            const formatBidAsk = (bid, ask) => {
              const hasBid = Number.isFinite(Number(bid));
              const hasAsk = Number.isFinite(Number(ask));
              if (!hasBid && !hasAsk) return "—";
              if (!hasBid) return `— / ${Number(ask).toFixed(2)}`;
              if (!hasAsk) return `${Number(bid).toFixed(2)} / —`;
              return `${Number(bid).toFixed(2)} / ${Number(ask).toFixed(2)}`;
            };
            const formatPercent = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;
            const hideContextMenu = () => { botContextMenu.style.display = "none"; };
            const showContextMenu = (x, y) => {
              botContextMenu.style.left = `${x}px`;
              botContextMenu.style.top = `${y}px`;
              botContextMenu.style.display = "block";
            };
            const metricCard = (label, value) => `
              <div class="metric-card"><div class="metric-label">${label}</div><div class="metric-value">${value}</div></div>
            `;
            const renderHoldTimeBoxplot = (stats) => {
              const count = Number(stats.count || 0);
              if (count <= 0) return "<p class='muted'>No active hold streaks in this test window.</p>";
              const min = Number(stats.min || 0);
              const q1 = Number(stats.q1 || 0);
              const median = Number(stats.median || 0);
              const q3 = Number(stats.q3 || 0);
              const max = Number(stats.max || 0);
              const span = Math.max(1e-9, max - min);
              const scale = (value) => 22 + ((value - min) / span) * 236;
              const minX = scale(min);
              const q1X = scale(q1);
              const medX = scale(median);
              const q3X = scale(q3);
              const maxX = scale(max);
              const label = (x, y, text) => `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" fill="#cfd8e6" font-size="10" text-anchor="middle">${text}</text>`;
              return `<div style="display:inline-flex; flex-direction:column; gap:0.35rem;">
                <svg width="280" height="96" viewBox="0 0 280 96" xmlns="http://www.w3.org/2000/svg">
                  <line x1="${minX.toFixed(1)}" y1="48" x2="${maxX.toFixed(1)}" y2="48" stroke="#8ca0bf" stroke-width="1.2"/>
                  <line x1="${minX.toFixed(1)}" y1="39" x2="${minX.toFixed(1)}" y2="57" stroke="#8ca0bf" stroke-width="1.2"/>
                  <line x1="${maxX.toFixed(1)}" y1="39" x2="${maxX.toFixed(1)}" y2="57" stroke="#8ca0bf" stroke-width="1.2"/>
                  <rect x="${q1X.toFixed(1)}" y="34" width="${Math.max(2, q3X - q1X).toFixed(1)}" height="28" fill="none" stroke="#8ca0bf" stroke-width="1.4" rx="4"/>
                  <line x1="${medX.toFixed(1)}" y1="31" x2="${medX.toFixed(1)}" y2="65" stroke="#cfd8e6" stroke-width="1.5"/>
                  ${label(minX, 23, `min ${min.toFixed(1)}`)}
                  ${label(q1X, 18, `q1 ${q1.toFixed(1)}`)}
                  ${label(medX, 76, `med ${median.toFixed(1)}`)}
                  ${label(q3X, 18, `q3 ${q3.toFixed(1)}`)}
                  ${label(maxX, 23, `max ${max.toFixed(1)}`)}
                </svg>
                <span class="muted" style="font-size:0.82rem;">Hold streak samples: ${count}</span>
              </div>`;
            };
            const openBotDetails = async (botId) => {
              hideContextMenu();
              const response = await fetch(`/bots/${botId}`);
              if (!response.ok) {
                alert("Failed to load bot details.");
                return;
              }
              const bot = await response.json();
              const metrics = bot.metrics || {};
              const holdStats = metrics.hold_time_stats || {};
              const holdChart = renderHoldTimeBoxplot(holdStats);
              botDetailTitle.textContent = `Bot Details: ${String(bot.name || "")}`;
              botDetailMeta.textContent = `Model ${String(bot.model_name || "")} • ${String(bot.ticker || "")} • ${String(bot.timeframe || "")}`;
              botMetricsGrid.innerHTML = [
                metricCard("Max Drawdown", formatCurrency(metrics.max_drawdown)),
                metricCard("Avg Drawdown", formatCurrency(metrics.avg_drawdown)),
                metricCard("PnL (24h)", formatCurrency(metrics.pnl_day)),
                metricCard("PnL (7d)", formatCurrency(metrics.pnl_week)),
                metricCard("PnL (30d)", formatCurrency(metrics.pnl_month)),
                metricCard("Trades", String(metrics.trade_count || 0)),
                metricCard("Sharpe", Number(metrics.sharpe || 0).toFixed(3)),
                metricCard("Win Rate / Trades", `${(Number(metrics.win_rate || 0) * 100).toFixed(2)}% / ${Number(metrics.closed_trade_count || 0)}`),
                metricCard("Average Gain per Trade", `${(Number(metrics.avg_gain_per_trade || 0) * 100).toFixed(4)}%`),
                metricCard("Stop-loss Exits", String(metrics.stop_loss_exits || 0)),
                `<div style="grid-column:1 / -1;"><p class="muted" style="margin-bottom:0.35rem;">Hold Time Distribution (bars/candles)</p>${holdChart}</div>`,
              ].join("");
              const trades = Array.isArray(bot.trades) ? bot.trades : [];
              botTradeHistoryBody.innerHTML = "";
              if (!trades.length) {
                botTradeHistoryBody.innerHTML = "<tr><td colspan='5' class='muted'>No trades yet.</td></tr>";
              } else {
                trades.slice().reverse().forEach((trade) => {
                  const tr = document.createElement("tr");
                  tr.innerHTML = `
                    <td>${String(trade.timestamp || "")}</td>
                    <td>${String(trade.side || "")}</td>
                    <td>${Number(trade.price || 0).toFixed(4)}</td>
                    <td>${Number(trade.size || 0).toFixed(2)}</td>
                    <td>${formatCurrency(trade.pnl || 0)}</td>
                  `;
                  botTradeHistoryBody.appendChild(tr);
                });
              }
              botDetailModal.style.display = "flex";
            };

            const renderRows = () => {
              const search = searchInput.value.trim().toLowerCase();
              const filtered = allBots.filter((bot) => bot.search_name.includes(search));
              tableBody.innerHTML = "";
              filtered.forEach((bot) => {
                const status = String(bot.status || "stopped").toLowerCase();
                const row = document.createElement("tr");
                row.className = "clickable-row";
                row.innerHTML = `
                  <td></td>
                  <td class="${status === "running" ? "status-running" : "status-stopped"}"></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td></td>
                  <td class="p-up-cell"></td>
                  <td></td>
                  <td></td>
                `;
                row.children[0].textContent = String(bot.name || "");
                row.children[1].textContent = status === "running" ? "Running" : "Stopped";
                row.children[2].textContent = formatBidAsk(bot.last_polled_bid, bot.last_polled_ask);
                row.children[3].textContent = formatCurrency(bot.day_pnl);
                row.children[4].textContent = formatCurrency(bot.total_pnl);
                row.children[5].textContent = Number(bot.position || 0).toFixed(2);
                row.children[6].textContent = formatCurrency(bot.cash);
                const pUpCell = row.children[7];
                const currentPUp = Number(bot.p_up || 0);
                pUpCell.textContent = formatPercent(currentPUp);
                const prevPUp = previousPUpByBotId[bot.id];
                if (typeof prevPUp === "number") {
                  if (currentPUp > prevPUp) {
                    pUpCell.classList.add("flash-up");
                  } else if (currentPUp < prevPUp) {
                    pUpCell.classList.add("flash-down");
                  }
                }
                row.addEventListener("click", () => openBotDetails(bot.id));
                row.addEventListener("contextmenu", (event) => {
                  event.preventDefault();
                  selectedBotId = bot.id;
                  showContextMenu(event.clientX, event.clientY);
                });

                const actionsCell = row.children[8];
                const startBtn = document.createElement("button");
                startBtn.className = "btn btn-small btn-start";
                startBtn.textContent = "Start";
                startBtn.disabled = status === "running";
                startBtn.addEventListener("click", (event) => {
                  event.stopPropagation();
                  botAction(`/bots/start/${bot.id}`);
                });

                const stopBtn = document.createElement("button");
                stopBtn.className = "btn btn-small btn-stop";
                stopBtn.textContent = "Stop";
                stopBtn.disabled = status !== "running";
                stopBtn.addEventListener("click", (event) => {
                  event.stopPropagation();
                  botAction(`/bots/stop/${bot.id}`);
                });

                actionsCell.appendChild(startBtn);
                actionsCell.appendChild(stopBtn);
                tableBody.appendChild(row);
              });
              botsEmptyState.style.display = filtered.length ? "none" : "block";
            };

            const loadBots = async () => {
              const response = await fetch("/api/bots");
              if (!response.ok) {
                throw new Error("Failed to load bots");
              }
              const payload = await response.json();
              previousPUpByBotId = Object.fromEntries(
                allBots.map((bot) => [bot.id, Number(bot.p_up || 0)]),
              );
              allBots = payload.map((bot) => ({
                ...bot,
                search_name: String(bot.name || "").toLowerCase(),
              }));
              renderRows();
            };

            const botAction = async (url) => {
              const response = await fetch(url, { method: "POST" });
              if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                alert(payload.error || "Action failed.");
                return;
              }
              await loadBots();
            };
            const deleteSelectedBot = async () => {
              if (!selectedBotId) return;
              const confirmed = window.confirm("Delete this bot? This cannot be undone.");
              if (!confirmed) return;
              const response = await fetch(`/bots/${selectedBotId}`, { method: "DELETE" });
              if (!response.ok) {
                const payload = await response.json().catch(() => ({}));
                alert(payload.error || "Delete failed.");
                return;
              }
              hideContextMenu();
              await loadBots();
            };

            const botModel = document.getElementById("botModel");
            const botTicker = document.getElementById("botTicker");
            const botTimeframe = document.getElementById("botTimeframe");
            const botDailyBuyTimingWrap = document.getElementById("botDailyBuyTimingWrap");
            const botDailyBuyTiming = document.getElementById("botDailyBuyTiming");
            const botIntradayTradeInterval = document.getElementById("botIntradayTradeInterval");
            const botStartingMoney = document.getElementById("botStartingMoney");
            const botBuyThreshold = document.getElementById("botBuyThreshold");
            const botSellThreshold = document.getElementById("botSellThreshold");
            const botStopLossStrategy = document.getElementById("botStopLossStrategy");
            const botFixedStopPctWrap = document.getElementById("botFixedStopPctWrap");
            const botFixedStopPct = document.getElementById("botFixedStopPct");
            const botTakeProfit = document.getElementById("botTakeProfit");
            const botName = document.getElementById("botName");
            const syncDailyBuyTimingVisibility = () => {
              const show = String(botTimeframe.value || "").toLowerCase() === "1d";
              if (botDailyBuyTimingWrap) botDailyBuyTimingWrap.style.display = show ? "block" : "none";
              if (botDailyBuyTiming) botDailyBuyTiming.disabled = !show;
            };
            const syncBotStopLossFields = () => {
              const strategy = botStopLossStrategy ? botStopLossStrategy.value : "none";
              const showFixed = strategy === "fixed_percentage" || strategy === "trailing_stop";
              if (botFixedStopPctWrap) botFixedStopPctWrap.style.display = showFixed ? "block" : "none";
              if (botFixedStopPct) botFixedStopPct.disabled = !showFixed;
            };

            const loadBotFormOptions = async () => {
              const mode = location.pathname.startsWith("/spot") ? "spot" : "options";
              const response = await fetch(`/api/bots/form-options?mode=${mode}`);
              if (!response.ok) throw new Error("Failed to load bot options.");
              const options = await response.json();
              botModel.innerHTML = "";
              (options.models || []).forEach((name) => {
                const opt = document.createElement("option");
                opt.value = name;
                opt.textContent = name;
                botModel.appendChild(opt);
              });
              if (!botModel.options.length) {
                const opt = document.createElement("option");
                opt.value = "demo-model";
                opt.textContent = "demo-model";
                botModel.appendChild(opt);
              }
              botTimeframe.innerHTML = "";
              (options.timeframes || ["1m"]).forEach((tf) => {
                const opt = document.createElement("option");
                opt.value = tf;
                opt.textContent = tf;
                botTimeframe.appendChild(opt);
              });
              botDailyBuyTiming.innerHTML = "";
              (options.daily_buy_timing_options || []).forEach((item) => {
                const opt = document.createElement("option");
                opt.value = String(item.value || "start_of_day");
                opt.textContent = String(item.label || item.value || "start_of_day");
                botDailyBuyTiming.appendChild(opt);
              });
              if (!botDailyBuyTiming.options.length) {
                const optStart = document.createElement("option");
                optStart.value = "start_of_day";
                optStart.textContent = "Beginning of day";
                botDailyBuyTiming.appendChild(optStart);
                const optEnd = document.createElement("option");
                optEnd.value = "end_of_day";
                optEnd.textContent = "Last minute (end of day)";
                botDailyBuyTiming.appendChild(optEnd);
              }
              botIntradayTradeInterval.innerHTML = "";
              (options.intraday_trade_interval_options || []).forEach((item) => {
                const opt = document.createElement("option");
                opt.value = String(item.value || "unlimited");
                opt.textContent = String(item.label || item.value || "Unlimited");
                botIntradayTradeInterval.appendChild(opt);
              });
              if (!botIntradayTradeInterval.options.length) {
                const opt = document.createElement("option");
                opt.value = "unlimited";
                opt.textContent = "Unlimited";
                botIntradayTradeInterval.appendChild(opt);
              }
              botTicker.value = options.default_ticker || "AAPL";
              syncDailyBuyTimingVisibility();
            };

            const openEditModal = async (botId) => {
              hideContextMenu();
              const response = await fetch(`/bots/${botId}`);
              if (!response.ok) {
                alert("Failed to load bot settings.");
                return;
              }
              const bot = await response.json();
              selectedBotId = botId;
              document.getElementById("editBotName").value = String(bot.name || "");
              document.getElementById("editBotTicker").value = String(bot.ticker || "");
              document.getElementById("editBotTimeframe").value = String(bot.timeframe || "");
              document.getElementById("editBotTradeSize").value = Number(bot.trade_size || 1).toString();
              document.getElementById("editBotBuyThreshold").value = Number(bot.buy_threshold || 0.6).toString();
              document.getElementById("editBotSellThreshold").value = Number(bot.sell_threshold || 0.4).toString();
              document.getElementById("editBotDailyBuyTiming").value = String(bot.daily_buy_timing || "start_of_day");
              document.getElementById("editBotIntradayTradeInterval").value = String(bot.intraday_trade_interval || "unlimited");
              document.getElementById("editBotFixedStopPct").value = (Number(bot.stop_loss || 0.02) * 100).toString();
              document.getElementById("editBotTakeProfit").value = Number(bot.take_profit || 0.05).toString();
              editBotModal.style.display = "flex";
            };

            createBotButton.addEventListener("click", async () => {
              await loadBotFormOptions();
              syncBotStopLossFields();
              createBotModal.style.display = "flex";
            });
            cancelCreateBot.addEventListener("click", () => { createBotModal.style.display = "none"; });
            closeBotDetail.addEventListener("click", () => { botDetailModal.style.display = "none"; });
            cancelEditBot.addEventListener("click", () => { editBotModal.style.display = "none"; });
            contextEditBot.addEventListener("click", () => {
              if (selectedBotId) openEditModal(selectedBotId);
            });
            contextDeleteBot.addEventListener("click", () => {
              deleteSelectedBot().catch(() => alert("Failed to delete bot."));
            });
            document.addEventListener("click", (event) => {
              if (!botContextMenu.contains(event.target)) {
                hideContextMenu();
              }
            });
            document.addEventListener("scroll", hideContextMenu, true);
            if (botStopLossStrategy) {
              botStopLossStrategy.addEventListener("change", syncBotStopLossFields);
            }
            if (botTimeframe) {
              botTimeframe.addEventListener("change", syncDailyBuyTimingVisibility);
            }

            submitCreateBot.addEventListener("click", async () => {
              const mode = location.pathname.startsWith("/spot") ? "spot" : "options";
              const payload = {
                model: botModel.value,
                ticker: botTicker.value,
                timeframe: botTimeframe.value,
                mode,
                starting_money: Number(botStartingMoney.value),
                buy_threshold: Number(botBuyThreshold.value),
                sell_threshold: Number(botSellThreshold.value),
                daily_buy_timing: botDailyBuyTiming.value,
                intraday_trade_interval: botIntradayTradeInterval.value,
                stop_loss_strategy: botStopLossStrategy.value,
                fixed_stop_pct: Number(botFixedStopPct.value),
                take_profit: Number(botTakeProfit.value),
                name: botName.value,
              };
              const response = await fetch("/bots/create", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });
              if (!response.ok) {
                const payloadErr = await response.json().catch(() => ({}));
                alert(payloadErr.error || "Failed to create bot.");
                return;
              }
              createBotModal.style.display = "none";
              botName.value = "";
              await loadBots();
            });

            submitEditBot.addEventListener("click", async () => {
              if (!selectedBotId) return;
              const payload = {
                name: document.getElementById("editBotName").value,
                ticker: document.getElementById("editBotTicker").value,
                timeframe: document.getElementById("editBotTimeframe").value,
                trade_size: Number(document.getElementById("editBotTradeSize").value),
                buy_threshold: Number(document.getElementById("editBotBuyThreshold").value),
                sell_threshold: Number(document.getElementById("editBotSellThreshold").value),
                daily_buy_timing: document.getElementById("editBotDailyBuyTiming").value,
                intraday_trade_interval: document.getElementById("editBotIntradayTradeInterval").value,
                fixed_stop_pct: Number(document.getElementById("editBotFixedStopPct").value),
                take_profit: Number(document.getElementById("editBotTakeProfit").value),
                stop_loss_strategy: "fixed_percentage",
              };
              const response = await fetch(`/bots/${selectedBotId}/settings`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });
              if (!response.ok) {
                const payloadErr = await response.json().catch(() => ({}));
                alert(payloadErr.error || "Failed to update bot.");
                return;
              }
              editBotModal.style.display = "none";
              await loadBots();
            });

            searchInput.addEventListener("input", renderRows);
            loadBots().catch((err) => {
              tableBody.innerHTML = `<tr><td colspan="9">${String(err.message || err)}</td></tr>`;
            });
            setInterval(() => { loadBots().catch(() => {}); }, 5000);
          </script>
        </body>
        </html>
        """
        return (
            template.replace("__HOME_HREF__", home_href)
            .replace("__MANAGE_HREF__", manage_href)
            .replace("__RUN_MODELS_HREF__", run_models_href)
            .replace("__BOTS_HREF__", bots_href)
            .replace("__MODE_SWITCH_HREF__", mode_switch_href)
            .replace("__MODE_SWITCH_LABEL__", mode_switch_label)
            .replace("__BRAND_LABEL__", brand_label)
            .replace("__THEME_BG__", theme_bg)
            .replace("__THEME_TEXT__", theme_text)
            .replace("__THEME_PANEL__", theme_panel)
            .replace("__THEME_PANEL2__", theme_panel2)
            .replace("__THEME_BORDER__", theme_border)
            .replace("__THEME_MUTED__", theme_muted)
            .replace("__THEME_ACCENT__", theme_accent)
            .replace("__THEME_TOPBAR_BG__", theme_topbar_bg)
            .replace("__THEME_BRAND__", theme_brand)
            .replace("__THEME_TAB__", theme_tab)
            .replace("__THEME_TAB_ACTIVE__", theme_tab_active)
            .replace("__THEME_TAB_HOVER_BG__", theme_tab_hover_bg)
            .replace("__THEME_SURFACE__", theme_surface)
            .replace("__THEME_SECONDARY_BG__", theme_secondary_bg)
            .replace("__THEME_SECONDARY_TEXT__", theme_secondary_text)
        )

    @app.route("/bots/create", methods=["POST"])
    def create_bot_endpoint() -> object:
        try:
            config = _parse_bot_create_payload(request.get_json(silent=True))
            bot = create_bot(config)
            return jsonify(_bot_payload(bot)), 201
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/bots/start/<bot_id>", methods=["POST"])
    def start_bot_endpoint(bot_id: str) -> object:
        try:
            return jsonify(_bot_payload(start_bot(bot_id)))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/bots/stop/<bot_id>", methods=["POST"])
    def stop_bot_endpoint(bot_id: str) -> object:
        try:
            return jsonify(_bot_payload(stop_bot(bot_id)))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/bots/<bot_id>", methods=["GET"])
    def get_bot_endpoint(bot_id: str) -> object:
        bot = get_bot(bot_id)
        if bot is None:
            return jsonify({"error": f"Bot '{bot_id}' not found"}), 404
        return jsonify(_bot_details_payload(bot))

    @app.route("/bots/<bot_id>/settings", methods=["PATCH"])
    def update_bot_settings_endpoint(bot_id: str) -> object:
        bot = get_bot(bot_id)
        if bot is None:
            return jsonify({"error": f"Bot '{bot_id}' not found"}), 404
        try:
            updates = _parse_bot_update_payload(request.get_json(silent=True), existing_bot=bot)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        for field, value in updates.items():
            setattr(bot, field, value)
        persist_bot(bot)
        return jsonify(_bot_details_payload(bot))

    @app.route("/bots/<bot_id>", methods=["DELETE"])
    def delete_bot_endpoint(bot_id: str) -> object:
        try:
            delete_bot(bot_id)
            return jsonify({"ok": True})
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.route("/manage-models", methods=["GET", "POST"])
    @app.route("/spot/manage-models", methods=["GET", "POST"])
    def manage_models() -> str:
        is_spot = request.path.startswith("/spot")
        is_mobile_ui = is_mobile_request(request)
        mode_key = SPOT_MODE if is_spot else OPTIONS_MODE
        home_href = "/spot" if is_spot else "/"
        manage_href = "/spot/manage-models" if is_spot else "/manage-models"
        run_models_href = "/spot/run-models" if is_spot else "/run-models"
        bots_href = "/spot/bots" if is_spot else "/bots"
        mode_switch_href = "/" if is_spot else "/spot"
        mode_switch_label = "Switch to Options Mode" if is_spot else "Switch to Spot Mode"
        brand_label = "Quant Trader • Spot Mode" if is_spot else "Quant Trader • Options Mode"
        heading_label = "Manage Spot Models" if is_spot else "Manage Options Models"
        theme_bg = "#100d07" if is_spot else "#090c12"
        theme_text = "#efe0be" if is_spot else "#c6ccd7"
        theme_panel = "#19130b" if is_spot else "#121722"
        theme_panel2 = "#221b10" if is_spot else "#1a202c"
        theme_border = "#4a3a20" if is_spot else "#3a4455"
        theme_muted = "#b79f66" if is_spot else "#98a2b3"
        theme_accent = "#d4af37" if is_spot else "#c0c0c0"
        theme_topbar_bg = "rgba(16, 13, 7, 0.94)" if is_spot else "rgba(9, 12, 18, 0.94)"
        theme_brand = "#f2dd9f" if is_spot else "#d8dde6"
        theme_tab = "#c8ac60" if is_spot else "#aab3c2"
        theme_tab_active = "#f3e2b5" if is_spot else "#e5e7eb"
        theme_tab_hover_bg = "#2a210f" if is_spot else "#1a212f"
        theme_surface = "#130f08" if is_spot else "#0f141e"
        theme_context_bg = "#181108" if is_spot else "#141b28"
        theme_context_hover = "#2a200d" if is_spot else "#20293a"
        theme_glow_primary = "#5f4514" if is_spot else "#2d364a"
        theme_glow_secondary = "#3f2e0d" if is_spot else "#1f2636"
        theme_badge = "#f1cb6c" if is_spot else "#cfd5df"
        theme_table_head = "#e0c380" if is_spot else "#b5bfce"
        theme_link = "#e5c46a" if is_spot else "#d6d9df"
        theme_secondary_bg = "#312511" if is_spot else "#2a3342"
        theme_secondary_text = "#f3e2b6" if is_spot else "#e5e7eb"
        message_html = ""
        error_html = ""
        model_configs = load_model_configs(mode_key)
        saved_models = list_saved_models(mode_key)
        model_configs = {name: get_model_config(name, model_configs) for name in saved_models}

        if request.method == "POST":
            action = request.form.get("action", "").strip()
            model_name = request.form.get("model_name", "").strip()
            try:
                if action == "save_config":
                    if model_name not in saved_models:
                        raise ValueError("Model not found.")
                    ticker = request.form.get("ticker", "AAPL").upper().strip()
                    interval = request.form.get("interval", "1d").strip()
                    rows_raw = request.form.get("rows", "250").strip()
                    buy_raw = request.form.get("buy_threshold", "").strip()
                    sell_raw = request.form.get("sell_threshold", "").strip()
                    stop_loss_strategy = parse_stop_loss_strategy(request.form.get("stop_loss_strategy", StopLossStrategy.NONE.value))
                    fixed_stop_raw = request.form.get("fixed_stop_pct", "2.0").strip()
                    take_profit_raw = request.form.get("take_profit_pct", "").strip()
                    max_hold_raw = request.form.get("max_hold_bars", "").strip()
                    fixed_stop_pct = 2.0
                    if stop_loss_strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
                        fixed_stop_pct = validate_fixed_stop_pct(float(fixed_stop_raw or "2.0"))
                    take_profit_pct = 0.0 if not take_profit_raw else validate_take_profit_pct(float(take_profit_raw))
                    max_hold_bars = 0 if not max_hold_raw else validate_max_hold_bars(int(max_hold_raw))
                    include_in_run_all = request.form.get("include_in_run_all", "0") == "1"
                    rows = int(rows_raw)
                    buy_threshold, sell_threshold = parse_thresholds(buy_raw, sell_raw)
                    if interval not in ("1d", "1h", "15m", "5m"):
                        raise ValueError("Candle length must be one of: 1d, 1h, 15m, 5m.")
                    if rows < 50:
                        raise ValueError("Rows must be at least 50.")
                    model_configs[model_name] = {
                        "ticker": ticker,
                        "interval": interval,
                        "rows": rows,
                        "include_in_run_all": include_in_run_all,
                        "buy_threshold": buy_threshold,
                        "sell_threshold": sell_threshold,
                        "stop_loss_strategy": stop_loss_strategy.value,
                        "fixed_stop_pct": fixed_stop_pct,
                        "take_profit_pct": take_profit_pct,
                        "max_hold_bars": max_hold_bars,
                    }
                    save_model_configs(mode_key, model_configs)
                    message_html = f"<p style='color:#7bd88f;'><strong>Saved settings for:</strong> {model_name}</p>"
                elif action == "save_all_configs":
                    if not saved_models:
                        raise ValueError("No models found to update.")
                    ticker_raw = request.form.get("ticker", "").upper().strip()
                    interval_raw = request.form.get("interval", "").strip()
                    rows_raw = request.form.get("rows", "").strip()
                    buy_raw = request.form.get("buy_threshold", "").strip()
                    sell_raw = request.form.get("sell_threshold", "").strip()
                    stop_loss_strategy_raw = request.form.get("stop_loss_strategy", "").strip()
                    fixed_stop_raw = request.form.get("fixed_stop_pct", "").strip()
                    take_profit_raw = request.form.get("take_profit_pct", "").strip()
                    max_hold_raw = request.form.get("max_hold_bars", "").strip()
                    include_in_run_all_raw = request.form.get("include_in_run_all", "").strip()

                    updates: Dict[str, object] = {}
                    if ticker_raw:
                        updates["ticker"] = ticker_raw
                    if interval_raw:
                        if interval_raw not in ("1d", "1h", "15m", "5m"):
                            raise ValueError("Candle length must be one of: 1d, 1h, 15m, 5m.")
                        updates["interval"] = interval_raw
                    if rows_raw:
                        rows = int(rows_raw)
                        if rows < 50:
                            raise ValueError("Rows must be at least 50.")
                        updates["rows"] = rows
                    if buy_raw or sell_raw:
                        buy_threshold, sell_threshold = parse_thresholds(buy_raw, sell_raw)
                        updates["buy_threshold"] = buy_threshold
                        updates["sell_threshold"] = sell_threshold
                    if stop_loss_strategy_raw:
                        stop_loss_strategy = parse_stop_loss_strategy(stop_loss_strategy_raw)
                        updates["stop_loss_strategy"] = stop_loss_strategy.value
                        if stop_loss_strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
                            updates["fixed_stop_pct"] = validate_fixed_stop_pct(float(fixed_stop_raw or "2.0"))
                    if fixed_stop_raw and not stop_loss_strategy_raw:
                        updates["fixed_stop_pct"] = validate_fixed_stop_pct(float(fixed_stop_raw))
                    if take_profit_raw:
                        updates["take_profit_pct"] = validate_take_profit_pct(float(take_profit_raw))
                    if max_hold_raw:
                        updates["max_hold_bars"] = validate_max_hold_bars(int(max_hold_raw))
                    if include_in_run_all_raw in ("0", "1"):
                        updates["include_in_run_all"] = include_in_run_all_raw == "1"
                    if not updates:
                        raise ValueError("Enter at least one value to apply to all model presets.")
                    for saved_model_name in saved_models:
                        model_cfg = get_model_config(saved_model_name, model_configs)
                        model_cfg.update(updates)
                        model_configs[saved_model_name] = model_cfg
                    save_model_configs(mode_key, model_configs)
                    message_html = f"<p style='color:#7bd88f;'><strong>Saved settings for all models:</strong> {len(saved_models)} model(s) updated.</p>"
                elif action == "rename_model":
                    new_name = sanitize_model_name(request.form.get("new_name", "").strip())
                    if model_name not in saved_models:
                        raise ValueError("Model not found.")
                    if not new_name:
                        raise ValueError("New model name cannot be empty.")
                    if new_name in saved_models:
                        raise ValueError("A model with that name already exists.")
                    old_path = os.path.join(mode_model_dir(mode_key), f"{model_name}.json")
                    new_path = os.path.join(mode_model_dir(mode_key), f"{new_name}.json")
                    os.rename(old_path, new_path)
                    if model_name in model_configs:
                        model_configs[new_name] = model_configs.pop(model_name)
                    save_model_configs(mode_key, model_configs)
                    return redirect(manage_href)
                elif action == "delete_model":
                    if model_name not in saved_models:
                        raise ValueError("Model not found.")
                    path = os.path.join(mode_model_dir(mode_key), f"{model_name}.json")
                    if os.path.exists(path):
                        os.remove(path)
                    if model_name in model_configs:
                        model_configs.pop(model_name)
                        save_model_configs(mode_key, model_configs)
                    return redirect(manage_href)
                elif action == "delete_all_models":
                    deleted_count = 0
                    for saved_model_name in saved_models:
                        path = os.path.join(mode_model_dir(mode_key), f"{saved_model_name}.json")
                        if os.path.exists(path):
                            os.remove(path)
                            deleted_count += 1
                    if saved_models:
                        save_model_configs(mode_key, {})
                    message_html = (
                        "<p style='color:#7bd88f;'><strong>Deleted all models:</strong> "
                        f"{deleted_count} model(s) removed.</p>"
                    )
            except Exception as exc:
                error_html = f"<p style='color:#ff7b7b;'><strong>Error:</strong> {exc}</p>"
            saved_models = list_saved_models(mode_key)
            model_configs = load_model_configs(mode_key)
            model_configs = {name: get_model_config(name, model_configs) for name in saved_models}

        model_rows = ""
        for model_name in saved_models:
            cfg = get_model_config(model_name, model_configs)
            include_badge = "Included in Run All" if cfg.get("include_in_run_all", True) else "Excluded from Run All"
            model_rows += (
                f"<tr class='model-row' "
                f"data-model='{model_name}' "
                f"data-ticker='{cfg.get('ticker')}' "
                f"data-interval='{cfg.get('interval')}' "
                f"data-rows='{int(cfg.get('rows', 250))}' "
                f"data-include='{1 if cfg.get('include_in_run_all', True) else 0}' "
                f"data-buy='{float(cfg.get('buy_threshold', 0.6)):.2f}' "
                f"data-sell='{float(cfg.get('sell_threshold', 0.4)):.2f}' "
                f"data-stop-loss='{cfg.get('stop_loss_strategy', StopLossStrategy.NONE.value)}' "
                f"data-fixed-stop='{float(cfg.get('fixed_stop_pct', 2.0)):.2f}' "
                f"data-take-profit='{float(cfg.get('take_profit_pct', 0.0)):.2f}' "
                f"data-max-hold='{int(cfg.get('max_hold_bars', 0))}'>"
                f"<td>{model_name}</td>"
                f"<td>{cfg.get('ticker')}</td>"
                f"<td>{cfg.get('interval')}</td>"
                f"<td>{int(cfg.get('rows', 250))}</td>"
                f"<td>{float(cfg.get('buy_threshold', 0.6)):.2f}</td>"
                f"<td>{float(cfg.get('sell_threshold', 0.4)):.2f}</td>"
                f"<td>{cfg.get('stop_loss_strategy', StopLossStrategy.NONE.value)}</td>"
                f"<td>{float(cfg.get('fixed_stop_pct', 2.0)):.2f}%</td>"
                f"<td>{float(cfg.get('take_profit_pct', 0.0)):.2f}%</td>"
                f"<td>{int(cfg.get('max_hold_bars', 0))}</td>"
                f"<td>{include_badge}</td>"
                "</tr>"
            )

        return f"""
        <html>
          <head>
            <title>Manage Models</title>
            <meta name="viewport" content="width=device-width, initial-scale=1" />
          </head>
          <body class="{'mobile-ui' if is_mobile_ui else 'desktop-ui'}">
            <style>
              :root {{
                --bg: {theme_bg};
                --panel: {theme_panel};
                --panel-2: {theme_panel2};
                --border: {theme_border};
                --text: {theme_text};
                --muted: {theme_muted};
                --accent: {theme_accent};
              }}
              * {{ box-sizing: border-box; }}
              body {{ margin: 0; background: radial-gradient(circle at 8% -10%, {theme_glow_primary} 0%, transparent 42%), radial-gradient(circle at 92% -20%, {theme_glow_secondary} 0%, transparent 48%), var(--bg); color: var(--text); font-family: Inter, Segoe UI, Arial, sans-serif; }}
              .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}
              .topbar {{ position: sticky; top: 0; z-index: 50; background: {theme_topbar_bg}; border-bottom: 1px solid var(--border); backdrop-filter: blur(6px); }}
              .topbar-inner {{ max-width: 1100px; margin: 0 auto; padding: 0.9rem 2rem; display: flex; align-items: center; gap: 1rem; }}
              .brand {{ font-weight: 700; color: {theme_brand}; text-decoration: none; margin-right: auto; }}
              .mobile-menu-toggle {{ display: none; border: 1px solid var(--border); background: transparent; color: var(--text); border-radius: 8px; padding: 0.35rem 0.55rem; font-size: 1rem; line-height: 1; cursor: pointer; }}
              .mobile-menu-toggle:hover {{ background: {theme_tab_hover_bg}; }}
              .nav-links {{ display: flex; align-items: center; gap: 1rem; }}
              .tab-link {{ color: {theme_tab}; text-decoration: none; padding: 0.4rem 0.65rem; border-radius: 8px; border: 1px solid transparent; }}
              .tab-link:hover, .tab-link.active {{ color: {theme_tab_active}; border-color: var(--border); background: {theme_tab_hover_bg}; }}
              .card {{ background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: 1rem; }}
              .muted {{ color: var(--muted); }}
              .models-toolbar {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 0.8rem; margin-bottom: 0.8rem; }}
              .models-toolbar input {{ max-width: 360px; }}
              .toolbar-right {{ display: flex; flex-direction: column; align-items: flex-end; gap: 0.45rem; }}
              .toolbar-right .small-btn {{ border: 1px solid var(--border); border-radius: 8px; padding: 0.4rem 0.6rem; background: {theme_secondary_bg}; color: {theme_secondary_text}; cursor: pointer; font-size: 0.82rem; }}
              .toolbar-right .small-btn:hover {{ filter: brightness(1.07); }}
              .models-table-wrap {{ border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }}
              .models-table {{ width: 100%; border-collapse: collapse; }}
              .models-table thead th {{ background: {theme_panel2}; color: {theme_table_head}; padding: 0.6rem; font-size: 0.9rem; letter-spacing: 0.01em; }}
              .model-row {{ cursor: pointer; }}
              .model-row td {{ padding: 0.62rem 0.6rem; color: var(--text); }}
              .models-table tbody tr.model-row:nth-child(4n+1),
              .models-table tbody tr.model-row:nth-child(4n+2) {{ background: {theme_surface}; }}
              .models-table tbody tr.model-row:nth-child(4n+3),
              .models-table tbody tr.model-row:nth-child(4n+4) {{ background: {theme_panel}; }}
              .models-table tbody tr.model-row:hover {{ outline: 1px solid {theme_badge}; outline-offset: -1px; }}
              .hidden-row {{ display: none; }}
              table {{ width: 100%; border-collapse: collapse; }}
              th, td {{ border-bottom: 1px solid var(--border); padding: 0.45rem 0.35rem; text-align: left; }}
              th {{ color: {theme_table_head}; }}
              td {{ color: var(--text); }}
              a, button {{ color: inherit; }}
              .btn-link {{ display: inline-block; margin-bottom: 0.8rem; color: {theme_link}; text-decoration: none; }}
              #contextMenu {{ position: fixed; display: none; z-index: 40; min-width: 170px; background: {theme_context_bg}; border: 1px solid var(--border); border-radius: 10px; box-shadow: 0 16px 30px rgba(0,0,0,{'0.18' if is_spot else '0.45'}); }}
              #contextMenu button {{ width: 100%; border: none; background: transparent; color: var(--text); text-align: left; padding: 0.65rem 0.75rem; cursor: pointer; }}
              #contextMenu button:hover {{ background: {theme_context_hover}; }}
              .modal-backdrop {{ position: fixed; inset: 0; background: rgba(0,0,0,0.55); display: none; align-items: center; justify-content: center; z-index: 30; }}
              .modal {{ width: min(560px, 92vw); background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); border: 1px solid var(--border); border-radius: 14px; padding: 1rem; }}
              .form-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }}
              label {{ display: block; color: var(--muted); font-size: 0.92rem; }}
              input, select {{ width: 100%; margin-top: 0.35rem; background: {theme_surface}; color: var(--text); border: 1px solid var(--border); border-radius: 10px; padding: 0.58rem 0.65rem; }}
              .row-actions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.9rem; }}
              .row-actions button {{ border: none; border-radius: 10px; padding: 0.62rem 0.65rem; cursor: pointer; font-weight: 600; }}
              .primary {{ background: var(--accent); color: #ffffff; }}
              .secondary {{ background: {theme_secondary_bg}; color: {theme_secondary_text}; border: 1px solid var(--border); }}
              body.mobile-ui .container {{ padding: 1rem 0.75rem; }}
              body.mobile-ui .topbar-inner {{ padding: 0.7rem 0.75rem; gap: 0.5rem; flex-wrap: wrap; }}
              body.mobile-ui .brand {{ margin-right: 0; }}
              body.mobile-ui .mobile-menu-toggle {{ display: inline-flex; align-items: center; justify-content: center; margin-left: auto; }}
              body.mobile-ui .nav-links {{ display: none; width: 100%; flex-direction: column; align-items: stretch; gap: 0.4rem; }}
              body.mobile-ui .nav-links.open {{ display: flex; }}
              body.mobile-ui .tab-link {{ flex: 1 1 100%; text-align: left; }}
              body.mobile-ui .models-toolbar {{ flex-direction: column; align-items: stretch; }}
              body.mobile-ui .toolbar-right {{ align-items: stretch; }}
              body.mobile-ui .models-table-wrap {{ overflow-x: auto; }}
              body.mobile-ui .form-grid {{ grid-template-columns: 1fr; }}
              body.mobile-ui .row-actions {{ grid-template-columns: 1fr; }}
              body.mobile-ui #contextMenu {{ display: none !important; }}
            </style>
            <nav class="topbar">
              <div class="topbar-inner">
                <a href="{home_href}" class="brand">{brand_label}</a>
                <button type="button" id="mobileNavToggle" class="mobile-menu-toggle" aria-label="Toggle menu" aria-controls="primaryNavMenu" aria-expanded="false">&#9776;</button>
                <div id="primaryNavMenu" class="nav-links">
                  <a href="{home_href}" class="tab-link">Model</a>
                  <a href="{manage_href}" class="tab-link active">Manage Models</a>
                  <a href="{run_models_href}" class="tab-link">Run Models</a>
                  <a href="{bots_href}" class="tab-link">Bots</a>
                  <a href="{mode_switch_href}" class="tab-link">{mode_switch_label}</a>
                </div>
              </div>
            </nav>
            <div class="container">
              <h1>{heading_label}</h1>
              <p class="muted">Click a model to edit preset settings (ticker, candle length, rows, buy/sell thresholds, stop-loss strategy, include in Run All). Right-click a model for rename/delete.</p>
              {message_html}
              {error_html}
              <div class="card">
                <h2>Saved Models</h2>
                <div class="models-toolbar">
                  <label for="modelSearch" class="muted">Search models by name</label>
                  <div class="toolbar-right">
                    <button type="button" class="small-btn" onclick="openAllSettings()">Edit all model presets</button>
                    <button type="button" class="small-btn" onclick="deleteAllModels()">Delete all models</button>
                    <input type="text" id="modelSearch" placeholder="Type model name (e.g. ab)" />
                  </div>
                </div>
                {f'''
                <div class="models-table-wrap">
                  <table class="models-table">
                    <thead>
                      <tr>
                        <th>Model Name</th>
                        <th>Ticker</th>
                        <th>Interval</th>
                        <th>Rows</th>
                        <th>Buy &gt;</th>
                        <th>Sell &lt;</th>
                        <th>Stop Loss</th>
                        <th>Fixed Stop</th>
                        <th>Take Profit</th>
                        <th>Max Hold</th>
                        <th>Run All</th>
                      </tr>
                    </thead>
                    <tbody id="modelsTableBody">
                      {model_rows}
                    </tbody>
                  </table>
                </div>
                <p id="modelSearchEmpty" class="muted" style="display:none; margin-top:0.8rem;">No models match that search.</p>
                ''' if model_rows else "<p class='muted'>No saved models yet. Train and save one from the main page.</p>"}
              </div>
            </div>

            <div id="contextMenu">
              <button type="button" onclick="openRename()">Rename model</button>
              <button type="button" onclick="deleteModel()">Delete model</button>
            </div>

            <div id="settingsModal" class="modal-backdrop">
              <div class="modal">
                <h3 id="modalTitle">Model Settings</h3>
                <form method="post">
                  <input type="hidden" name="action" value="save_config" />
                  <input type="hidden" name="model_name" id="cfgModelName" />
                  <div class="form-grid">
                    <label>Ticker<input type="text" name="ticker" id="cfgTicker" required /></label>
                    <label>Candle Length
                      <select name="interval" id="cfgInterval">
                        <option value="1d">Daily</option>
                        <option value="1h">1 hour</option>
                        <option value="15m">15 min</option>
                        <option value="5m">5 min</option>
                      </select>
                    </label>
                    <label>Rows<input type="number" min="50" name="rows" id="cfgRows" required /></label>
                    <label>BUY if P(Up) &gt;
                      <input type="number" min="0" max="1" step="0.01" name="buy_threshold" id="cfgBuyThreshold" placeholder="0.60" />
                    </label>
                    <label>SELL if P(Up) &lt;
                      <input type="number" min="0" max="1" step="0.01" name="sell_threshold" id="cfgSellThreshold" placeholder="0.40" />
                    </label>
                    <label>Stop Loss Strategy
                      <select name="stop_loss_strategy" id="cfgStopLossStrategy">
                        <option value="none">None</option>
                        <option value="atr">Volatility Buffer (ATR-Based)</option>
                        <option value="model_invalidation">Model Invalidation (MAE-Linked)</option>
                        <option value="time_decay">Time-Decay (Temporal Exit)</option>
                        <option value="fixed_percentage">Fixed Percentage</option>
                        <option value="trailing_stop">Trailing Stop Loss</option>
                      </select>
                    </label>
                    <label id="cfgFixedStopWrap">Fixed Stop Loss %
                      <input type="number" min="0.01" step="0.1" name="fixed_stop_pct" id="cfgFixedStopPct" placeholder="2.0" />
                    </label>
                    <label>Take Profit %
                      <input type="number" min="0.01" step="0.01" name="take_profit_pct" id="cfgTakeProfitPct" placeholder="1.5" />
                    </label>
                    <label>Max Hold Bars
                      <input type="number" min="1" step="1" name="max_hold_bars" id="cfgMaxHoldBars" placeholder="10" />
                    </label>
                    <label>Include in Run All
                      <select name="include_in_run_all" id="cfgInclude">
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                      </select>
                    </label>
                  </div>
                  <div class="row-actions">
                    <button class="primary" type="submit">Save Settings</button>
                    <button class="secondary" type="button" onclick="closeModals()">Cancel</button>
                  </div>
                  <div class="row-actions">
                    <button class="secondary" type="button" onclick="openRenameFromSettings()">Rename Model</button>
                    <button class="secondary" type="button" onclick="deleteFromSettings()">Delete Model</button>
                  </div>
                </form>
              </div>
            </div>

            <div id="renameModal" class="modal-backdrop">
              <div class="modal">
                <h3>Rename Model</h3>
                <form method="post">
                  <input type="hidden" name="action" value="rename_model" />
                  <input type="hidden" name="model_name" id="renameModelName" />
                  <label>New Name<input type="text" name="new_name" id="renameInput" required /></label>
                  <div class="row-actions">
                    <button class="primary" type="submit">Rename</button>
                    <button class="secondary" type="button" onclick="closeModals()">Cancel</button>
                  </div>
                </form>
              </div>
            </div>

            <div id="allSettingsModal" class="modal-backdrop">
              <div class="modal">
                <h3>Global Preset Settings (All Models)</h3>
                <form method="post">
                  <input type="hidden" name="action" value="save_all_configs" />
                  <p class="muted">Leave fields blank to keep existing values unchanged.</p>
                  <div class="form-grid">
                    <label>Ticker<input type="text" name="ticker" id="allTicker" /></label>
                    <label>Candle Length
                      <select name="interval" id="allInterval">
                        <option value="">No change</option>
                        <option value="1d">Daily</option>
                        <option value="1h">1 hour</option>
                        <option value="15m">15 min</option>
                        <option value="5m">5 min</option>
                      </select>
                    </label>
                    <label>Rows<input type="number" min="50" name="rows" id="allRows" /></label>
                    <label>BUY if P(Up) &gt;
                      <input type="number" min="0" max="1" step="0.01" name="buy_threshold" id="allBuyThreshold" placeholder="0.60" />
                    </label>
                    <label>SELL if P(Up) &lt;
                      <input type="number" min="0" max="1" step="0.01" name="sell_threshold" id="allSellThreshold" placeholder="0.40" />
                    </label>
                    <label>Stop Loss Strategy
                      <select name="stop_loss_strategy" id="allStopLossStrategy">
                        <option value="">No change</option>
                        <option value="none">None</option>
                        <option value="atr">Volatility Buffer (ATR-Based)</option>
                        <option value="model_invalidation">Model Invalidation (MAE-Linked)</option>
                        <option value="time_decay">Time-Decay (Temporal Exit)</option>
                        <option value="fixed_percentage">Fixed Percentage</option>
                        <option value="trailing_stop">Trailing Stop Loss</option>
                      </select>
                    </label>
                    <label id="allFixedStopWrap">Fixed Stop Loss %
                      <input type="number" min="0.01" step="0.1" name="fixed_stop_pct" id="allFixedStopPct" placeholder="2.0" />
                    </label>
                    <label>Take Profit %
                      <input type="number" min="0.01" step="0.01" name="take_profit_pct" id="allTakeProfitPct" placeholder="No change" />
                    </label>
                    <label>Max Hold Bars
                      <input type="number" min="1" step="1" name="max_hold_bars" id="allMaxHoldBars" placeholder="No change" />
                    </label>
                    <label>Include in Run All
                      <select name="include_in_run_all" id="allInclude">
                        <option value="">No change</option>
                        <option value="1">Yes</option>
                        <option value="0">No</option>
                      </select>
                    </label>
                  </div>
                  <div class="row-actions">
                    <button class="primary" type="submit">Save For All Models</button>
                    <button class="secondary" type="button" onclick="closeModals()">Cancel</button>
                  </div>
                </form>
              </div>
            </div>

            <form id="deleteForm" method="post" style="display:none;">
              <input type="hidden" name="action" value="delete_model" />
              <input type="hidden" name="model_name" id="deleteModelName" />
            </form>

            <form id="deleteAllForm" method="post" style="display:none;">
              <input type="hidden" name="action" value="delete_all_models" />
            </form>

            <script>
              const settingsModal = document.getElementById("settingsModal");
              const renameModal = document.getElementById("renameModal");
              const allSettingsModal = document.getElementById("allSettingsModal");
              const menu = document.getElementById("contextMenu");
              const mobileNavToggle = document.getElementById("mobileNavToggle");
              const primaryNavMenu = document.getElementById("primaryNavMenu");
              let menuModelName = "";

              if (mobileNavToggle && primaryNavMenu) {{
                mobileNavToggle.addEventListener("click", () => {{
                  const isOpen = primaryNavMenu.classList.toggle("open");
                  mobileNavToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
                }});
              }}

              function closeModals() {{
                settingsModal.style.display = "none";
                renameModal.style.display = "none";
                allSettingsModal.style.display = "none";
                menu.style.display = "none";
              }}

              function openModel(card) {{
                const model = card.dataset.model;
                document.getElementById("modalTitle").textContent = `Preset Settings • ${{model}}`;
                document.getElementById("cfgModelName").value = model;
                document.getElementById("cfgTicker").value = card.dataset.ticker || "AAPL";
                document.getElementById("cfgInterval").value = card.dataset.interval || "1d";
                document.getElementById("cfgRows").value = card.dataset.rows || "250";
                document.getElementById("cfgBuyThreshold").value = card.dataset.buy || "0.60";
                document.getElementById("cfgSellThreshold").value = card.dataset.sell || "0.40";
                document.getElementById("cfgStopLossStrategy").value = card.dataset.stopLoss || "none";
                document.getElementById("cfgFixedStopPct").value = card.dataset.fixedStop || "2.0";
                document.getElementById("cfgTakeProfitPct").value = card.dataset.takeProfit || "";
                document.getElementById("cfgMaxHoldBars").value = card.dataset.maxHold || "";
                document.getElementById("cfgInclude").value = card.dataset.include || "1";
                toggleCfgFixedStop();
                settingsModal.style.display = "flex";
              }}

              function toggleCfgFixedStop() {{
                const strategy = document.getElementById("cfgStopLossStrategy").value;
                const wrap = document.getElementById("cfgFixedStopWrap");
                const fixedStopInput = document.getElementById("cfgFixedStopPct");
                const isFixed = strategy === "fixed_percentage" || strategy === "trailing_stop";
                wrap.style.display = isFixed ? "block" : "none";
                if (fixedStopInput) {{
                  fixedStopInput.disabled = !isFixed;
                }}
              }}

              function openRename() {{
                if (!menuModelName) return;
                document.getElementById("renameModelName").value = menuModelName;
                document.getElementById("renameInput").value = menuModelName;
                renameModal.style.display = "flex";
                menu.style.display = "none";
              }}

              function openRenameFromSettings() {{
                const activeModel = document.getElementById("cfgModelName").value || "";
                if (!activeModel) return;
                menuModelName = activeModel;
                openRename();
              }}

              function openAllSettings() {{
                document.getElementById("allTicker").value = "";
                document.getElementById("allInterval").value = "";
                document.getElementById("allRows").value = "";
                document.getElementById("allBuyThreshold").value = "";
                document.getElementById("allSellThreshold").value = "";
                document.getElementById("allStopLossStrategy").value = "";
                document.getElementById("allFixedStopPct").value = "";
                document.getElementById("allTakeProfitPct").value = "";
                document.getElementById("allMaxHoldBars").value = "";
                document.getElementById("allInclude").value = "";
                toggleAllFixedStop();
                allSettingsModal.style.display = "flex";
              }}

              function toggleAllFixedStop() {{
                const strategy = document.getElementById("allStopLossStrategy").value;
                const wrap = document.getElementById("allFixedStopWrap");
                const fixedStopInput = document.getElementById("allFixedStopPct");
                const isFixed = strategy === "fixed_percentage" || strategy === "trailing_stop";
                wrap.style.display = isFixed ? "block" : "none";
                if (fixedStopInput) {{
                  fixedStopInput.disabled = !isFixed;
                }}
              }}

              function deleteModel() {{
                if (!menuModelName) return;
                if (confirm(`Delete model "${{menuModelName}}"?`)) {{
                  document.getElementById("deleteModelName").value = menuModelName;
                  document.getElementById("deleteForm").submit();
                }}
              }}

              function deleteFromSettings() {{
                const activeModel = document.getElementById("cfgModelName").value || "";
                if (!activeModel) return;
                menuModelName = activeModel;
                deleteModel();
              }}

              function deleteAllModels() {{
                if (confirm("Delete ALL saved models in this mode? This cannot be undone.")) {{
                  document.getElementById("deleteAllForm").submit();
                }}
              }}

              const modelRows = Array.from(document.querySelectorAll(".model-row"));
              const modelSearch = document.getElementById("modelSearch");
              const modelSearchEmpty = document.getElementById("modelSearchEmpty");

              function applyModelFilter() {{
                if (!modelSearch) return;
                const query = modelSearch.value.trim().toLowerCase();
                let visibleCount = 0;
                modelRows.forEach((row) => {{
                  const name = (row.dataset.model || "").toLowerCase();
                  const show = !query || name.includes(query);
                  row.classList.toggle("hidden-row", !show);
                  if (show) visibleCount += 1;
                }});
                if (modelSearchEmpty) {{
                  modelSearchEmpty.style.display = visibleCount === 0 ? "block" : "none";
                }}
              }}

              modelRows.forEach((card) => {{
                card.addEventListener("click", () => openModel(card));
                card.addEventListener("contextmenu", (evt) => {{
                  evt.preventDefault();
                  menuModelName = card.dataset.model;
                  menu.style.left = `${{evt.clientX}}px`;
                  menu.style.top = `${{evt.clientY}}px`;
                  menu.style.display = "block";
                }});
              }});
              if (modelSearch) {{
                modelSearch.addEventListener("input", applyModelFilter);
              }}
              applyModelFilter();
              document.getElementById("cfgStopLossStrategy").addEventListener("change", toggleCfgFixedStop);
              document.getElementById("allStopLossStrategy").addEventListener("change", toggleAllFixedStop);
              toggleCfgFixedStop();
              toggleAllFixedStop();

              window.addEventListener("click", (evt) => {{
                if (evt.target === settingsModal || evt.target === renameModal || evt.target === allSettingsModal) {{
                  closeModals();
                  return;
                }}
                if (!menu.contains(evt.target)) {{
                  menu.style.display = "none";
                }}
              }});
            </script>
          </body>
        </html>
        """

    @app.route("/run-models", methods=["GET", "POST"])
    @app.route("/spot/run-models", methods=["GET", "POST"])
    def run_models_page() -> str:
        is_spot = request.path.startswith("/spot")
        is_mobile_ui = is_mobile_request(request)
        mode_key = SPOT_MODE if is_spot else OPTIONS_MODE
        home_href = "/spot" if is_spot else "/"
        manage_href = "/spot/manage-models" if is_spot else "/manage-models"
        run_models_href = "/spot/run-models" if is_spot else "/run-models"
        bots_href = "/spot/bots" if is_spot else "/bots"
        mode_switch_href = "/" if is_spot else "/spot"
        mode_switch_label = "Switch to Options Mode" if is_spot else "Switch to Spot Mode"
        brand_label = "Quant Trader • Spot Mode" if is_spot else "Quant Trader • Options Mode"
        present_rule_text = (
            "Decision rule: BUY if P(Up) exceeds BUY threshold, SELL to exit if P(Up) is below SELL threshold, else HOLD."
            if is_spot
            else "Decision rule: BUY if P(Up) exceeds BUY threshold, SELL(short) if P(Up) is below SELL threshold, else HOLD."
        )
        theme_bg = "#100d07" if is_spot else "#090c12"
        theme_text = "#efe0be" if is_spot else "#c6ccd7"
        theme_panel = "#19130b" if is_spot else "#121722"
        theme_panel2 = "#221b10" if is_spot else "#1a202c"
        theme_border = "#4a3a20" if is_spot else "#3a4455"
        theme_muted = "#b79f66" if is_spot else "#98a2b3"
        theme_accent = "#d4af37" if is_spot else "#c0c0c0"
        theme_topbar_bg = "rgba(16, 13, 7, 0.94)" if is_spot else "rgba(9, 12, 18, 0.94)"
        theme_brand = "#f2dd9f" if is_spot else "#d8dde6"
        theme_tab = "#c8ac60" if is_spot else "#aab3c2"
        theme_tab_active = "#f3e2b5" if is_spot else "#e5e7eb"
        theme_tab_hover_bg = "#2a210f" if is_spot else "#1a212f"
        theme_surface = "#130f08" if is_spot else "#0f141e"
        theme_secondary_bg = "#312511" if is_spot else "#2a3342"
        theme_secondary_text = "#f3e2b6" if is_spot else "#e5e7eb"

        message_html = ""
        error_html = ""
        present_html = ""
        run_all_html = ""
        provider_notices: list[str] = []
        present_ticker = request.form.get("present_ticker", "AAPL").upper().strip()
        present_interval = request.form.get("present_interval", "1d")
        present_rows = request.form.get("present_rows", "250")
        present_buy_raw = request.form.get("present_buy_threshold", "").strip()
        present_sell_raw = request.form.get("present_sell_threshold", "").strip()
        present_model = request.form.get("present_model", "__new__")
        present_stop_loss_strategy_raw = request.form.get("present_stop_loss_strategy", StopLossStrategy.NONE.value).strip()
        present_fixed_stop_pct_raw = request.form.get("present_fixed_stop_pct", "2.0").strip()
        present_take_profit_pct_raw = request.form.get("present_take_profit_pct", "").strip()
        present_max_hold_bars_raw = request.form.get("present_max_hold_bars", "").strip()
        prediction_horizon_raw = request.form.get("prediction_horizon", "5").strip()
        split_style = request.form.get("split_style", "shuffled")
        evaluation_split_raw = request.form.get("evaluation_split", "").strip()
        feature_set = normalize_feature_set(request.form.get("feature_set", "feature2"))
        dqn_episodes_raw = request.form.get("dqn_episodes", "120").strip()
        data_provider = request.form.get("data_provider", "yfinance").strip().lower()
        if data_provider not in ("yfinance", "twelvedata", "massive"):
            data_provider = "yfinance"
        twelve_api_key = os.getenv("TWELVE_DATA_API_KEY", "e90093c59e7a436d9436e34b56a6e6a5").strip()
        massive_api_key = os.getenv("MASSIVE_API_KEY", "5_9g13XovqZWWKsHiPp9B8L9LaReqjbn").strip()
        webhook_url = get_app_setting(mode_key, "discord_webhook_url", "")
        saved_models = list_saved_models(mode_key)
        run_all_rows = ""

        if request.method == "POST":
            mode = request.form.get("mode", "").strip()
            try:
                if mode == "present":
                    present_row_count = int(present_rows)
                    if present_row_count < 50:
                        raise ValueError("Rows must be at least 50.")
                    present_buy_threshold, present_sell_threshold = parse_thresholds(present_buy_raw, present_sell_raw)
                    present_stop_loss_strategy = parse_stop_loss_strategy(present_stop_loss_strategy_raw)
                    present_fixed_stop_pct = 2.0
                    if present_stop_loss_strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
                        present_fixed_stop_pct = validate_fixed_stop_pct(float(present_fixed_stop_pct_raw or "2.0"))
                    present_take_profit_pct = 0.0 if not present_take_profit_pct_raw else validate_take_profit_pct(float(present_take_profit_pct_raw))
                    present_max_hold_bars = 0 if not present_max_hold_bars_raw else validate_max_hold_bars(int(present_max_hold_bars_raw))
                    dataset, provider_notice = fetch_market_rows(
                        ticker=present_ticker,
                        interval=present_interval,
                        row_count=present_row_count,
                        provider=data_provider,
                        twelve_api_key=twelve_api_key,
                        massive_api_key=massive_api_key,
                        prediction_horizon=int(prediction_horizon_raw or "5"),
                    )
                    if provider_notice:
                        provider_notices.append(provider_notice)
                    latest_row = dataset[-1]
                    present_dqn_episodes = int(dqn_episodes_raw or "120")
                    if present_model == "__new__":
                        bundle = train_strategy_models(dataset, split_style=split_style, feature_set=feature_set, dqn_episodes=present_dqn_episodes)
                    else:
                        bundle = load_model_bundle(mode_key, present_model)
                    prediction = predict_signal(
                        bundle,
                        latest_row,
                        buy_threshold=present_buy_threshold,
                        sell_threshold=present_sell_threshold,
                        long_only=is_spot,
                    )
                    present_stop_price = stop_loss_price(
                        strategy=present_stop_loss_strategy,
                        action=str(prediction["action"]),
                        reference_price=float(latest_row.get("close", 0.0)),
                        expected_return=float(prediction["expected_return"]),
                        fixed_pct=present_fixed_stop_pct,
                        atr_fraction=float(latest_row.get("atr_frac", 0.0)),
                        model_mae=MODEL_MAE_DEFAULT,
                    )
                    present_html = (
                        "<section class='results'><article class='card'>"
                        f"<h2>Present Mode • {present_ticker} ({present_interval})</h2>"
                        f"<p class='muted'>Model: {'Freshly trained on current dataset' if present_model == '__new__' else present_model}</p>"
                        f"<p class='muted'>{present_rule_text} (BUY&gt;{present_buy_threshold:.2f}, SELL&lt;{present_sell_threshold:.2f}).</p>"
                        f"<p><span class='muted'>Expected Return (next candle)</span> <strong>{prediction['expected_return']:+.4%}</strong></p>"
                        f"<p><span class='muted'>P(Up)</span> <strong>{prediction['p_up']:.2%}</strong></p>"
                        f"<p><span class='muted'>Stop Loss Strategy</span> <strong>{present_stop_loss_strategy.value}</strong></p>"
                        f"<p><span class='muted'>Calculated Stop Price</span> <strong>{f'{present_stop_price:.4f}' if present_stop_price is not None else 'n/a'}</strong></p>"
                        f"<p><span class='muted'>Action</span> <strong>{prediction['action']}</strong></p>"
                        "</article></section>"
                    )
                elif mode == "present_all":
                    model_configs = load_model_configs(mode_key)
                    model_configs = {name: get_model_config(name, model_configs) for name in saved_models}
                    model_configs["__ui_data_provider__"] = data_provider
                    model_configs["__ui_twelve_api_key__"] = twelve_api_key
                    model_configs["__ui_massive_api_key__"] = massive_api_key
                    run_all_rows = build_run_all_rows(saved_models, model_configs, mode=mode_key, long_only=is_spot)
                    run_all_html = "<p class='muted'>Latest outputs for all models currently included in Run All.</p>"
                elif mode == "run_all_monitor":
                    monitor_action = request.form.get("monitor_action", "").strip()
                    webhook_input = request.form.get("discord_webhook_url", "").strip()
                    if webhook_input:
                        webhook_url = webhook_input
                        set_app_setting(mode_key, "discord_webhook_url", webhook_url)
                    if monitor_action == "start":
                        if not webhook_url:
                            raise ValueError("Please enter and save a Discord webhook URL before starting continuous mode.")
                        RUN_ALL_MONITOR.start(
                            mode=mode_key,
                            long_only=is_spot,
                            data_provider=data_provider,
                            twelve_api_key=twelve_api_key,
                            massive_api_key=massive_api_key,
                            webhook_url=webhook_url,
                        )
                        message_html = "<p style='color:#7bd88f;'><strong>Continuous Run All mode started.</strong></p>"
                    elif monitor_action == "stop":
                        RUN_ALL_MONITOR.stop(mode_key, webhook_url)
                        message_html = "<p style='color:#7bd88f;'><strong>Continuous Run All mode stopped.</strong></p>"
                    elif monitor_action == "save_webhook":
                        message_html = "<p style='color:#7bd88f;'><strong>Webhook URL saved.</strong></p>"
            except Exception as exc:
                error_html = f"<p style='color:red;'><strong>Error:</strong> {escape(str(exc))}</p>"

        monitor_state = RUN_ALL_MONITOR.status(mode_key)
        worker_alive = bool(monitor_state.get("worker_alive"))
        worker_state = escape(str(monitor_state.get("worker_state", "stopped")))
        monitor_running = bool(monitor_state.get("running"))
        monitor_last_error = escape(str(monitor_state.get("last_error", "")))
        monitor_next_tick = escape(format_display_time(monitor_state.get("next_tick_at", "")))
        monitor_rows = monitor_state.get("last_rows", [])
        monitor_rows_html = build_run_all_rows_from_results(monitor_rows) if isinstance(monitor_rows, list) else ""
        error_line = f"<br><span style='color:#ff7b7b;'><strong>Last error:</strong> {monitor_last_error}</span>" if monitor_last_error else ""
        monitor_status_html = (
            f"<p class='muted'><strong>Bot:</strong> {'Online' if worker_alive else 'Offline'} | "
            f"<strong>Lifecycle:</strong> {'Crashed' if worker_state == 'crashed' else ('Running' if monitor_running else 'Stopped')} | "
            f"<strong>Next aligned update:</strong> {monitor_next_tick or 'n/a'}</p>{error_line}"
        )
        provider_notice_html = ""
        if provider_notices:
            provider_notice_html = "<div class='card'><h3>Data Provider Notices</h3><ul>" + "".join(
                f"<li>{escape(note)}</li>" for note in dict.fromkeys(provider_notices)
            ) + "</ul></div>"

        return f"""
        <html><head><title>Run Models</title><meta name="viewport" content="width=device-width, initial-scale=1" /></head><body class="{'mobile-ui' if is_mobile_ui else 'desktop-ui'}">
        <style>
        :root {{--bg:{theme_bg};--panel:{theme_panel};--panel2:{theme_panel2};--border:{theme_border};--text:{theme_text};--muted:{theme_muted};--accent:{theme_accent};}}
        *{{box-sizing:border-box;}} body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;}}
        .topbar{{position:sticky;top:0;z-index:50;background:{theme_topbar_bg};border-bottom:1px solid var(--border);}} .topbar-inner{{max-width:1100px;margin:0 auto;padding:0.9rem 2rem;display:flex;align-items:center;gap:1rem;}}
        .brand{{font-weight:700;color:{theme_brand};text-decoration:none;margin-right:auto;}} .mobile-menu-toggle{{display:none;border:1px solid var(--border);background:transparent;color:var(--text);border-radius:8px;padding:.35rem .55rem;font-size:1rem;line-height:1;cursor:pointer;}} .mobile-menu-toggle:hover{{background:{theme_tab_hover_bg};}} .nav-links{{display:flex;align-items:center;gap:1rem;}} .tab-link{{color:{theme_tab};text-decoration:none;padding:.4rem .65rem;border-radius:8px;border:1px solid transparent;}}
        .tab-link:hover,.tab-link.active{{color:{theme_tab_active};border-color:var(--border);background:{theme_tab_hover_bg};}} .container{{max-width:1100px;margin:0 auto;padding:2rem;}}
        .card{{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);border:1px solid var(--border);border-radius:14px;padding:1rem 1.1rem;margin-bottom:1rem;}}
        .form-grid{{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:.8rem;}} label{{display:block;color:var(--muted);font-size:.92rem;}}
        input,select{{width:100%;margin-top:.35rem;background:{theme_surface};color:var(--text);border:1px solid var(--border);border-radius:10px;padding:.58rem .65rem;}}
        ul,li{{color:var(--text);}}
        .muted{{color:var(--muted);}}
        table{{width:100%;border-collapse:collapse;color:var(--text);}} th,td{{padding:.5rem;border-bottom:1px solid var(--border);text-align:left;color:var(--text);}}
        .run-all-group-row td{{background:rgba(255,255,255,0.02);color:var(--text);}}
        .run-all-group-divider td{{padding:0;border-bottom:none;height:.35rem;}}
        details,summary{{color:var(--text);}}
        button{{border:none;border-radius:10px;padding:.62rem .8rem;cursor:pointer;background:{theme_accent};color:#111;font-weight:700;}} .secondary{{background:{theme_secondary_bg};color:{theme_secondary_text};border:1px solid var(--border);}}
        body.mobile-ui .topbar-inner{{padding:.7rem .75rem;gap:.5rem;flex-wrap:wrap;}}
        body.mobile-ui .brand{{margin-right:0;}}
        body.mobile-ui .mobile-menu-toggle{{display:inline-flex;align-items:center;justify-content:center;margin-left:auto;}}
        body.mobile-ui .nav-links{{display:none;width:100%;flex-direction:column;align-items:stretch;gap:.4rem;}}
        body.mobile-ui .nav-links.open{{display:flex;}}
        body.mobile-ui .tab-link{{flex:1 1 100%;text-align:left;}}
        body.mobile-ui .container{{padding:1rem .75rem;}}
        body.mobile-ui .form-grid{{grid-template-columns:1fr;}}
        body.mobile-ui table{{display:block;overflow-x:auto;white-space:nowrap;}}
        </style>
        <nav class="topbar"><div class="topbar-inner">
            <a href="{home_href}" class="brand">{brand_label}</a>
            <button type="button" id="mobileNavToggle" class="mobile-menu-toggle" aria-label="Toggle menu" aria-controls="primaryNavMenu" aria-expanded="false">&#9776;</button>
            <div id="primaryNavMenu" class="nav-links">
              <a href="{home_href}" class="tab-link">Model</a>
              <a href="{manage_href}" class="tab-link">Manage Models</a>
              <a href="{run_models_href}" class="tab-link active">Run Models</a>
              <a href="{bots_href}" class="tab-link">Bots</a>
              <a href="{mode_switch_href}" class="tab-link">{mode_switch_label}</a>
            </div>
        </div></nav>
        <div class="container">
          <h1>Run Models</h1>
          <form method="post" class="card">
            <input type="hidden" name="mode" value="present" />
            <input type="hidden" name="data_provider" value="{data_provider}" />
            <h2>Run Model</h2>
            <div class="form-grid">
              <label>Ticker<input type="text" name="present_ticker" value="{present_ticker}" required /></label>
              <label>Candle Length<select name="present_interval"><option value="1d" {"selected" if present_interval == "1d" else ""}>Daily</option><option value="1h" {"selected" if present_interval == "1h" else ""}>1 hour</option><option value="15m" {"selected" if present_interval == "15m" else ""}>15 min</option><option value="5m" {"selected" if present_interval == "5m" else ""}>5 min</option></select></label>
              <label>Rows<input type="number" min="50" name="present_rows" value="{present_rows}" required /></label>
              <label>Model<select name="present_model"><option value="__new__">Train new model</option>{"".join(f'<option value="{name}" {"selected" if present_model == name else ""}>{name}</option>' for name in saved_models)}</select></label>
              <label>BUY threshold<input type="number" min="0" max="1" step="0.01" name="present_buy_threshold" value="{present_buy_raw}" placeholder="0.60" /></label>
              <label>SELL threshold<input type="number" min="0" max="1" step="0.01" name="present_sell_threshold" value="{present_sell_raw}" placeholder="0.40" /></label>
              <label>Stop Loss Strategy<select name="present_stop_loss_strategy"><option value="none" {"selected" if present_stop_loss_strategy_raw == "none" else ""}>None</option><option value="atr" {"selected" if present_stop_loss_strategy_raw == "atr" else ""}>ATR</option><option value="model_invalidation" {"selected" if present_stop_loss_strategy_raw == "model_invalidation" else ""}>Model Invalidation</option><option value="time_decay" {"selected" if present_stop_loss_strategy_raw == "time_decay" else ""}>Time Decay</option><option value="fixed_percentage" {"selected" if present_stop_loss_strategy_raw == "fixed_percentage" else ""}>Fixed Percentage</option><option value="trailing_stop" {"selected" if present_stop_loss_strategy_raw == "trailing_stop" else ""}>Trailing Stop</option></select></label>
              <label>Fixed Stop %<input type="number" min="0.01" step="any" name="present_fixed_stop_pct" value="{present_fixed_stop_pct_raw}" /></label>
              <label>Take Profit %<input type="number" min="0.01" step="0.01" name="present_take_profit_pct" value="{present_take_profit_pct_raw}" placeholder="1.5" /></label>
              <label>Max Hold Bars<input type="number" min="1" step="1" name="present_max_hold_bars" value="{present_max_hold_bars_raw}" placeholder="10" /></label>
              <label>&nbsp;<button type="submit">Run Model</button></label>
            </div>
          </form>
          <form method="post" class="card">
            <input type="hidden" name="mode" value="present_all" />
            <input type="hidden" name="data_provider" value="{data_provider}" />
            <h2>Run All Preset Models</h2>
            <button type="submit">Run All Presets</button>
            {run_all_html}
            <table id="runAllTable"><tr><th>Model</th><th>Ticker</th><th>Candle</th><th data-sort-type="number">Rows</th><th>BUY/SELL</th><th>Stop Strategy</th><th data-sort-type="percent">Expected Return</th><th data-sort-type="percent">P(Up)</th><th data-sort-type="number">Stop Price</th><th>Action</th></tr>
            {run_all_rows if run_all_rows else "<tr><td colspan='10'>Press 'Run All Presets' to generate outputs.</td></tr>"}</table>
          </form>
          <form method="post" class="card">
            <input type="hidden" name="mode" value="run_all_monitor" />
            <input type="hidden" name="data_provider" value="{data_provider}" />
            <h2>Continuous Run All + Discord</h2>
            {monitor_status_html}
            <label>Discord Webhook URL<input type="text" name="discord_webhook_url" value="{escape(webhook_url)}" /></label>
            <div style="margin-top:.8rem;display:flex;gap:.6rem;"><button type="submit" name="monitor_action" value="save_webhook" class="secondary">Save Webhook</button><button type="submit" name="monitor_action" value="start">Start</button><button type="submit" name="monitor_action" value="stop" class="secondary">Stop</button></div>
            <p class="muted" style="margin-top:.65rem;">Continuous mode runs a full Run All update every aligned 5 minutes (:00, :05, :10, ...). Discord posts bot up/down/crash lifecycle events and model action changes.</p>
            <table><tr><th>Model</th><th>Ticker</th><th>Candle</th><th>Rows</th><th>BUY/SELL</th><th>Stop Strategy</th><th>Expected Return</th><th>P(Up)</th><th>Stop Price</th><th>Action</th></tr>
            {monitor_rows_html if monitor_rows_html else "<tr><td colspan='10'>No continuous updates yet. Start continuous mode to begin.</td></tr>"}</table>
          </form>
          {message_html}{error_html}{present_html}{provider_notice_html}
        </div>
        <script>
          const runAllTable = document.getElementById("runAllTable");
          const mobileNavToggle = document.getElementById("mobileNavToggle");
          const primaryNavMenu = document.getElementById("primaryNavMenu");
          const sortDirections = {{}};

          if (mobileNavToggle && primaryNavMenu) {{
            mobileNavToggle.addEventListener("click", () => {{
              const isOpen = primaryNavMenu.classList.toggle("open");
              mobileNavToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
            }});
          }}

          function parseSortValue(rawValue, sortType) {{
            const textValue = (rawValue || "").trim();
            if (sortType === "number") {{
              const parsed = Number.parseFloat(textValue.replace(/,/g, ""));
              return Number.isFinite(parsed) ? parsed : null;
            }}
            if (sortType === "percent") {{
              const parsed = Number.parseFloat(textValue.replace("%", "").replace(/,/g, ""));
              return Number.isFinite(parsed) ? parsed : null;
            }}
            return textValue.toLowerCase();
          }}

          function sortRunAllTable(columnIndex, direction, sortType) {{
            if (!runAllTable) return;
            const allRows = Array.from(runAllTable.querySelectorAll("tr")).slice(1);
            if (!allRows.length) return;
            const sortableRows = allRows.filter((row) => !row.classList.contains("run-all-group-row") && !row.classList.contains("run-all-group-divider"));
            if (!sortableRows.length) return;

            sortableRows.sort((leftRow, rightRow) => {{
              const leftCell = leftRow.children[columnIndex];
              const rightCell = rightRow.children[columnIndex];
              const leftValue = parseSortValue(leftCell?.textContent || "", sortType);
              const rightValue = parseSortValue(rightCell?.textContent || "", sortType);

              if (sortType === "number" || sortType === "percent") {{
                if (leftValue === null && rightValue === null) return 0;
                if (leftValue === null) return 1;
                if (rightValue === null) return -1;
                return direction === "desc" ? rightValue - leftValue : leftValue - rightValue;
              }}

              if (leftValue === rightValue) return 0;
              if (direction === "desc") {{
                return String(rightValue).localeCompare(String(leftValue));
              }}
              return String(leftValue).localeCompare(String(rightValue));
            }});

            allRows.forEach((row) => row.remove());
            sortableRows.forEach((row) => runAllTable.appendChild(row));
          }}

          if (runAllTable) {{
            const headerCells = Array.from(runAllTable.querySelectorAll("th"));
            headerCells.forEach((headerCell, index) => {{
              headerCell.style.cursor = "pointer";
              headerCell.title = "Click to sort";
              headerCell.addEventListener("click", () => {{
                const currentDirection = sortDirections[index] || "asc";
                const nextDirection = currentDirection === "asc" ? "desc" : "asc";
                sortDirections[index] = nextDirection;
                sortRunAllTable(index, nextDirection, headerCell.dataset.sortType || "text");
              }});
            }});
          }}
        </script>
        </body></html>
        """

    @app.route("/", methods=["GET", "POST"])
    @app.route("/spot", methods=["GET", "POST"])
    def index() -> str:
        is_spot = request.path.startswith("/spot")
        is_mobile_ui = is_mobile_request(request)
        mode_key = SPOT_MODE if is_spot else OPTIONS_MODE
        allow_short = not is_spot
        home_href = "/spot" if is_spot else "/"
        manage_href = "/spot/manage-models" if is_spot else "/manage-models"
        run_models_href = "/spot/run-models" if is_spot else "/run-models"
        bots_href = "/spot/bots" if is_spot else "/bots"
        mode_switch_href = "/" if is_spot else "/spot"
        mode_switch_label = "Switch to Options Mode" if is_spot else "Switch to Spot Mode"
        brand_label = "Quant Trader • Spot Mode" if is_spot else "Quant Trader • Options Mode"
        trainer_heading = "Spot Model Trainer (Long-only)" if is_spot else "Options Model Trainer (Long/Short)"
        present_rule_text = (
            "Decision rule: BUY if P(Up) exceeds BUY threshold, SELL to exit if P(Up) is below SELL threshold, else HOLD."
            if is_spot
            else "Decision rule: BUY if P(Up) exceeds BUY threshold, SELL(short) if P(Up) is below SELL threshold, else HOLD."
        )
        strategy_mode_text = "Long-only PnL simulation" if is_spot else "Long/Short PnL simulation"
        theme_bg = "#100d07" if is_spot else "#090c12"
        theme_text = "#efe0be" if is_spot else "#c6ccd7"
        theme_panel = "#19130b" if is_spot else "#121722"
        theme_panel2 = "#221b10" if is_spot else "#1a202c"
        theme_border = "#4a3a20" if is_spot else "#3a4455"
        theme_muted = "#b79f66" if is_spot else "#98a2b3"
        theme_accent = "#d4af37" if is_spot else "#c0c0c0"
        theme_topbar_bg = "rgba(16, 13, 7, 0.94)" if is_spot else "rgba(9, 12, 18, 0.94)"
        theme_brand = "#f2dd9f" if is_spot else "#d8dde6"
        theme_tab = "#c8ac60" if is_spot else "#aab3c2"
        theme_tab_active = "#f3e2b5" if is_spot else "#e5e7eb"
        theme_tab_hover_bg = "#2a210f" if is_spot else "#1a212f"
        theme_surface = "#130f08" if is_spot else "#0f141e"
        theme_details_bg = "#1e170d" if is_spot else "#131a26"
        theme_overlay = "rgba(10, 8, 4, 0.82)" if is_spot else "rgba(6, 8, 14, 0.78)"
        theme_table_head = "#e0c380" if is_spot else "#b5bfce"
        theme_heading = "#f3e2b5" if is_spot else "#d6dde8"
        theme_glow_primary = "#5f4514" if is_spot else "#2d364a"
        theme_glow_secondary = "#3f2e0d" if is_spot else "#1f2636"
        theme_shadow_alpha = "0.34" if is_spot else "0.35"
        theme_secondary_bg = "#312511" if is_spot else "#2a3342"
        theme_secondary_text = "#f3e2b6" if is_spot else "#e5e7eb"
        theme_progress_start = "#b38a2a" if is_spot else "#8f99aa"
        theme_progress_end = "#f0cf73" if is_spot else "#d4d8de"
        result_html = ""
        error_html = ""
        ticker = request.form.get("ticker", "AAPL").upper().strip()
        interval = request.form.get("interval", "1d")
        rows = request.form.get("rows", "250")
        prediction_horizon_raw = request.form.get("prediction_horizon", "5").strip()
        split_style = request.form.get("split_style", "shuffled")
        evaluation_split_raw = request.form.get("evaluation_split", "").strip()
        feature_set = normalize_feature_set(request.form.get("feature_set", "feature2"))
        dqn_episodes_raw = request.form.get("dqn_episodes", "120").strip()
        buy_threshold_raw = request.form.get("buy_threshold", "").strip()
        sell_threshold_raw = request.form.get("sell_threshold", "").strip()
        model_name = request.form.get("model_name", "").strip()
        stop_loss_strategy_raw = request.form.get("stop_loss_strategy", StopLossStrategy.NONE.value).strip()
        fixed_stop_pct_raw = request.form.get("fixed_stop_pct", "2.0").strip()
        take_profit_pct_raw = request.form.get("take_profit_pct", "").strip()
        max_hold_bars_raw = request.form.get("max_hold_bars", "").strip()
        selected_model = request.form.get("selected_model", "__new__")
        use_manual_weights_raw = request.form.get("use_manual_weights", "no").strip().lower()
        manual_weights_json = request.form.get("manual_feature_weights", "").strip()
        monte_carlo_method = request.form.get("monte_carlo_method", "none").strip().lower()
        if monte_carlo_method not in {"none", "bootstrap", "shuffle", "block"}:
            monte_carlo_method = "none"
        monte_carlo_n_sim_raw = request.form.get("monte_carlo_n_sim", "500").strip()
        monte_carlo_block_size_raw = request.form.get("monte_carlo_block_size", "20").strip()
        monte_carlo_seed_raw = request.form.get("monte_carlo_seed", "").strip()
        present_ticker = request.form.get("present_ticker", ticker).upper().strip()
        present_interval = request.form.get("present_interval", interval)
        present_rows = request.form.get("present_rows", rows)
        present_buy_raw = request.form.get("present_buy_threshold", "").strip()
        present_sell_raw = request.form.get("present_sell_threshold", "").strip()
        present_model = request.form.get("present_model", selected_model)
        present_stop_loss_strategy_raw = request.form.get("present_stop_loss_strategy", stop_loss_strategy_raw).strip()
        present_fixed_stop_pct_raw = request.form.get("present_fixed_stop_pct", fixed_stop_pct_raw).strip()
        present_take_profit_pct_raw = request.form.get("present_take_profit_pct", take_profit_pct_raw).strip()
        present_max_hold_bars_raw = request.form.get("present_max_hold_bars", max_hold_bars_raw).strip()
        mode = request.form.get("mode", "train")
        train_action = request.form.get("train_action", "train")
        evaluate_like_actions = {"evaluate", "evaluate_historical", "evaluate_update"}
        evaluate_historical_only = train_action == "evaluate_historical"
        if evaluate_historical_only:
            split_style = "chronological"
        data_provider = request.form.get("data_provider", "yfinance").strip().lower()
        if data_provider not in ("yfinance", "twelvedata", "massive"):
            data_provider = "yfinance"
        twelve_api_key = os.getenv("TWELVE_DATA_API_KEY", "e90093c59e7a436d9436e34b56a6e6a5").strip()
        massive_api_key = os.getenv("MASSIVE_API_KEY", "5_9g13XovqZWWKsHiPp9B8L9LaReqjbn").strip()
        webhook_url = get_app_setting(mode_key, "discord_webhook_url", "")
        provider_notices: list[str] = []
        saved_models = list_saved_models(mode_key)
        saved_evaluations = list_evaluation_snapshots(mode_key)
        present_html = ""
        run_all_html = ""
        run_all_rows = ""
        message_html = ""
        current_evaluation_payload: Dict[str, object] | None = None
        feature_name_map = {
            key: get_strategy_feature_builder(key).names()
            for key in (
                "feature2",
                "dqn",
                "fvg2",
                "fvg3",
                "rsi_thresholds",
                "stoch_rsi_thresholds",
                "derivative",
                "derivative2",
                "ema",
                "bollinger_bands",
                "vwap_anchor",
                "vwap_intraday_reversion",
                "vwap_intraday_momentum",
                "vwap_intraday_5m_session",
                "vwap_breakout_reversion_regime",
                "open15_orb_intraday",
                "open15_vwap_reclaim_intraday",
                "open15_trend_momentum_daytrade",
                "open15_dual_breakout_daytrade",
                "open15_dual_breakout_daytrade_plus",
                "open15_dual_breakout_daytrade_scalp",
                "vwap_momentum_trend_5m_conservative",
                "vwap_momentum_trend_5m_pullback",
                "vwap_volume_long_momentum_5m",
                "vwap_volume_regime_adaptive_5m",
                "vwap_volume_first5_trend_momentum_5m",
                "vwap_volume_profile_first5_trend_momentum_5m",
                "close_hold_reversion",
                "close_hold_momentum",
                "new",
                "legacy",
            )
        }

        if request.method == "POST":
            try:
                if mode == "provider_toggle":
                    toggled_provider = request.form.get("toggle_to", "yfinance").strip().lower()
                    data_provider = toggled_provider if toggled_provider in ("yfinance", "twelvedata", "massive") else "yfinance"
                    message_html = f"<p style='color:#7bd88f;'><strong>Data provider:</strong> {data_provider}</p>"
                elif mode == "run_all_monitor":
                    monitor_action = request.form.get("monitor_action", "").strip()
                    webhook_input = request.form.get("discord_webhook_url", "").strip()
                    if webhook_input:
                        webhook_url = webhook_input
                        set_app_setting(mode_key, "discord_webhook_url", webhook_url)
                    if monitor_action == "start":
                        if not webhook_url:
                            raise ValueError("Please enter and save a Discord webhook URL before starting continuous mode.")
                        RUN_ALL_MONITOR.start(
                            mode=mode_key,
                            long_only=is_spot,
                            data_provider=data_provider,
                            twelve_api_key=twelve_api_key,
                            massive_api_key=massive_api_key,
                            webhook_url=webhook_url,
                        )
                        message_html = (
                            "<p style='color:#7bd88f;'><strong>Continuous Run All mode started.</strong> "
                            "Checks every aligned 5 minutes (:00, :05, :10, ...).</p>"
                        )
                    elif monitor_action == "stop":
                        RUN_ALL_MONITOR.stop(mode_key, webhook_url)
                        message_html = "<p style='color:#7bd88f;'><strong>Continuous Run All mode stopped.</strong></p>"
                    elif monitor_action == "save_webhook":
                        if not webhook_url:
                            raise ValueError("Please enter a Discord webhook URL.")
                        message_html = "<p style='color:#7bd88f;'><strong>Discord webhook URL saved.</strong></p>"
                    else:
                        raise ValueError("Unsupported monitor action.")
                elif mode == "saved_eval":
                    eval_action = request.form.get("eval_action", "").strip()
                    if eval_action == "save":
                        payload_raw = request.form.get("evaluation_payload", "").strip()
                        snapshot_name = request.form.get("evaluation_name", "").strip()
                        if not payload_raw:
                            raise ValueError("No evaluation payload was provided.")
                        payload = json.loads(payload_raw)
                        if not isinstance(payload, dict):
                            raise ValueError("Invalid evaluation payload.")
                        if not snapshot_name:
                            raise ValueError("Please provide a name for the saved evaluation.")
                        save_evaluation_snapshot(mode_key, snapshot_name, payload)
                        message_html = f"<p style='color:#7bd88f;'><strong>Saved evaluation:</strong> {escape(snapshot_name)}</p>"
                        current_evaluation_payload = payload
                        result_html = str(payload.get("result_html", ""))
                        form_state = payload.get("form_state", {})
                        if isinstance(form_state, dict):
                            ticker = str(form_state.get("ticker", ticker))
                            interval = str(form_state.get("interval", interval))
                            rows = str(form_state.get("rows", rows))
                            split_style = str(form_state.get("split_style", split_style))
                            evaluation_split_raw = str(form_state.get("evaluation_split", evaluation_split_raw)).strip()
                            feature_set = normalize_feature_set(str(form_state.get("feature_set", feature_set)))
                            buy_threshold_raw = str(form_state.get("buy_threshold", buy_threshold_raw))
                            sell_threshold_raw = str(form_state.get("sell_threshold", sell_threshold_raw))
                            selected_model = str(form_state.get("selected_model", selected_model))
                            model_name = str(form_state.get("model_name", model_name))
                            stop_loss_strategy_raw = str(form_state.get("stop_loss_strategy", stop_loss_strategy_raw))
                            fixed_stop_pct_raw = str(form_state.get("fixed_stop_pct", fixed_stop_pct_raw))
                            take_profit_pct_raw = str(form_state.get("take_profit_pct", take_profit_pct_raw))
                            max_hold_bars_raw = str(form_state.get("max_hold_bars", max_hold_bars_raw))
                            data_provider = str(form_state.get("data_provider", data_provider)).strip().lower()
                            dqn_episodes_raw = str(form_state.get("dqn_episodes", dqn_episodes_raw))
                            prediction_horizon_raw = str(form_state.get("prediction_horizon", prediction_horizon_raw))
                            use_manual_weights_raw = str(form_state.get("use_manual_weights", use_manual_weights_raw)).strip().lower()
                            manual_weights_json = str(form_state.get("manual_feature_weights", manual_weights_json)).strip()
                            monte_carlo_method = str(form_state.get("monte_carlo_method", monte_carlo_method)).strip().lower()
                            monte_carlo_n_sim_raw = str(form_state.get("monte_carlo_n_sim", monte_carlo_n_sim_raw)).strip()
                            monte_carlo_block_size_raw = str(form_state.get("monte_carlo_block_size", monte_carlo_block_size_raw)).strip()
                            monte_carlo_seed_raw = str(form_state.get("monte_carlo_seed", monte_carlo_seed_raw)).strip()
                    elif eval_action == "open":
                        snapshot_id = int(request.form.get("evaluation_id", "0"))
                        snapshot = load_evaluation_snapshot(mode_key, snapshot_id)
                        payload = snapshot["payload"]
                        if not isinstance(payload, dict):
                            raise ValueError("Saved evaluation payload is invalid.")
                        current_evaluation_payload = payload
                        result_html = str(payload.get("result_html", ""))
                        form_state = payload.get("form_state", {})
                        if not isinstance(form_state, dict):
                            raise ValueError("Saved evaluation form state is invalid.")
                        ticker = str(form_state.get("ticker", ticker))
                        interval = str(form_state.get("interval", interval))
                        rows = str(form_state.get("rows", rows))
                        split_style = str(form_state.get("split_style", split_style))
                        evaluation_split_raw = str(form_state.get("evaluation_split", evaluation_split_raw)).strip()
                        feature_set = normalize_feature_set(str(form_state.get("feature_set", feature_set)))
                        buy_threshold_raw = str(form_state.get("buy_threshold", buy_threshold_raw))
                        sell_threshold_raw = str(form_state.get("sell_threshold", sell_threshold_raw))
                        selected_model = str(form_state.get("selected_model", selected_model))
                        model_name = str(form_state.get("model_name", model_name))
                        stop_loss_strategy_raw = str(form_state.get("stop_loss_strategy", stop_loss_strategy_raw))
                        fixed_stop_pct_raw = str(form_state.get("fixed_stop_pct", fixed_stop_pct_raw))
                        take_profit_pct_raw = str(form_state.get("take_profit_pct", take_profit_pct_raw))
                        max_hold_bars_raw = str(form_state.get("max_hold_bars", max_hold_bars_raw))
                        dqn_episodes_raw = str(form_state.get("dqn_episodes", dqn_episodes_raw))
                        prediction_horizon_raw = str(form_state.get("prediction_horizon", prediction_horizon_raw))
                        use_manual_weights_raw = str(form_state.get("use_manual_weights", use_manual_weights_raw)).strip().lower()
                        manual_weights_json = str(form_state.get("manual_feature_weights", manual_weights_json)).strip()
                        monte_carlo_method = str(form_state.get("monte_carlo_method", monte_carlo_method)).strip().lower()
                        monte_carlo_n_sim_raw = str(form_state.get("monte_carlo_n_sim", monte_carlo_n_sim_raw)).strip()
                        monte_carlo_block_size_raw = str(form_state.get("monte_carlo_block_size", monte_carlo_block_size_raw)).strip()
                        monte_carlo_seed_raw = str(form_state.get("monte_carlo_seed", monte_carlo_seed_raw)).strip()
                        message_html = (
                            f"<p style='color:#7bd88f;'><strong>Loaded saved evaluation:</strong> "
                            f"{escape(str(snapshot.get('name', 'Saved evaluation')))}</p>"
                        )
                    elif eval_action == "delete":
                        snapshot_id = int(request.form.get("evaluation_id", "0"))
                        delete_evaluation_snapshot(mode_key, snapshot_id)
                        message_html = "<p style='color:#7bd88f;'><strong>Deleted saved evaluation.</strong></p>"
                    else:
                        raise ValueError("Unsupported saved evaluation action.")
                    saved_evaluations = list_evaluation_snapshots(mode_key)
                elif mode == "present":
                    present_row_count = int(present_rows)
                    present_buy_threshold, present_sell_threshold = parse_thresholds(present_buy_raw, present_sell_raw)
                    present_stop_loss_strategy = parse_stop_loss_strategy(present_stop_loss_strategy_raw)
                    present_fixed_stop_pct = 2.0
                    if present_stop_loss_strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
                        present_fixed_stop_pct = validate_fixed_stop_pct(float(present_fixed_stop_pct_raw or "2.0"))
                    dataset, provider_notice = fetch_market_rows(
                        ticker=present_ticker,
                        interval=present_interval,
                        row_count=present_row_count,
                        provider=data_provider,
                        twelve_api_key=twelve_api_key,
                        massive_api_key=massive_api_key,
                        prediction_horizon=int(prediction_horizon_raw or "5"),
                    )
                    if provider_notice:
                        provider_notices.append(provider_notice)
                    present_rows_used_note = "" if len(dataset) >= present_row_count else f"Only {len(dataset)} frames were available and used for this run."
                    latest_row = dataset[-1]
                    present_dqn_episodes = int(dqn_episodes_raw or "120")
                    if present_dqn_episodes < 1:
                        raise ValueError("DQN episodes must be at least 1.")
                    if present_model == "__new__":
                        bundle = train_strategy_models(dataset, split_style=split_style, feature_set=feature_set, dqn_episodes=present_dqn_episodes)
                    else:
                        bundle = load_model_bundle(mode_key, present_model)
                    prediction = predict_signal(
                        bundle,
                        latest_row,
                        buy_threshold=present_buy_threshold,
                        sell_threshold=present_sell_threshold,
                        long_only=is_spot,
                    )
                    present_stop_price = stop_loss_price(
                        strategy=present_stop_loss_strategy,
                        action=str(prediction["action"]),
                        reference_price=float(latest_row.get("close", 0.0)),
                        expected_return=float(prediction["expected_return"]),
                        fixed_pct=present_fixed_stop_pct,
                        atr_fraction=float(latest_row.get("atr_frac", 0.0)),
                        model_mae=MODEL_MAE_DEFAULT,
                    )
                    present_stop_price_html = f"{present_stop_price:.4f}" if present_stop_price is not None else "n/a"
                    present_html = f"""
                    <section class="results">
                      <article class="card">
                        <h2>Present Mode • {present_ticker} ({present_interval})</h2>
                        <p class="muted">Model: {"Freshly trained on current dataset" if present_model == "__new__" else present_model}</p>
                        <p class="muted">{present_rule_text} (BUY&gt;{present_buy_threshold:.2f}, SELL&lt;{present_sell_threshold:.2f}).</p>
                        <p class="muted">{present_rows_used_note or f"Using requested {present_row_count} frames."}</p>
                        <p><span class="muted">Expected Return (next candle)</span> <strong>{prediction['expected_return']:+.4%}</strong></p>
                        <p><span class="muted">P(Up)</span> <strong>{prediction['p_up']:.2%}</strong></p>
                        <p><span class="muted">Stop Loss Strategy</span> <strong>{present_stop_loss_strategy.value}</strong></p>
                        <p><span class="muted">Calculated Stop Price</span> <strong>{present_stop_price_html}</strong></p>
                        <p><span class="muted">Action</span> <strong>{prediction['action']}</strong></p>
                      </article>
                    </section>
                    """
                elif mode == "present_all":
                    model_configs = load_model_configs(mode_key)
                    model_configs = {name: get_model_config(name, model_configs) for name in saved_models}
                    model_configs["__ui_data_provider__"] = data_provider
                    model_configs["__ui_twelve_api_key__"] = twelve_api_key
                    model_configs["__ui_massive_api_key__"] = massive_api_key
                    run_all_rows = build_run_all_rows(saved_models, model_configs, mode=mode_key, long_only=is_spot)
                    run_all_html = "<p class='muted'>Latest outputs for all models currently included in Run All.</p>"
                else:
                    row_count = int(rows)
                    prediction_horizon = int(prediction_horizon_raw or "5")
                    if prediction_horizon < 1:
                        raise ValueError("Prediction horizon must be at least 1.")
                    buy_threshold, sell_threshold = parse_thresholds(buy_threshold_raw, sell_threshold_raw)
                    stop_loss_strategy = parse_stop_loss_strategy(stop_loss_strategy_raw)
                    fixed_stop_pct = 2.0
                    if stop_loss_strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
                        fixed_stop_pct = validate_fixed_stop_pct(float(fixed_stop_pct_raw or "2.0"))
                    take_profit_pct = 0.0 if not take_profit_pct_raw else validate_take_profit_pct(float(take_profit_pct_raw))
                    max_hold_bars = 0 if not max_hold_bars_raw else validate_max_hold_bars(int(max_hold_bars_raw))
                    stop_loss_config = StopLossConfig(
                        strategy=stop_loss_strategy,
                        fixed_pct=fixed_stop_pct,
                        take_profit_pct=take_profit_pct,
                        max_hold_bars=max_hold_bars,
                        model_mae=MODEL_MAE_DEFAULT,
                        time_decay_bars=25,
                    )
                    if split_style not in ("shuffled", "chronological"):
                        raise ValueError("Split style must be either shuffled (legacy) or chronological (time-aware).")
                    if evaluation_split_raw == "":
                        evaluation_split = 0.25
                    else:
                        evaluation_split = float(evaluation_split_raw)
                        if not (0.0 < evaluation_split < 1.0):
                            raise ValueError("Evaluation split must be between 0 and 1 (exclusive).")
                    train_ratio = 1.0 - evaluation_split
                    dqn_episodes = int(dqn_episodes_raw or "120")
                    if dqn_episodes < 1:
                        raise ValueError("DQN episodes must be at least 1.")
                    monte_carlo_n_sim = int(monte_carlo_n_sim_raw or "500")
                    if monte_carlo_n_sim < 1:
                        raise ValueError("Monte Carlo simulations must be at least 1.")
                    monte_carlo_block_size = int(monte_carlo_block_size_raw or "20")
                    if monte_carlo_block_size < 1:
                        raise ValueError("Monte Carlo block size must be at least 1.")
                    monte_carlo_seed = int(monte_carlo_seed_raw) if monte_carlo_seed_raw else None
                    use_manual_weights = use_manual_weights_raw == "yes"
                    tickers = parse_csv_values(ticker, uppercase=True)
                    if not tickers:
                        raise ValueError("Please enter at least one ticker symbol.")
                    model_names = parse_csv_values(model_name, uppercase=False) if model_name else []
                    if len(tickers) > 1:
                        if train_action == "evaluate_update":
                            raise ValueError("Evaluate + Update Preset currently supports single-ticker runs only.")
                        if use_manual_weights:
                            raise ValueError("Manual feature weights currently support single-ticker runs only.")
                        if selected_model != "__new__":
                            raise ValueError("Multi-ticker run only supports training new models (not loading an existing saved model).")
                        if model_names and len(model_names) != len(tickers):
                            raise ValueError("When training multiple tickers, provide the same number of model names as tickers.")
                        multi_rows = []
                        model_configs = load_model_configs(mode_key)
                        multi_rows_used_notes = []
                        for idx, ticker_symbol in enumerate(tickers):
                            try:
                                dataset, provider_notice = fetch_market_rows(
                                    ticker=ticker_symbol,
                                    interval=interval,
                                    row_count=row_count,
                                    provider=data_provider,
                                    twelve_api_key=twelve_api_key,
                                    massive_api_key=massive_api_key,
                                    prediction_horizon=prediction_horizon,
                                )
                                if provider_notice:
                                    provider_notices.append(provider_notice)
                                if len(dataset) < row_count:
                                    multi_rows_used_notes.append(f"{ticker_symbol}: {len(dataset)} frames used")
                                bundle = train_strategy_models(dataset, split_style=split_style, feature_set=feature_set, dqn_episodes=dqn_episodes, test_ratio=evaluation_split)
                                metrics = evaluate_bundle(
                                    bundle,
                                    bundle["x_test_raw"],
                                    bundle["y_test_ret"],
                                    bundle["y_test_dir"],
                                    eval_rows=dataset,
                                    split_style=split_style,
                                    buy_threshold=buy_threshold,
                                    sell_threshold=sell_threshold,
                                    allow_short=allow_short,
                                    stop_loss=stop_loss_config,
                                    monte_carlo_method=monte_carlo_method,
                                    monte_carlo_n_sim=monte_carlo_n_sim,
                                    monte_carlo_block_size=monte_carlo_block_size,
                                    monte_carlo_seed=monte_carlo_seed,
                                )
                                trained_model_name = ""
                                if train_action == "train":
                                    candidate_name = model_names[idx] if model_names else ""
                                    trained_model_name = sanitize_model_name(candidate_name) if candidate_name else build_default_model_name(
                                        ticker=ticker_symbol,
                                        interval=interval,
                                        row_count=row_count,
                                        feature_set=feature_set,
                                        prediction_horizon=prediction_horizon,
                                    )
                                    bundle["historical_monte_carlo"] = metrics.get("monte_carlo")
                                    bundle["forward_monte_carlo_train"] = metrics.get("forward_buy_now")
                                    save_model_bundle(mode_key, trained_model_name, bundle)
                                    model_configs[trained_model_name] = {
                                        "ticker": ticker_symbol,
                                        "interval": interval,
                                        "rows": row_count,
                                        "include_in_run_all": True,
                                        "buy_threshold": buy_threshold,
                                        "sell_threshold": sell_threshold,
                                        "stop_loss_strategy": stop_loss_strategy.value,
                                        "fixed_stop_pct": fixed_stop_pct,
                                        "take_profit_pct": take_profit_pct,
                                        "max_hold_bars": max_hold_bars,
                                        "prediction_horizon": prediction_horizon,
                                    }
                                multi_rows.append(
                                    "<tr>"
                                    f"<td>{ticker_symbol}</td>"
                                    f"<td>{trained_model_name or '(not saved)'}</td>"
                                    f"<td>{metrics['accuracy']:.4f}</td>"
                                    f"<td>{metrics['mse']:.8f}</td>"
                                    f"<td>{metrics['strategy']['total_return']:+.2%}</td>"
                                    f"<td>{int(metrics['strategy']['trade_count'])}</td>"
                                    "</tr>"
                                )
                            except Exception as exc:
                                error_message = str(exc)
                                print(f"Multi-ticker run skipped {ticker_symbol}: {error_message}")
                                multi_rows.append(
                                    "<tr>"
                                    f"<td>{ticker_symbol}</td>"
                                    "<td>(error)</td>"
                                    "<td>—</td>"
                                    "<td>—</td>"
                                    f"<td colspan='2'>Error: {error_message}</td>"
                                    "</tr>"
                                )
                        if train_action == "train":
                            save_model_configs(mode_key, model_configs)
                            saved_models = list_saved_models(mode_key)
                        result_html = (
                            "<section class='results'>"
                            "<div class='section-heading'>"
                            f"<h2>Multi-Ticker Results • {interval}</h2>"
                            f"<p class='muted'>Tickers: {', '.join(tickers)} | Rows: {row_count} | Split: {split_style} | Train/Eval: {train_ratio:.0%}/{evaluation_split:.0%}</p>"
                            f"<p class='muted'>{' | '.join(multi_rows_used_notes) if multi_rows_used_notes else f'Using requested {row_count} frames for each ticker.'}</p>"
                            "</div>"
                            "<article class='card table-card'>"
                            "<table>"
                            "<tr><th>Ticker</th><th>Saved Model</th><th>Accuracy</th><th>MSE</th><th>Strategy Return</th><th>Trades</th></tr>"
                            f"{''.join(multi_rows)}"
                            "</table>"
                            "</article>"
                            "</section>"
                        )
                    else:
                        ticker = tickers[0]
                    if len(tickers) == 1:
                        fetch_row_count = row_count
                        eval_tail_rows = row_count
                        anchored_eval_note = ""
                        if (
                            selected_model != "__new__"
                            and train_action in evaluate_like_actions
                            and split_style == "chronological"
                        ):
                            model_configs = load_model_configs(mode_key)
                            model_cfg = get_model_config(selected_model, model_configs)
                            configured_rows_raw = model_cfg.get("rows", row_count)
                            try:
                                configured_rows = int(configured_rows_raw)
                            except (TypeError, ValueError):
                                configured_rows = row_count
                            if configured_rows >= row_count:
                                fetch_row_count = configured_rows
                                anchored_eval_note = (
                                    f"Evaluation anchored to model training window ({configured_rows} rows) "
                                    f"and scored on its latest {row_count} rows."
                                )
                        dataset, provider_notice = fetch_market_rows(
                            ticker=ticker,
                            interval=interval,
                            row_count=fetch_row_count,
                            provider=data_provider,
                            twelve_api_key=twelve_api_key,
                            massive_api_key=massive_api_key,
                            prediction_horizon=prediction_horizon,
                        )
                        if provider_notice:
                            provider_notices.append(provider_notice)
                        if train_action in evaluate_like_actions:
                            if anchored_eval_note:
                                rows_used_note = anchored_eval_note
                            else:
                                rows_used_note = ""
                            if len(dataset) < fetch_row_count:
                                availability_note = f"Only {len(dataset)} frames were available and used for evaluation."
                                rows_used_note = f"{rows_used_note} {availability_note}".strip() if rows_used_note else availability_note
                            if train_action == "evaluate_historical":
                                rows_used_note = (
                                    "Historical evaluation mode: forcing chronological split on downloaded market history."
                                    + (f" {rows_used_note}" if rows_used_note else "")
                                )
                            eval_rows = dataset[-eval_tail_rows:] if eval_tail_rows > 0 else dataset
                        else:
                            rows_used_note = "" if len(dataset) >= row_count else f"Only {len(dataset)} frames were available and used for training."
                            _, eval_rows = train_test_split(dataset, test_ratio=evaluation_split, split_style=split_style)
                        eval_start_date, eval_end_date = extract_timestamp_range(eval_rows)
                        y_test_ret = [r["return_next"] for r in eval_rows]
                        y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]
                        if selected_model != "__new__":
                            if use_manual_weights:
                                raise ValueError("Manual feature weights cannot be used while loading an existing saved model.")
                            loaded = load_model_bundle(mode_key, selected_model)
                            loaded_feature_set = infer_bundle_feature_set(loaded)
                            features = get_strategy_feature_builder(loaded_feature_set)
                            x_test_raw = features.transform(eval_rows)
                            metrics = evaluate_bundle(
                                loaded,
                                x_test_raw,
                                y_test_ret,
                                y_test_dir,
                                eval_rows=dataset,
                                split_style=split_style,
                                buy_threshold=buy_threshold,
                                sell_threshold=sell_threshold,
                                allow_short=allow_short,
                                stop_loss=stop_loss_config,
                                monte_carlo_method=monte_carlo_method,
                                monte_carlo_n_sim=monte_carlo_n_sim,
                                monte_carlo_block_size=monte_carlo_block_size,
                                monte_carlo_seed=monte_carlo_seed,
                            )
                            metrics["train_size"] = "saved-model"
                            metrics["loaded_model"] = selected_model
                        else:
                            if use_manual_weights:
                                if feature_set == "dqn":
                                    raise ValueError("Manual feature weights are not supported with the DQN feature pipeline.")
                                feature_builder = get_strategy_feature_builder(feature_set)
                                feature_names = feature_builder.names()
                                manual_weights = parse_manual_feature_weights(manual_weights_json, len(feature_names))
                                bundle = {
                                    "feature_names": feature_names,
                                    "feature_set": feature_set,
                                    "means": [0.0] * len(feature_names),
                                    "stds": [1.0] * len(feature_names),
                                    "lin_weights": manual_weights,
                                    "lin_bias": 0.0,
                                    "logit_weights": manual_weights,
                                    "logit_bias": 0.0,
                                    "train_size": "manual-weights",
                                    "test_size": len(y_test_ret),
                                    "split_style": split_style,
                                }
                                x_test_raw = feature_builder.transform(eval_rows)
                            else:
                                model_test_ratio = 0.0 if train_action in evaluate_like_actions else evaluation_split
                                bundle = train_strategy_models(
                                    dataset,
                                    split_style=split_style,
                                    feature_set=feature_set,
                                    dqn_episodes=dqn_episodes,
                                    test_ratio=model_test_ratio,
                                )
                                if train_action in evaluate_like_actions:
                                    bundle_feature_set = infer_bundle_feature_set(bundle)
                                    feature_builder = get_strategy_feature_builder(bundle_feature_set)
                                    x_test_raw = feature_builder.transform(eval_rows)
                                else:
                                    x_test_raw = bundle["x_test_raw"]
                            metrics = evaluate_bundle(
                                bundle,
                                x_test_raw,
                                y_test_ret if use_manual_weights or train_action in evaluate_like_actions else bundle["y_test_ret"],
                                y_test_dir if use_manual_weights or train_action in evaluate_like_actions else bundle["y_test_dir"],
                                eval_rows=dataset,
                                split_style=split_style,
                                buy_threshold=buy_threshold,
                                sell_threshold=sell_threshold,
                                allow_short=allow_short,
                                stop_loss=stop_loss_config,
                                monte_carlo_method=monte_carlo_method,
                                monte_carlo_n_sim=monte_carlo_n_sim,
                                monte_carlo_block_size=monte_carlo_block_size,
                                monte_carlo_seed=monte_carlo_seed,
                            )
                            metrics["train_size"] = bundle["train_size"]
                            if use_manual_weights:
                                metrics["manual_weights"] = True
                            if train_action == "train":
                                model_name_to_save = sanitize_model_name(model_name) if model_name else build_default_model_name(
                                    ticker=ticker,
                                    interval=interval,
                                    row_count=row_count,
                                    feature_set=feature_set,
                                    prediction_horizon=prediction_horizon,
                                )
                                bundle["historical_monte_carlo"] = metrics.get("monte_carlo")
                                bundle["forward_monte_carlo_train"] = metrics.get("forward_buy_now")
                                save_model_bundle(mode_key, model_name_to_save, bundle)
                                model_configs = load_model_configs(mode_key)
                                model_configs[model_name_to_save] = {
                                    "ticker": ticker,
                                    "interval": interval,
                                    "rows": row_count,
                                    "include_in_run_all": True,
                                    "buy_threshold": buy_threshold,
                                    "sell_threshold": sell_threshold,
                                    "stop_loss_strategy": stop_loss_strategy.value,
                                    "fixed_stop_pct": fixed_stop_pct,
                                    "take_profit_pct": take_profit_pct,
                                    "max_hold_bars": max_hold_bars,
                                    "prediction_horizon": prediction_horizon,
                                }
                                save_model_configs(mode_key, model_configs)
                                metrics["saved_model"] = model_name_to_save
                                saved_models = list_saved_models(mode_key)

                        if train_action == "evaluate_update":
                            if selected_model == "__new__":
                                raise ValueError("Evaluate + Update Preset requires selecting an existing saved model.")
                            if use_manual_weights:
                                raise ValueError("Evaluate + Update Preset cannot be used with manual feature weights.")
                            model_configs = load_model_configs(mode_key)
                            model_configs[selected_model] = {
                                "ticker": ticker,
                                "interval": interval,
                                "rows": row_count,
                                "include_in_run_all": True,
                                "buy_threshold": buy_threshold,
                                "sell_threshold": sell_threshold,
                                "stop_loss_strategy": stop_loss_strategy.value,
                                "fixed_stop_pct": fixed_stop_pct,
                                "take_profit_pct": take_profit_pct,
                                "max_hold_bars": max_hold_bars,
                                "prediction_horizon": prediction_horizon,
                            }
                            save_model_configs(mode_key, model_configs)
                            loaded_bundle = load_model_bundle(mode_key, selected_model)
                            loaded_bundle["historical_monte_carlo"] = metrics.get("monte_carlo")
                            loaded_bundle["forward_monte_carlo_train"] = metrics.get("forward_buy_now")
                            save_model_bundle(mode_key, selected_model, loaded_bundle)
                            message_html = (
                                "<p style='color:#7bd88f;'><strong>Updated preset and Monte Carlo distributions</strong> "
                                f"for model <strong>{escape(selected_model)}</strong>.</p>"
                            )
    
                        preview_rows = "".join(
                            f"<tr><td>{idx + 1}</td><td>{p['expected_return']:+.4%}</td><td>{p['p_up']:.2%}</td><td>{p['actual_return']:+.4%}</td></tr>"
                            for idx, p in enumerate(metrics["preview"])
                        )
                        model_msg = ""
                        if "saved_model" in metrics:
                            model_msg += f"<p><strong>Saved model:</strong> {metrics['saved_model']}</p>"
                        if "loaded_model" in metrics:
                            model_msg += f"<p><strong>Loaded model:</strong> {metrics['loaded_model']}</p>"
                        is_dqn_eval = str(metrics.get("model_type", "")).lower() == "dqn"
                        linear_weight_rows = "".join(
                            f"<tr><td>{name}</td><td>{weight:+.6f}</td></tr>" for name, weight in metrics["lin_weights"]
                        )
                        logistic_weight_rows = "".join(
                            f"<tr><td>{name}</td><td>{weight:+.6f}</td></tr>" for name, weight in metrics["logit_weights"]
                        )
                        calibration_rows = "".join(
                            "<tr>"
                            f"<td>{item['bucket_low']:.2f} - {item['bucket_high']:.2f}</td>"
                            f"<td>{int(item['count'])}</td>"
                            f"<td>{item['predicted_mean']:.4f}</td>"
                            f"<td>{item['actual_win_rate']:.4f}</td>"
                            "</tr>"
                            for item in metrics["calibration"]
                        )
                        confidence_edge_rows = "".join(
                            "<tr>"
                            f"<td>P &gt; {item.get('threshold', 0.0):.2f}</td>"
                            f"<td>{int(item['count'])}</td>"
                            f"<td>{item['accuracy']:.4f}</td>"
                            "</tr>"
                            for item in sorted(metrics.get("confidence_edge", {}).values(), key=lambda value: float(value.get("threshold", 0.0)))
                        )
                        pnl_signal_rows = "".join(
                            "<tr>"
                            f"<td>{item['bucket']}</td>"
                            f"<td>{int(item['count'])}</td>"
                            f"<td>{item['avg_pnl']:+.4%}</td>"
                            f"<td>{item['total_pnl']:+.2%}</td>"
                            "</tr>"
                            for item in metrics["pnl_by_signal_strength"]
                        )
                        pnl_regime_rows = "".join(
                            "<tr>"
                            f"<td>{item['regime']}</td>"
                            f"<td>{int(item['count'])}</td>"
                            f"<td>{item['avg_pnl']:+.4%}</td>"
                            f"<td>{item['total_pnl']:+.2%}</td>"
                            "</tr>"
                            for item in metrics["pnl_by_regime"]
                        )
                        def format_optional_price(value: object) -> str:
                            return "n/a" if value is None else f"{float(value):.4f}"
                        trade_log_rows = "".join(
                            "<tr>"
                            f"<td>{idx + 1}</td>"
                            f"<td>{escape(str(item.get('side', '')))}</td>"
                            f"<td>{escape(str(item.get('entry_label', 'n/a')))}</td>"
                            f"<td>{escape(str(item.get('exit_label', 'n/a')))}</td>"
                            f"<td>{float(item.get('entry_price', 0.0)):.4f}</td>"
                            f"<td>{format_optional_price(item.get('entry_raw_price'))}</td>"
                            f"<td>{float(item.get('exit_price', 0.0)):.4f}</td>"
                            f"<td>{format_optional_price(item.get('exit_raw_price'))}</td>"
                            f"<td>{float(item.get('bars_held', 0.0)):.0f}</td>"
                            f"<td>{float(item.get('gross_pnl', 0.0)):+.4%}</td>"
                            f"<td>{float(item.get('net_pnl', 0.0)):+.4%}</td>"
                            f"<td>{float(item.get('max_drawdown_during_trade', 0.0)):+.4%}</td>"
                            f"<td>{float(item.get('max_upside_during_trade', 0.0)):+.4%}</td>"
                            f"<td>{escape(str(item.get('exit_reason', 'signal')))}</td>"
                            "</tr>"
                            for idx, item in enumerate(metrics["strategy"].get("trade_log", []))
                        )
                        walk_forward_rows = "".join(
                            "<tr>"
                            f"<td>{int(item['window'])}</td>"
                            f"<td>{int(item['train_size'])}</td>"
                            f"<td>{int(item['test_size'])}</td>"
                            f"<td>{item['accuracy']:.4f}</td>"
                            f"<td>{item['mse']:.8f}</td>"
                            "</tr>"
                            for item in metrics["walk_forward"]
                        )
                        ablation_rows = "".join(
                            "<tr>"
                            f"<td>{item['removed_feature']}</td>"
                            f"<td>{item['accuracy_delta']:+.4f}</td>"
                            f"<td>{item['mse_delta']:+.8f}</td>"
                            "</tr>"
                            for item in metrics["feature_ablation"]
                        )
                        feature_items = "".join(
                            f"<li><code>{feature_name}</code></li>"
                            for feature_name in metrics["features"]
                        )
                        hold_time_boxplot = render_hold_time_boxplot(
                            metrics["strategy"]["hold_time_stats"],
                            stroke=theme_border,
                            accent=theme_table_head,
                        )
                        monte_carlo_html = ""
                        monte_carlo_results = metrics.get("monte_carlo")
                        forward_buy_now = metrics.get("forward_buy_now")
                        latest_signal = metrics.get("latest_signal") if isinstance(metrics.get("latest_signal"), dict) else {}
                        if isinstance(monte_carlo_results, dict):
                            mc_summary = monte_carlo_results.get("summary", {})
                            mc_raw_results = monte_carlo_results.get("raw_results", [])
                            if isinstance(mc_summary, dict):
                                raw_total_returns = [
                                    float(item.get("total_return", 0.0))
                                    for item in mc_raw_results
                                    if isinstance(item, dict)
                                ]
                                distribution_chart_html = render_distribution_histogram(
                                    raw_total_returns,
                                    stroke=theme_border,
                                    fill=theme_table_head,
                                )
                                forward_buy_now_html = ""
                                if isinstance(forward_buy_now, dict):
                                    action_text = str(latest_signal.get("action", "hold")).upper()
                                    forward_buy_now_html = (
                                        "<details open>"
                                        "<summary>Forward Monte Carlo: If You Buy Now</summary>"
                                        f"<p><span class='muted'>Current signal</span> <strong>{action_text}</strong> "
                                        f"(P(Up)={float(latest_signal.get('p_up', 0.5)):.2%}, "
                                        f"Expected next return={float(latest_signal.get('expected_return', 0.0)):+.2%})</p>"
                                        f"<p><span class='muted'>Projection horizon</span> <strong>{int(forward_buy_now.get('horizon_bars', 0))} bars</strong> "
                                        f"from historical evaluation returns ({int(forward_buy_now.get('simulations', 0))} simulations)</p>"
                                        "<table>"
                                        "<tr><th>Metric</th><th>Value</th></tr>"
                                        f"<tr><td>Expected total return</td><td><strong>{float(forward_buy_now.get('expected_return', 0.0)):+.2%}</strong></td></tr>"
                                        f"<tr><td>Median total return</td><td>{float(forward_buy_now.get('median_return', 0.0)):+.2%}</td></tr>"
                                        f"<tr><td>Best / Worst return</td><td>{float(forward_buy_now.get('best_return', 0.0)):+.2%} / {float(forward_buy_now.get('worst_return', 0.0)):+.2%}</td></tr>"
                                        f"<tr><td>5th / 95th return percentile</td><td>{float(forward_buy_now.get('p5_return', 0.0)):+.2%} / {float(forward_buy_now.get('p95_return', 0.0)):+.2%}</td></tr>"
                                        f"<tr><td>Probability of profit / loss</td><td>{float(forward_buy_now.get('probability_profit', 0.0)):.2%} / {float(forward_buy_now.get('probability_loss', 0.0)):.2%}</td></tr>"
                                        f"<tr><td>Expected terminal value ($10,000)</td><td><strong>${float(forward_buy_now.get('expected_terminal_value', 0.0)):,.2f}</strong></td></tr>"
                                        f"<tr><td>Median terminal value ($10,000)</td><td>${float(forward_buy_now.get('median_terminal_value', 0.0)):,.2f}</td></tr>"
                                        f"<tr><td>5th / 95th terminal percentile</td><td>${float(forward_buy_now.get('p5_terminal_value', 0.0)):,.2f} / ${float(forward_buy_now.get('p95_terminal_value', 0.0)):,.2f}</td></tr>"
                                        "</table>"
                                        "<p class='muted'>Use this as a scenario range, not a guarantee. It re-samples your evaluation-period behavior.</p>"
                                        "</details>"
                                    )
                                monte_carlo_html = (
                                    "<article class='card'>"
                                    "<h3>Monte Carlo Robustness</h3>"
                                    f"<p><span class='muted'>Method</span> <strong>{escape(monte_carlo_method)}</strong></p>"
                                    f"<p><span class='muted'>Simulations</span> <strong>{int(monte_carlo_n_sim)}</strong></p>"
                                    f"<p><span class='muted'>Mean Return</span> <strong>{float(mc_summary.get('mean_return', 0.0)):+.2%}</strong></p>"
                                    f"<p><span class='muted'>Median Return</span> <strong>{float(mc_summary.get('median_return', 0.0)):+.2%}</strong></p>"
                                    f"<p><span class='muted'>Return Std Dev</span> <strong>{float(mc_summary.get('std_return', 0.0)):.2%}</strong></p>"
                                    f"<p><span class='muted'>Return Range</span> {float(mc_summary.get('min_return', 0.0)):+.2%} to {float(mc_summary.get('max_return', 0.0)):+.2%}</p>"
                                    f"<p><span class='muted'>5th / 95th Percentile</span> {float(mc_summary.get('p5_return', 0.0)):+.2%} / {float(mc_summary.get('p95_return', 0.0)):+.2%}</p>"
                                    f"<p><span class='muted'>CVaR 5%</span> <strong>{float(mc_summary.get('cvar_5_return', 0.0)):+.2%}</strong></p>"
                                    f"<p><span class='muted'>Log Mean / Median Return</span> {float(mc_summary.get('log_mean_return', 0.0)):+.2%} / {float(mc_summary.get('log_median_return', 0.0)):+.2%}</p>"
                                    f"<p><span class='muted'>Skewness / Kurtosis</span> {float(mc_summary.get('skewness', 0.0)):+.3f} / {float(mc_summary.get('kurtosis', 0.0)):+.3f}</p>"
                                    f"<p><span class='muted'>Mean / Median Sharpe</span> {float(mc_summary.get('mean_sharpe', 0.0)):.3f} / {float(mc_summary.get('median_sharpe', 0.0)):.3f}</p>"
                                    f"<p><span class='muted'>Mean / Worst Drawdown</span> {float(mc_summary.get('mean_drawdown', 0.0)):.2%} / {float(mc_summary.get('worst_drawdown', 0.0)):.2%}</p>"
                                    f"<p><span class='muted'>Probability of Loss</span> <strong>{float(mc_summary.get('probability_of_loss', 0.0)):.2%}</strong></p>"
                                    f"<p><span class='muted'>P(Return &lt; -50%)</span> {float(mc_summary.get('probability_of_large_loss', 0.0)):.2%}</p>"
                                    f"<p><span class='muted'>P(Ruin &lt; -90%)</span> {float(mc_summary.get('probability_of_ruin', 0.0)):.2%}</p>"
                                    "<details>"
                                    "<summary>Hidden: Return Distribution Plot</summary>"
                                    f"{distribution_chart_html}"
                                    "</details>"
                                    f"{forward_buy_now_html}"
                                    "</article>"
                                )
                        model_cards_html = ""
                        if is_dqn_eval:
                            action_counts = metrics["dqn_policy"]["action_counts"]
                            avg_q_values = metrics["dqn_policy"]["avg_q_values"]
                            action_returns = metrics["dqn_action_returns"]
                            recent_episode_rewards = metrics["dqn_episode_rewards"]
                            episode_rewards_text = ", ".join(f"{float(reward):+.4f}" for reward in recent_episode_rewards) if recent_episode_rewards else "n/a"
                            model_cards_html = (
                                "<article class='card'>"
                                "<h3>DQN Policy Model</h3>"
                                f"<p><span class='muted'>Final epsilon</span> <strong>{metrics['dqn_last_epsilon']:.4f}</strong></p>"
                                f"<p><span class='muted'>Avg Q-values (HOLD / BUY / SELL)</span><br>{avg_q_values['hold']:+.6f} / {avg_q_values['buy']:+.6f} / {avg_q_values['sell']:+.6f}</p>"
                                f"<p><span class='muted'>Action counts (test)</span><br>HOLD={int(action_counts['hold'])} BUY={int(action_counts['buy'])} SELL={int(action_counts['sell'])}</p>"
                                f"<p><span class='muted'>Estimated action return (HOLD / BUY / SELL)</span><br>{float(action_returns[0]):+.6f} / {float(action_returns[1]):+.6f} / {float(action_returns[2]):+.6f}</p>"
                                "</article>"
                                "<article class='card'>"
                                "<h3>DQN Test Error</h3>"
                                f"<p><span class='muted'>Expected-return MSE</span> <strong>{metrics['mse']:.8f}</strong></p>"
                                f"<p><span class='muted'>Expected-return MAE</span> <strong>{metrics['mae']:.8f}</strong></p>"
                                "<p class='muted'>Supervised linear/logistic metrics are hidden for DQN runs.</p>"
                                f"<p><span class='muted'>Recent episode rewards</span><br>{episode_rewards_text}</p>"
                                "</article>"
                            )
                        else:
                            model_cards_html = (
                                "<article class='card'>"
                                "<h3>Linear Model</h3>"
                                f"<p><span class='muted'>Test MSE</span> <strong>{metrics['mse']:.8f}</strong></p>"
                                f"<p><span class='muted'>Test MAE</span> <strong>{metrics['mae']:.8f}</strong></p>"
                                f"<p><span class='muted'>Zero baseline (MSE/MAE)</span><br>{metrics['baseline_zero_mse']:.8f} / {metrics['baseline_zero_mae']:.8f}</p>"
                                f"<p><span class='muted'>Edge vs baseline (MSE/MAE)</span><br>{metrics['mse_vs_zero_baseline']:+.8f} / {metrics['mae_vs_zero_baseline']:+.8f}</p>"
                                "</article>"
                                "<article class='card'>"
                                "<h3>Logistic Model</h3>"
                                f"<p><span class='muted'>Accuracy</span> <strong>{metrics['accuracy']:.4f}</strong></p>"
                                f"<p><span class='muted'>Always-UP baseline</span> {metrics['baseline_always_up_accuracy']:.4f} (edge {metrics['accuracy_vs_baseline']:+.4f})</p>"
                                f"<p><span class='muted'>Precision / Recall / F1</span><br>{metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}</p>"
                                f"<p><span class='muted'>Confusion Matrix</span><br>TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}</p>"
                                "</article>"
                            )
                        diagnostics_html = ""
                        if not is_dqn_eval:
                            diagnostics_html = f"""
                      <div class="table-grid">
                        <article class="card table-card">
                          <h3>Calibration Buckets</h3>
                          <table>
                            <tr><th>Bucket</th><th>Count</th><th>Pred Mean</th><th>Actual Win</th></tr>
                            {calibration_rows}
                          </table>
                        </article>
                        <article class="card table-card">
                          <h3>Confidence Edge</h3>
                          <table>
                            <tr><th>Rule</th><th>Count</th><th>Accuracy</th></tr>
                            {confidence_edge_rows}
                          </table>
                        </article>
                      </div>

                      <details>
                        <summary>Walk-forward & Feature Ablation</summary>
                        <div class="table-grid">
                          <article class="card table-card">
                            <h3>Walk-forward Validation</h3>
                            <table>
                              <tr><th>Window</th><th>Train</th><th>Test</th><th>Accuracy</th><th>MSE</th></tr>
                              {walk_forward_rows}
                            </table>
                          </article>
                          <article class="card table-card">
                            <h3>Feature Ablation</h3>
                            <table>
                              <tr><th>Removed Feature</th><th>Δ Accuracy</th><th>Δ MSE</th></tr>
                              {ablation_rows}
                            </table>
                          </article>
                        </div>
                      </details>

                      <details>
                        <summary>Error Analysis</summary>
                        <pre>{json.dumps(metrics['error_analysis'], indent=2)}</pre>
                      </details>

                      <div class="table-grid">
                        <article class="card table-card">
                          <h3>Linear Weights (Bias {metrics['lin_bias']:+.6f})</h3>
                          <table>
                            <tr><th>Feature</th><th>Weight</th></tr>
                            {linear_weight_rows}
                          </table>
                        </article>
                        <article class="card table-card">
                          <h3>Logistic Weights (Bias {metrics['logit_bias']:+.6f})</h3>
                          <table>
                            <tr><th>Feature</th><th>Weight</th></tr>
                            {logistic_weight_rows}
                          </table>
                        </article>
                      </div>
                            """
                        result_html = f"""
                    <section class="results">
                      <div class="section-heading">
                        <div class="results-heading-row">
                          <h2>Results • {ticker} ({interval})</h2>
                          <button type="button" id="saveEvaluationBtn" class="secondary save-eval-btn">Save</button>
                        </div>
                        <p class="muted">Rows: {row_count} | Train: {metrics['train_size']} | Test: {metrics['test_size']} | Split: {metrics['split_style']}</p>
                        <p class="muted">Evaluation window: {eval_start_date} → {eval_end_date}</p>
                        <p class="muted">{rows_used_note or f"Using requested {row_count} frames."}</p>
                        {model_msg}
                      </div>
                      <div class="card-grid model-overview-grid">
                        {model_cards_html}
                        <article class="card">
                          <h3>Decision Strategy</h3>
                          <p class="muted">{strategy_mode_text} · BUY P&gt;{metrics['strategy']['long_threshold']:.2f} · SELL P&lt;{metrics['strategy']['short_threshold']:.2f} · Stop {metrics['strategy']['stop_loss_strategy']} · Fee 0.00% · Slippage 0.02%</p>
                          <p><span class="muted">Total Return</span> <strong>{metrics['strategy']['total_return']:+.2%}</strong></p>
                          <p><span class="muted">Buy &amp; Hold Return (test rows)</span> <strong>{metrics['strategy']['buy_hold_total_return']:+.2%}</strong></p>
                          <p><span class="muted">Alpha (vs Buy &amp; Hold)</span> <strong>{float(metrics['strategy'].get('alpha', metrics['strategy']['total_return'] - metrics['strategy'].get('buy_hold_total_return', 0.0))):+.2%}</strong></p>
                          <p><span class="muted">CAPM Alpha (vs S&amp;P 500 Buy &amp; Hold)</span> <strong>{float(metrics['strategy'].get('alpha_capm_sp500_buy_hold', 0.0)):+.2%}</strong></p>
                          <p class="muted">(β={float(metrics['strategy'].get('beta_vs_sp500', 0.0)):.3f}, Rf={float(metrics['strategy'].get('risk_free_rate', 0.0)):.2%})</p>
                          <p><span class="muted">Sharpe</span> <strong>{metrics['strategy']['sharpe']:.3f}</strong></p>
                          <p><span class="muted">Max Drawdown</span> {metrics['strategy']['max_drawdown']:.2%}</p>
                          <p><span class="muted">Average Drawdown</span> {metrics['strategy']['avg_drawdown']:.2%}</p>
                          <p><span class="muted">Win Rate / Trades</span> {metrics['strategy']['win_rate']:.2%} / {int(metrics['strategy']['trade_count'])}</p>
                          <p><span class="muted">Average Gain per Trade</span> {metrics['strategy']['avg_gain_per_trade']:+.4%}</p>
                          <p><span class="muted">Max Loss per Trade</span> {metrics['strategy']['max_loss_per_trade']:+.4%}</p>
                          <p><span class="muted">Stop-loss Exits</span> {int(metrics['strategy']['stop_loss_exits'])}</p>
                          <p><span class="muted">Time-decay Exits</span> {int(metrics['strategy']['time_decay_exits'])} (limit: {int(metrics['strategy']['time_decay_bars'])} bars)</p>
                          <div style="margin-top:0.55rem;">
                            <p class="muted" style="margin-bottom:0.35rem;">Hold Time Distribution (bars/candles)</p>
                            {hold_time_boxplot}
                          </div>
                        </article>
                        {monte_carlo_html}
                      </div>

                      <article class="card">
                        <h3>Feature Set (Current Model Inputs)</h3>
                        <p class="muted">Total Features: <strong>{len(metrics['features'])}</strong> · Synced to current model configuration</p>
                        <ul class="feature-list">
                          {feature_items}
                        </ul>
                      </article>
                      {diagnostics_html}
    
                      <div class="table-grid">
                        <article class="card table-card">
                          <h3>PnL by Signal Strength</h3>
                          <table>
                            <tr><th>Bucket</th><th>Count</th><th>Avg PnL</th><th>Total PnL</th></tr>
                            {pnl_signal_rows}
                          </table>
                        </article>
                        <article class="card table-card">
                          <h3>PnL by Market Regime</h3>
                          <table>
                            <tr><th>Regime</th><th>Count</th><th>Avg PnL</th><th>Total PnL</th></tr>
                            {pnl_regime_rows}
                          </table>
                        </article>
                      </div>
                      <article class="card table-card">
                        <h3>Example Predictions</h3>
                        <table>
                          <tr><th>Row</th><th>Expected Return</th><th>P(Up)</th><th>Actual Return</th></tr>
                          {preview_rows}
                        </table>
                      </article>

                      <details>
                        <summary>Hidden: Trade-by-Trade Execution Log</summary>
                        <article class="card table-card">
                          <h3>Every Closed Trade</h3>
                          <p class="muted">Includes normalized model prices plus raw API prices for entry/exit, hold time, P&amp;L after costs/slippage, plus per-trade max drawdown/upside excursions.</p>
                          <table>
                            <tr><th>#</th><th>Side</th><th>Date Bought/Opened</th><th>Date Sold/Closed</th><th>Entry (Normalized)</th><th>Entry (Raw API)</th><th>Exit (Normalized)</th><th>Exit (Raw API)</th><th>Bars</th><th>Gross PnL</th><th>Net PnL</th><th>Max Drawdown (Trade)</th><th>Max Upside (Trade)</th><th>Exit Reason</th></tr>
                            {trade_log_rows if trade_log_rows else "<tr><td colspan='14' class='muted'>No closed trades for this evaluation.</td></tr>"}
                          </table>
                        </article>
                      </details>
                    </section>
                    """
                        current_evaluation_payload = {
                            "saved_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                            "form_state": {
                                "ticker": ticker,
                                "interval": interval,
                                "rows": str(row_count),
                                "split_style": split_style,
                                "evaluation_split": evaluation_split_raw,
                                "feature_set": feature_set,
                                "buy_threshold": buy_threshold_raw,
                                "sell_threshold": sell_threshold_raw,
                                "selected_model": selected_model,
                                "model_name": model_name,
                                "stop_loss_strategy": stop_loss_strategy_raw,
                                "fixed_stop_pct": fixed_stop_pct_raw,
                                "take_profit_pct": take_profit_pct_raw,
                                "max_hold_bars": max_hold_bars_raw,
                                "data_provider": data_provider,
                                "dqn_episodes": dqn_episodes_raw,
                                "prediction_horizon": prediction_horizon_raw,
                                "use_manual_weights": use_manual_weights_raw,
                                "manual_feature_weights": manual_weights_json,
                                "monte_carlo_method": monte_carlo_method,
                                "monte_carlo_n_sim": monte_carlo_n_sim_raw,
                                "monte_carlo_block_size": monte_carlo_block_size_raw,
                                "monte_carlo_seed": monte_carlo_seed_raw,
                            },
                            "result_html": result_html,
                        }
            except Exception as exc:
                error_html = f"<p style='color:red;'><strong>Error:</strong> {exc}</p>"

        provider_notice_html = ""
        if provider_notices:
            provider_notice_items = "".join(f"<li>{escape(note)}</li>" for note in dict.fromkeys(provider_notices))
            provider_notice_html = (
                "<div class='card' style='margin-top:1rem;'><h3>Data Provider Notices</h3>"
                "<p class='muted'>Some requests fell back to yfinance.</p>"
                f"<ul>{provider_notice_items}</ul></div>"
            )
        monitor_state = RUN_ALL_MONITOR.status(mode_key)
        monitor_running = bool(monitor_state.get("running"))
        worker_alive = bool(monitor_state.get("worker_alive"))
        worker_state = escape(str(monitor_state.get("worker_state", "stopped")))
        monitor_started_at = escape(format_display_time(monitor_state.get("started_at", "")))
        monitor_last_tick = escape(format_display_time(monitor_state.get("last_tick", "")))
        monitor_last_error = escape(str(monitor_state.get("last_error", "")))
        monitor_market_state = str(monitor_state.get("last_market_state", "unknown")).strip().lower()
        if monitor_market_state == "open":
            market_label = "Open"
        elif monitor_market_state == "closed":
            market_label = "Closed"
        else:
            market_label = "Unknown"
        worker_label = "Online" if worker_alive else "Offline"
        lifecycle_label = "Crashed" if worker_state == "crashed" else ("Running" if monitor_running else "Stopped")
        error_line = f"<br><span style='color:#ff7b7b;'><strong>Last error:</strong> {monitor_last_error}</span>" if monitor_last_error else ""
        monitor_status_html = (
            f"<p class='muted'><strong>Bot:</strong> {worker_label}"
            f" | <strong>Lifecycle:</strong> {lifecycle_label}"
            f" | <strong>Market:</strong> {market_label}"
            f" | <strong>Started:</strong> {monitor_started_at or 'n/a'}"
            f" | Last poll: {monitor_last_tick or 'n/a'}"
            f" | Tracked models: {int(monitor_state.get('last_actions_count', 0))}</p>"
            f"{error_line}"
        )

        current_eval_payload_json = (json.dumps(current_evaluation_payload).replace("</", "<\\/") if current_evaluation_payload else "null")
        saved_eval_items_json = json.dumps(saved_evaluations).replace("</", "<\\/")
        saved_eval_rows_html = "".join(
            (
                "<button type='button' class='saved-eval-item' "
                f"data-id='{int(item['id'])}' "
                f"data-name='{escape(str(item['name']))}' "
                f"data-updated-at='{escape(format_display_time(item['updated_at']))}'>"
                f"<strong>{escape(str(item['name']))}</strong>"
                f"<span>{escape(format_display_time(item['updated_at']))}</span>"
                "</button>"
            )
            for item in saved_evaluations
        )
        provider_cycle = {"yfinance": "twelvedata", "twelvedata": "massive", "massive": "yfinance"}
        provider_labels = {"yfinance": "YFinance", "twelvedata": "Twelve Data", "massive": "Massive"}
        next_provider = provider_cycle.get(data_provider, "yfinance")

        return f"""
        <html>
          <head><title>Quant Model Trainer</title><meta name="viewport" content="width=device-width, initial-scale=1" /></head>
          <body class="{'mobile-ui' if is_mobile_ui else 'desktop-ui'}">
            <style>
              :root {{
                --bg: {theme_bg};
                --panel: {theme_panel};
                --panel-2: {theme_panel2};
                --border: {theme_border};
                --text: {theme_text};
                --muted: {theme_muted};
                --accent: {theme_accent};
              }}
              * {{ box-sizing: border-box; }}
              body {{
                margin: 0;
                background: radial-gradient(circle at 8% -10%, {theme_glow_primary} 0%, transparent 40%), radial-gradient(circle at 92% -20%, {theme_glow_secondary} 0%, transparent 48%), linear-gradient(180deg, #090b10 0%, var(--bg) 58%);
                color: var(--text);
                font-family: Inter, Segoe UI, Arial, sans-serif;
              }}
              .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
              }}
              .topbar {{ position: sticky; top: 0; z-index: 50; background: {theme_topbar_bg}; border-bottom: 1px solid var(--border); backdrop-filter: blur(6px); }}
              .topbar-inner {{ max-width: 1200px; margin: 0 auto; padding: 0.9rem 2rem; display: flex; align-items: center; gap: 1rem; }}
              .brand {{ font-weight: 700; color: {theme_brand}; text-decoration: none; margin-right: auto; }}
              .mobile-menu-toggle {{ display: none; border: 1px solid var(--border); background: transparent; color: var(--text); border-radius: 8px; padding: 0.35rem 0.55rem; font-size: 1rem; line-height: 1; cursor: pointer; }}
              .mobile-menu-toggle:hover {{ background: {theme_tab_hover_bg}; }}
              .nav-links {{ display: flex; align-items: center; gap: 1rem; }}
              .tab-link {{ color: {theme_tab}; text-decoration: none; padding: 0.4rem 0.65rem; border-radius: 8px; border: 1px solid transparent; }}
              .tab-link:hover, .tab-link.active {{ color: {theme_tab_active}; border-color: var(--border); background: {theme_tab_hover_bg}; }}
              h1, h2, h3 {{ margin-top: 0; }}
              .card {{
                background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, {theme_shadow_alpha});
              }}
              .form-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 0.9rem;
              }}
              label {{ display: block; font-size: 0.92rem; color: var(--muted); }}
              input, select, button {{
                width: 100%;
                margin-top: 0.4rem;
                background: {theme_surface};
                color: var(--text);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.65rem 0.7rem;
              }}
              button {{
                background: var(--accent);
                color: #ffffff;
                font-weight: 600;
                border: none;
                cursor: pointer;
              }}
              .button-row {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 0.5rem;
              }}
              .manual-weights-grid {{
                display: grid;
                gap: 0.5rem;
                max-height: 250px;
                overflow-y: auto;
                padding-right: 0.35rem;
              }}
              .manual-weight-item {{
                display: grid;
                grid-template-columns: minmax(0, 1fr) 120px;
                gap: 0.5rem;
                align-items: center;
              }}
              .manual-weight-item code {{
                color: var(--text);
                font-size: 0.82rem;
                word-break: break-all;
              }}
              .manual-weight-item input {{
                margin-top: 0;
              }}
              .secondary {{
                background: {theme_secondary_bg};
                color: {theme_secondary_text};
                border: 1px solid var(--border);
              }}
              .card-grid, .table-grid {{
                display: grid;
                gap: 1rem;
                grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
              }}
              .model-overview-grid {{
                grid-template-columns: repeat(2, minmax(270px, 1fr));
              }}
              @media (max-width: 720px) {{
                .model-overview-grid {{
                  grid-template-columns: 1fr;
                }}
              }}
              .muted {{ color: var(--muted); }}
              table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9rem;
              }}
              th, td {{
                border-bottom: 1px solid var(--border);
                text-align: left;
                padding: 0.45rem 0.35rem;
              }}
              th {{ color: {theme_table_head}; font-weight: 600; }}
              td {{ color: var(--text); }}
              #runAllTable td,
              #runAllTable td strong {{
                color: var(--text) !important;
              }}
              pre {{
                white-space: pre-wrap;
                background: {theme_surface};
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.8rem;
              }}
              details {{
                background: {theme_details_bg};
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 0.7rem 0.9rem;
                margin-bottom: 1rem;
              }}
              summary {{
                cursor: pointer;
                color: {theme_heading};
                font-weight: 600;
                margin-bottom: 0.6rem;
              }}
              .loading-overlay {{
                position: fixed;
                inset: 0;
                z-index: 1000;
                display: none;
                align-items: center;
                justify-content: center;
                background: {theme_overlay};
                backdrop-filter: blur(4px);
              }}
              .loading-card {{
                width: min(520px, 92vw);
                background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                box-shadow: 0 12px 34px rgba(0, 0, 0, 0.42);
              }}
              .loading-title {{
                margin: 0 0 0.55rem;
                color: {theme_heading};
              }}
              .progress-track {{
                width: 100%;
                height: 14px;
                border-radius: 999px;
                background: {theme_surface};
                border: 1px solid var(--border);
                overflow: hidden;
              }}
              .progress-fill {{
                width: 0%;
                height: 100%;
                background: linear-gradient(90deg, {theme_progress_start} 0%, {theme_progress_end} 100%);
                transition: width 0.2s ease;
              }}
              .loading-meta {{
                margin-top: 0.55rem;
                display: flex;
                justify-content: space-between;
                color: var(--muted);
                font-size: 0.92rem;
              }}
              .saved-eval-actions {{
                display: flex;
                gap: 0.6rem;
                flex-wrap: wrap;
              }}
              .results-heading-row {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 0.5rem;
              }}
              .results-heading-row h2 {{
                margin: 0;
              }}
              .save-eval-btn {{
                margin-left: auto;
                width: auto;
                padding: 0.45rem 0.7rem;
                font-size: 0.82rem;
              }}
              .feature-list {{
                margin: 0.65rem 0 0;
                padding: 0;
                list-style: none;
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
              }}
              .feature-list li {{
                margin: 0;
              }}
              .feature-list code {{
                display: inline-block;
                background: {theme_surface};
                border: 1px solid var(--border);
                border-radius: 999px;
                padding: 0.25rem 0.6rem;
                color: var(--text);
                font-size: 0.82rem;
              }}
              @media (max-width: 720px) {{
                .results-heading-row {{
                  align-items: flex-start;
                  flex-direction: column;
                }}
                .save-eval-btn {{
                  margin-left: 0;
                }}
              }}
              .saved-eval-actions button {{
                width: auto;
                padding: 0.55rem 0.85rem;
              }}
              .modal {{
                position: fixed;
                inset: 0;
                z-index: 900;
                display: none;
                align-items: center;
                justify-content: center;
                background: rgba(0, 0, 0, 0.58);
              }}
              .modal-card {{
                width: min(560px, 94vw);
                max-height: 80vh;
                overflow: auto;
                background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 1rem;
              }}
              .saved-eval-list {{
                display: grid;
                gap: 0.6rem;
                margin-top: 0.8rem;
              }}
              .saved-eval-item {{
                text-align: left;
                background: {theme_surface};
                border: 1px solid var(--border);
              }}
              .saved-eval-item strong {{
                display: block;
                color: {theme_heading};
              }}
              .saved-eval-item span {{
                display: block;
                margin-top: 0.2rem;
                color: var(--muted);
                font-size: 0.82rem;
              }}
              body.mobile-ui .topbar-inner {{
                padding: 0.7rem 0.75rem;
                gap: 0.5rem;
                flex-wrap: wrap;
              }}
              body.mobile-ui .brand {{ margin-right: 0; }}
              body.mobile-ui .mobile-menu-toggle {{ display: inline-flex; align-items: center; justify-content: center; margin-left: auto; }}
              body.mobile-ui .nav-links {{ display: none; width: 100%; flex-direction: column; align-items: stretch; gap: 0.4rem; }}
              body.mobile-ui .nav-links.open {{ display: flex; }}
              body.mobile-ui .tab-link {{ flex: 1 1 100%; text-align: left; }}
              body.mobile-ui .container {{
                padding: 1rem 0.75rem;
              }}
              body.mobile-ui .button-row {{
                grid-template-columns: 1fr;
              }}
              body.mobile-ui table {{
                display: block;
                overflow-x: auto;
                white-space: nowrap;
              }}
              .context-menu {{
                position: fixed;
                z-index: 1100;
                display: none;
                min-width: 180px;
                border: 1px solid var(--border);
                border-radius: 10px;
                background: var(--panel-2);
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.45);
                padding: 0.25rem;
              }}
              .context-menu button {{
                width: 100%;
                text-align: left;
                border: none;
                background: transparent;
                color: var(--text);
                padding: 0.55rem 0.6rem;
              }}
              .context-menu button:hover {{
                background: {theme_surface};
              }}
              .provider-pill {{
                width: auto;
                margin-top: 0;
                font-size: 0.82rem;
                padding: 0.32rem 0.55rem;
              }}
              .topbar-btn {{
                font-size: 0.85rem;
                padding: 0.32rem 0.55rem;
                width: auto;
                margin-top: 0;
              }}
            </style>
            <nav class="topbar">
              <div class="topbar-inner">
                <a href="{home_href}" class="brand">{brand_label}</a>
                <button type="button" id="mobileNavToggle" class="mobile-menu-toggle" aria-label="Toggle menu" aria-controls="primaryNavMenu" aria-expanded="false">&#9776;</button>
                <div id="primaryNavMenu" class="nav-links">
                  <a href="{home_href}" class="tab-link active">Model</a>
                  <a href="{manage_href}" class="tab-link">Manage Models</a>
                  <a href="{run_models_href}" class="tab-link">Run Models</a>
                  <a href="{bots_href}" class="tab-link">Bots</a>
                  <button type="button" id="openEvaluationsBtn" class="secondary topbar-btn">Saved</button>
                  <form method="post" style="margin:0;">
                    <input type="hidden" name="mode" value="provider_toggle" />
                    <input type="hidden" name="toggle_to" value="{next_provider}" />
                    <input type="hidden" name="data_provider" value="{data_provider}" />
                    <button type="submit" class="secondary provider-pill">Data: {provider_labels.get(data_provider, 'YFinance')}</button>
                  </form>
                  <a href="{mode_switch_href}" class="tab-link">{mode_switch_label}</a>
                </div>
              </div>
            </nav>
            <div class="container">
            <h1>{trainer_heading}</h1>
            <form method="post" class="card">
              <input type="hidden" name="mode" value="train" />
              <input type="hidden" name="data_provider" value="{data_provider}" />
              <div class="form-grid">
              <label>Ticker:
                <input type="text" name="ticker" value="{ticker}" required />
              </label>
              <label>Interval:
                <select name="interval">
                  <option value="1d" {"selected" if interval == "1d" else ""}>Daily</option>
                  <option value="1h" {"selected" if interval == "1h" else ""}>Hourly</option>
                  <option value="15m" {"selected" if interval == "15m" else ""}>15 min</option>
                  <option value="5m" {"selected" if interval == "5m" else ""}>5 min</option>
                </select>
              </label>
              <label>Rows:
                <input type="number" min="50" name="rows" value="{rows}" required />
              </label>
              <label>Prediction Horizon (candles):
                <input type="number" min="1" step="1" name="prediction_horizon" value="{prediction_horizon_raw}" required />
              </label>
              <label>Split Style:
                <select name="split_style">
                  <option value="shuffled" {"selected" if split_style == "shuffled" else ""}>Legacy (shuffled)</option>
                  <option value="chronological" {"selected" if split_style == "chronological" else ""}>Time-aware (chronological)</option>
                </select>
              </label>
              <label>Evaluation Split (optional):
                <input type="number" min="0.01" max="0.99" step="0.01" name="evaluation_split" value="{evaluation_split_raw}" placeholder="0.25" />
              </label>
              <label>Feature Pipeline:
                <select name="feature_set">
                  <option value="feature2" {"selected" if feature_set == "feature2" else ""}>Feature 2 (default)</option>
                  <option value="hybrid_sharpe_core" {"selected" if feature_set == "hybrid_sharpe_core" else ""}>Hybrid Sharpe Core (EMA + Derivative)</option>
                  <option value="hybrid_sharpe_core_no_stack" {"selected" if feature_set == "hybrid_sharpe_core_no_stack" else ""}>Hybrid Sharpe Core (No EMA Stack Flags)</option>
                  <option value="hybrid_sharpe_momentum" {"selected" if feature_set == "hybrid_sharpe_momentum" else ""}>Hybrid Sharpe Momentum (expanded)</option>
                  <option value="hybrid_sharpe_selective" {"selected" if feature_set == "hybrid_sharpe_selective" else ""}>Hybrid Sharpe Selective (compact blend)</option>
                  <option value="hybrid_sharpe_regime" {"selected" if feature_set == "hybrid_sharpe_regime" else ""}>Hybrid Sharpe Regime (context aware)</option>
                  <option value="hybrid_sharpe_volume_flow" {"selected" if feature_set == "hybrid_sharpe_volume_flow" else ""}>Hybrid Sharpe Volume Flow (no-stack + participation)</option>
                  <option value="hybrid_sharpe_volume_regime" {"selected" if feature_set == "hybrid_sharpe_volume_regime" else ""}>Hybrid Sharpe Volume Regime (flow + gating)</option>
                  <option value="close_hold_reversion" {"selected" if feature_set == "close_hold_reversion" else ""}>Close Hold Reversion (1-2 day mean reversion)</option>
                  <option value="close_hold_momentum" {"selected" if feature_set == "close_hold_momentum" else ""}>Close Hold Momentum (1-2 day continuation)</option>
                  <option value="war_shock_reversion" {"selected" if feature_set == "war_shock_reversion" else ""}>War Shock Reversion (high-volatility snapback)</option>
                  <option value="war_shock_momentum" {"selected" if feature_set == "war_shock_momentum" else ""}>War Shock Momentum (high-volatility continuation)</option>
                  <option value="dqn" {"selected" if feature_set == "dqn" else ""}>DQN (Q-learning model)</option>
                  <option value="fvg2" {"selected" if feature_set == "fvg2" else ""}>FVG 2 (legacy split extremes)</option>
                  <option value="fvg3" {"selected" if feature_set == "fvg3" else ""}>FVG 3</option>
                  <option value="rsi_thresholds" {"selected" if feature_set == "rsi_thresholds" else ""}>RSI Thresholds (2 features)</option>
                  <option value="stoch_rsi_thresholds" {"selected" if feature_set == "stoch_rsi_thresholds" else ""}>Stoch RSI Thresholds (2 features)</option>
                  <option value="derivative" {"selected" if feature_set == "derivative" else ""}>Derivative set</option>
                  <option value="derivative2" {"selected" if feature_set == "derivative2" else ""}>Derivatives 2 set</option>
                  <option value="ema" {"selected" if feature_set == "ema" else ""}>EMA set</option>
                  <option value="bollinger_bands" {"selected" if feature_set == "bollinger_bands" else ""}>Bollinger Bands set</option>
                  <option value="vwap_anchor" {"selected" if feature_set == "vwap_anchor" else ""}>VWAP Anchor set</option>
                  <option value="vwap_intraday_reversion" {"selected" if feature_set == "vwap_intraday_reversion" else ""}>VWAP Intraday Reversion</option>
                  <option value="vwap_intraday_momentum" {"selected" if feature_set == "vwap_intraday_momentum" else ""}>VWAP Intraday Momentum</option>
                  <option value="vwap_intraday_5m_session" {"selected" if feature_set == "vwap_intraday_5m_session" else ""}>VWAP Intraday 5m Session (daily reset)</option>
                  <option value="vwap_breakout_reversion_regime" {"selected" if feature_set == "vwap_breakout_reversion_regime" else ""}>VWAP Breakout vs Reversion Regime</option>
                  <option value="open15_orb_intraday" {"selected" if feature_set == "open15_orb_intraday" else ""}>Open 15 ORB Intraday (max-2 trade style)</option>
                  <option value="open15_vwap_reclaim_intraday" {"selected" if feature_set == "open15_vwap_reclaim_intraday" else ""}>Open 15 VWAP Reclaim Intraday</option>
                  <option value="open15_trend_momentum_daytrade" {"selected" if feature_set == "open15_trend_momentum_daytrade" else ""}>Open 15 Trend Momentum Daytrade (max-2/day)</option>
                  <option value="open15_dual_breakout_daytrade" {"selected" if feature_set == "open15_dual_breakout_daytrade" else ""}>Open 15 Dual Breakout Daytrade (follow-through)</option>
                  <option value="open15_dual_breakout_daytrade_plus" {"selected" if feature_set == "open15_dual_breakout_daytrade_plus" else ""}>Open 15 Dual Breakout+ Daytrade (up to 3/day)</option>
                  <option value="open15_dual_breakout_daytrade_scalp" {"selected" if feature_set == "open15_dual_breakout_daytrade_scalp" else ""}>Open 15 Dual Breakout Scalp (up to 5/day)</option>
                  <option value="vwap_momentum_trend_5m_conservative" {"selected" if feature_set == "vwap_momentum_trend_5m_conservative" else ""}>VWAP Momentum Trend 5m (conservative, up to 4/day)</option>
                  <option value="vwap_momentum_trend_5m_pullback" {"selected" if feature_set == "vwap_momentum_trend_5m_pullback" else ""}>VWAP Momentum Trend 5m Pullback (up to 3/day)</option>
                  <option value="vwap_volume_long_momentum_5m" {"selected" if feature_set == "vwap_volume_long_momentum_5m" else ""}>VWAP + Volume Long Momentum 5m (intraday-only long bias)</option>
                  <option value="vwap_volume_regime_adaptive_5m" {"selected" if feature_set == "vwap_volume_regime_adaptive_5m" else ""}>VWAP + Volume Regime Adaptive 5m (trend vs mean-revert)</option>
                  <option value="vwap_volume_first5_trend_momentum_5m" {"selected" if feature_set == "vwap_volume_first5_trend_momentum_5m" else ""}>VWAP + Volume First-5 Trend Momentum 5m (target 1-2/day)</option>
                  <option value="vwap_volume_profile_first5_trend_momentum_5m" {"selected" if feature_set == "vwap_volume_profile_first5_trend_momentum_5m" else ""}>VWAP + Volume Profile First-5 Trend Momentum 5m (target 1-2/day)</option>
                  <option value="new" {"selected" if feature_set == "new" else ""}>Current feature set</option>
                  <option value="legacy" {"selected" if feature_set == "legacy" else ""}>Old legacy</option>
                </select>
                <p class="muted" id="featurePipelineHint"></p>
              </label>
              <label>Use Manual Feature Weights:
                <select name="use_manual_weights" id="manualWeightsToggle">
                  <option value="no" {"selected" if use_manual_weights_raw != "yes" else ""}>No (train logistic/linear models)</option>
                  <option value="yes" {"selected" if use_manual_weights_raw == "yes" else ""}>Yes (user-defined weights)</option>
                </select>
              </label>
              <label id="manualWeightsWrap">Feature Weights:
                <div id="manualWeightsContainer" class="manual-weights-grid"></div>
                <input type="hidden" name="manual_feature_weights" id="manualFeatureWeightsInput" value='{manual_weights_json}' />
                <p class="muted">Weights map to the selected feature set and are auto-normalized to add up to 1.0.</p>
              </label>
              <label id="dqnEpisodesWrap">DQN Episodes:
                <input type="number" min="1" step="1" name="dqn_episodes" value="{dqn_episodes_raw}" />
              </label>
              <label>Saved Model:
                <select name="selected_model">
                  <option value="__new__">Train new model</option>
                  {"".join(f'<option value="{name}" {"selected" if selected_model == name else ""}>{name}</option>' for name in saved_models)}
                </select>
              </label>
              <label>New Model Name (optional):
                <input type="text" name="model_name" value="{model_name}" placeholder="momentum_v1" />
              </label>
              <label>BUY if P(Up) &gt; (optional):
                <input type="number" min="0" max="1" step="0.01" name="buy_threshold" value="{buy_threshold_raw}" placeholder="0.60" />
              </label>
              <label>SELL if P(Up) &lt; (optional):
                <input type="number" min="0" max="1" step="0.01" name="sell_threshold" value="{sell_threshold_raw}" placeholder="0.40" />
              </label>
              <label>Stop Loss Strategy:
                <select name="stop_loss_strategy" id="stopLossStrategy">
                  <option value="none" {"selected" if stop_loss_strategy_raw == "none" else ""}>None</option>
                  <option value="atr" {"selected" if stop_loss_strategy_raw == "atr" else ""}>Volatility Buffer (ATR-Based)</option>
                  <option value="model_invalidation" {"selected" if stop_loss_strategy_raw == "model_invalidation" else ""}>Model Invalidation (MAE-Linked)</option>
                  <option value="time_decay" {"selected" if stop_loss_strategy_raw == "time_decay" else ""}>Time-Decay (Temporal Exit)</option>
                  <option value="fixed_percentage" {"selected" if stop_loss_strategy_raw == "fixed_percentage" else ""}>Fixed Percentage</option>
                  <option value="trailing_stop" {"selected" if stop_loss_strategy_raw == "trailing_stop" else ""}>Trailing Stop Loss</option>
                </select>
              </label>
              <label id="fixedStopLossWrap">Fixed Stop Loss %:
                <input type="number" min="0.01" step="any" name="fixed_stop_pct" value="{fixed_stop_pct_raw}" placeholder="2.0" />
              </label>
              <label>Take Profit %:
                <input type="number" min="0.01" step="0.01" name="take_profit_pct" value="{take_profit_pct_raw}" placeholder="1.5" />
              </label>
              <label>Max Hold Bars:
                <input type="number" min="1" step="1" name="max_hold_bars" value="{max_hold_bars_raw}" placeholder="10" />
              </label>
              <label>Monte Carlo:
                <select name="monte_carlo_method">
                  <option value="none" {"selected" if monte_carlo_method == "none" else ""}>None</option>
                  <option value="bootstrap" {"selected" if monte_carlo_method == "bootstrap" else ""}>Bootstrap resampling</option>
                  <option value="shuffle" {"selected" if monte_carlo_method == "shuffle" else ""}>Shuffle returns</option>
                  <option value="block" {"selected" if monte_carlo_method == "block" else ""}>Block bootstrap</option>
                </select>
              </label>
              <label>Monte Carlo Simulations:
                <input type="number" min="1" step="1" name="monte_carlo_n_sim" value="{monte_carlo_n_sim_raw}" />
              </label>
              <label>Monte Carlo Block Size:
                <input type="number" min="1" step="1" name="monte_carlo_block_size" value="{monte_carlo_block_size_raw}" />
              </label>
              <label>Monte Carlo Seed (optional):
                <input type="number" step="1" name="monte_carlo_seed" value="{monte_carlo_seed_raw}" placeholder="42" />
              </label>
              <label>&nbsp;
                <div class="button-row">
                  <button type="submit" name="train_action" value="train">Download + Train</button>
                  <button type="submit" name="train_action" value="evaluate" class="secondary">Evaluate Only</button>
                  <button type="submit" name="train_action" value="evaluate_historical" class="secondary">Evaluate Real History</button>
                  <button type="submit" name="train_action" value="evaluate_update" class="secondary">Evaluate + Update Preset</button>
                </div>
              </label>
              </div>
            </form>
            <div class="card">
              <h2>Run Models</h2>
              <p class="muted">Run single-model and run-all preset workflows have moved to their own page.</p>
              <a href="{run_models_href}" class="tab-link" style="display:inline-block;">Open Run Models Page</a>
            </div>
            {message_html}
            {error_html}
            {present_html}
            {result_html}
            {provider_notice_html}
            </div>
            <div id="savedEvalsModal" class="modal" aria-hidden="true">
              <div class="modal-card">
                <h3>Saved Evaluations</h3>
                <p class="muted">Click to open. Right-click any item to delete it.</p>
                <div class="saved-eval-list">
                  {saved_eval_rows_html if saved_eval_rows_html else "<p class='muted'>No saved evaluations yet.</p>"}
                </div>
                <button type="button" id="closeSavedEvalsBtn" class="secondary" style="margin-top:0.8rem;">Close</button>
              </div>
            </div>
            <div id="savedEvalContextMenu" class="context-menu">
              <button type="button" id="deleteSavedEvalBtn">Delete saved evaluation</button>
            </div>
            <form id="openSavedEvalForm" method="post" style="display:none;">
              <input type="hidden" name="mode" value="saved_eval" />
              <input type="hidden" name="data_provider" value="{data_provider}" />
              <input type="hidden" name="eval_action" value="open" />
              <input type="hidden" name="evaluation_id" id="openSavedEvalId" />
            </form>
            <form id="deleteSavedEvalForm" method="post" style="display:none;">
              <input type="hidden" name="mode" value="saved_eval" />
              <input type="hidden" name="data_provider" value="{data_provider}" />
              <input type="hidden" name="eval_action" value="delete" />
              <input type="hidden" name="evaluation_id" id="deleteSavedEvalId" />
            </form>
            <form id="saveEvalForm" method="post" style="display:none;">
              <input type="hidden" name="mode" value="saved_eval" />
              <input type="hidden" name="data_provider" value="{data_provider}" />
              <input type="hidden" name="eval_action" value="save" />
              <input type="hidden" name="evaluation_name" id="evaluationNameInput" />
              <input type="hidden" name="evaluation_payload" id="evaluationPayloadInput" />
            </form>
            <div id="loadingOverlay" class="loading-overlay" aria-live="polite" aria-busy="true">
              <div class="loading-card">
                <h3 id="loadingTitle" class="loading-title">Working...</h3>
                <div class="progress-track">
                  <div id="progressFill" class="progress-fill"></div>
                </div>
                <div class="loading-meta">
                  <span id="progressText">0%</span>
                  <span id="etaText">Estimating...</span>
                </div>
              </div>
            </div>
            <script>
              const loadingOverlay = document.getElementById("loadingOverlay");
              const loadingTitle = document.getElementById("loadingTitle");
              const progressFill = document.getElementById("progressFill");
              const progressText = document.getElementById("progressText");
              const etaText = document.getElementById("etaText");
              const saveEvaluationBtn = document.getElementById("saveEvaluationBtn");
              const mobileNavToggle = document.getElementById("mobileNavToggle");
              const primaryNavMenu = document.getElementById("primaryNavMenu");
              const openEvaluationsBtn = document.getElementById("openEvaluationsBtn");
              const savedEvalsModal = document.getElementById("savedEvalsModal");
              const closeSavedEvalsBtn = document.getElementById("closeSavedEvalsBtn");
              const openSavedEvalForm = document.getElementById("openSavedEvalForm");
              const openSavedEvalId = document.getElementById("openSavedEvalId");
              const deleteSavedEvalForm = document.getElementById("deleteSavedEvalForm");
              const deleteSavedEvalId = document.getElementById("deleteSavedEvalId");
              const saveEvalForm = document.getElementById("saveEvalForm");
              const evaluationNameInput = document.getElementById("evaluationNameInput");
              const evaluationPayloadInput = document.getElementById("evaluationPayloadInput");
              const savedEvalContextMenu = document.getElementById("savedEvalContextMenu");
              const deleteSavedEvalBtn = document.getElementById("deleteSavedEvalBtn");
              const currentEvaluationPayload = {current_eval_payload_json};
              const savedEvaluations = {saved_eval_items_json};
              let loadingTimer = null;
              let contextTargetId = null;

              if (mobileNavToggle && primaryNavMenu) {{
                mobileNavToggle.addEventListener("click", () => {{
                  const isOpen = primaryNavMenu.classList.toggle("open");
                  mobileNavToggle.setAttribute("aria-expanded", isOpen ? "true" : "false");
                }});
              }}

              function formatEta(seconds) {{
                const sec = Math.max(0, Math.ceil(seconds));
                if (sec < 60) {{
                  return `${{sec}}s remaining`;
                }}
                const mins = Math.floor(sec / 60);
                const rem = sec % 60;
                return `${{mins}}m ${{rem}}s remaining`;
              }}

              function showLoading(title, estimatedSeconds) {{
                if (loadingTimer) {{
                  window.clearInterval(loadingTimer);
                  loadingTimer = null;
                }}
                loadingTitle.textContent = title;
                loadingOverlay.style.display = "flex";
                const started = Date.now();
                const estimateMs = Math.max(1000, estimatedSeconds * 1000);

                const update = () => {{
                  const elapsedMs = Date.now() - started;
                  const rawPct = (elapsedMs / estimateMs) * 100;
                  const pct = Math.min(99, rawPct);
                  progressFill.style.width = `${{pct.toFixed(1)}}%`;
                  progressText.textContent = `${{Math.floor(pct)}}%`;
                  const etaSeconds = Math.max(0, (estimateMs - elapsedMs) / 1000);
                  etaText.textContent = pct >= 99 ? "Finalizing..." : formatEta(etaSeconds);
                }};

                update();
                loadingTimer = window.setInterval(update, 220);
              }}

              function parseTickerCount(rawTickerValue) {{
                const parts = String(rawTickerValue || "")
                  .split(/[,\\n]/)
                  .map((token) => token.trim())
                  .filter(Boolean);
                return Math.max(1, parts.length);
              }}

              function estimateTrainOrEvalSeconds(form, action) {{
                const rows = Math.max(50, Number(form.querySelector('input[name="rows"]')?.value || "250"));
                const featureSet = form.querySelector('select[name="feature_set"]')?.value || "feature2";
                const selectedModel = form.querySelector('select[name="selected_model"]')?.value || "__new__";
                const dqnEpisodes = Math.max(1, Number(form.querySelector('input[name="dqn_episodes"]')?.value || "120"));
                const tickerCount = parseTickerCount(form.querySelector('input[name="ticker"]')?.value || "");
                const isDqn = featureSet === "dqn";
                const usesSavedModel = selectedModel !== "__new__";
                const isMultiTicker = tickerCount > 1;

                const perTickerDownloadSeconds = Math.max(3, rows / 35);
                const perTickerFeatureSeconds = Math.max(2, rows / 55);
                const perTickerTrainSeconds = isDqn
                  ? Math.max(22, (rows / 12) + (dqnEpisodes * 3.8))
                  : Math.max(5, rows / 18);
                const perTickerEvalSeconds = Math.max(3, rows / 28);
                const saveOverheadSeconds = action === "train" ? 3 : 0;

                const perTickerTotal = usesSavedModel && !isMultiTicker && action === "evaluate"
                  ? perTickerDownloadSeconds + perTickerFeatureSeconds + perTickerEvalSeconds
                  : perTickerDownloadSeconds + perTickerFeatureSeconds + perTickerTrainSeconds + perTickerEvalSeconds + saveOverheadSeconds;

                const seconds = 2 + (tickerCount * perTickerTotal);
                return Math.round(Math.min(7200, Math.max(10, seconds)));
              }}

              document.querySelectorAll("form").forEach((form) => {{
                form.addEventListener("submit", (evt) => {{
                  const mode = form.querySelector('input[name="mode"]')?.value || "";
                  const submitter = evt.submitter;
                  const action = submitter?.value || "";

                  if (mode === "train" && (action === "train" || action === "evaluate" || action === "evaluate_historical" || action === "evaluate_update")) {{
                    const manualMode = (form.querySelector('select[name="use_manual_weights"]')?.value || "no") === "yes";
                    const seconds = estimateTrainOrEvalSeconds(form, action);
                    const title = action === "train"
                      ? (manualMode ? "Downloading data, saving manual-weight model, and evaluating..." : "Downloading data and training model...")
                      : action === "evaluate_historical"
                        ? "Downloading data and evaluating on real historical split..."
                        : action === "evaluate_update"
                          ? "Evaluating model and updating preset + Monte Carlo distributions..."
                          : "Downloading data and evaluating model...";
                    showLoading(title, seconds);
                  }} else if (mode === "present_all") {{
                    const modelCount = Math.max(1, document.querySelectorAll('table tr').length - 1);
                    const seconds = Math.min(120, Math.max(10, modelCount * 8));
                    showLoading("Running all present-mode models...", seconds);
                  }}
                }});
              }});

              const featureSetEl = document.querySelector('select[name="feature_set"]');
              const manualWeightsToggleEl = document.getElementById("manualWeightsToggle");
              const manualWeightsWrapEl = document.getElementById("manualWeightsWrap");
              const manualWeightsContainerEl = document.getElementById("manualWeightsContainer");
              const manualFeatureWeightsInputEl = document.getElementById("manualFeatureWeightsInput");
              const dqnEpisodesWrapEl = document.getElementById("dqnEpisodesWrap");
              const featurePipelineHintEl = document.getElementById("featurePipelineHint");
              const stopLossStrategyEl = document.getElementById("stopLossStrategy");
              const fixedStopLossWrapEl = document.getElementById("fixedStopLossWrap");
              const runAllTable = document.getElementById("runAllTable");
              const sortDirections = {{}};
              const featureNameMap = {json.dumps(feature_name_map)};
              const featurePipelineDescriptions = {{
                vwap_intraday_5m_session: "5m session reset VWAP with EMA 3/9/21 context, VWAP delta-to-mean, ±1/±2 standard-deviation envelopes, price-to-band distances, and envelope ranges.",
                close_hold_reversion: "Built for market-close entries held 1-2 bars: overshoot/exhaustion cues using RSI/Stoch zones, Bollinger dislocation, and VWAP distance normalized by ATR.",
                close_hold_momentum: "Built for market-close entries held 1-2 bars: continuation cues from EMA slope/spread stack, MACD acceleration, and VWAP breakout-vs-breakdown pressure.",
                open15_orb_intraday: "Built for 5m day trading after observing first 15m: opening-range breakout/breakdown confirmation, session VWAP alignment, and explicit intraday time-window gating.",
                open15_vwap_reclaim_intraday: "Built for 5m day trading after first 15m: opening-range location plus VWAP reclaim/reversion triggers with late-session risk flags.",
                open15_trend_momentum_daytrade: "Built for 5m day trading after first 15m: ORB confirmation + EMA/MACD trend acceleration with intraday-only gating and max-2-trade/day control features.",
                open15_dual_breakout_daytrade: "Built for 5m day trading after first 15m: opening-range continuation plus VWAP/EMA follow-through filters with explicit same-day exit bias controls.",
                open15_dual_breakout_daytrade_plus: "Built for 5m day trading after first 15m: improved dual-breakout follow-through with VWAP reclaim quality filters and controlled scaling to ~0-3 trades/day.",
                open15_dual_breakout_daytrade_scalp: "Built for 5m day trading after first 15m: faster dual-breakout/reclaim entries with tighter momentum checks and higher turnover targeting ~1-5 trades/day.",
                vwap_momentum_trend_5m_conservative: "Built for 5m trend continuation around VWAP: EMA slope/stack + MACD agreement, VWAP breakout strength, and daily trade cap features targeting ~0-4 trades/day.",
                vwap_momentum_trend_5m_pullback: "Built for 5m trend pullbacks to VWAP: reclaim/reversion context + momentum re-acceleration, with strict intraday window and lower trade-frequency cap features.",
                vwap_volume_long_momentum_5m: "Built for 5m long momentum day trades: VWAP breakout and EMA/MACD trend alignment gated by volume expansion, with explicit avoid-overnight controls.",
                vwap_volume_regime_adaptive_5m: "Built for 5m adaptive trading: VWAP displacement + volume classify trend-vs-reversion day type, then apply regime-aware momentum/reversion context with intraday-only bias.",
                vwap_volume_first5_trend_momentum_5m: "Built for 5m day trading using the first 5 bars: VWAP + volume-confirmed trend momentum with max-2/day controls and explicit same-session exit bias.",
                vwap_volume_profile_first5_trend_momentum_5m: "Built for 5m day trading using first-5-bar context plus session VWAP volume-profile bands (acceptance/expansion) for max-2/day intraday-only momentum decisions.",
              }};

              function parseSortValue(rawValue, sortType) {{
                const textValue = (rawValue || "").trim();
                if (sortType === "number") {{
                  const parsed = Number.parseFloat(textValue.replace(/,/g, ""));
                  return Number.isFinite(parsed) ? parsed : null;
                }}
                if (sortType === "percent") {{
                  const parsed = Number.parseFloat(textValue.replace("%", "").replace(/,/g, ""));
                  return Number.isFinite(parsed) ? parsed : null;
                }}
                return textValue.toLowerCase();
              }}

              function sortRunAllTable(columnIndex, direction, sortType) {{
                if (!runAllTable) return;
                const rows = Array.from(runAllTable.querySelectorAll("tr")).slice(1);
                if (!rows.length) return;

                rows.sort((leftRow, rightRow) => {{
                  const leftCell = leftRow.children[columnIndex];
                  const rightCell = rightRow.children[columnIndex];
                  const leftValue = parseSortValue(leftCell?.textContent || "", sortType);
                  const rightValue = parseSortValue(rightCell?.textContent || "", sortType);

                  if (sortType === "number" || sortType === "percent") {{
                    if (leftValue === null && rightValue === null) return 0;
                    if (leftValue === null) return 1;
                    if (rightValue === null) return -1;
                    return direction === "desc" ? rightValue - leftValue : leftValue - rightValue;
                  }}

                  if (leftValue === rightValue) return 0;
                  if (direction === "desc") {{
                    return String(rightValue).localeCompare(String(leftValue));
                  }}
                  return String(leftValue).localeCompare(String(rightValue));
                }});

                rows.forEach((row) => runAllTable.appendChild(row));
              }}

              if (runAllTable) {{
                const headerCells = Array.from(runAllTable.querySelectorAll("th"));
                headerCells.forEach((headerCell, index) => {{
                  headerCell.style.cursor = "pointer";
                  headerCell.title = "Click to sort";
                  headerCell.addEventListener("click", () => {{
                    const currentDirection = sortDirections[index] || "asc";
                    const nextDirection = currentDirection === "asc" ? "desc" : "asc";
                    sortDirections[index] = nextDirection;
                    sortRunAllTable(index, nextDirection, headerCell.dataset.sortType || "text");
                  }});
                }});
              }}

              function toggleFixedStopField() {{
                if (!stopLossStrategyEl || !fixedStopLossWrapEl) return;
                const fixedStopInput = fixedStopLossWrapEl.querySelector('input[name="fixed_stop_pct"]');
                const isFixed = stopLossStrategyEl.value === "fixed_percentage" || stopLossStrategyEl.value === "trailing_stop";
                fixedStopLossWrapEl.style.display = isFixed ? "block" : "none";
                if (fixedStopInput) {{
                  fixedStopInput.disabled = !isFixed;
                  if (!isFixed) {{
                    fixedStopInput.setCustomValidity("");
                  }}
                }}
              }}
              if (stopLossStrategyEl) {{
                stopLossStrategyEl.addEventListener("change", toggleFixedStopField);
                toggleFixedStopField();
              }}

              function toggleDqnEpisodesField() {{
                if (!featureSetEl || !dqnEpisodesWrapEl) return;
                const episodesInput = dqnEpisodesWrapEl.querySelector('input[name="dqn_episodes"]');
                const isDqn = featureSetEl.value === "dqn";
                dqnEpisodesWrapEl.style.display = isDqn ? "block" : "none";
                if (episodesInput) {{
                  episodesInput.disabled = !isDqn;
                  if (!isDqn) {{
                    episodesInput.setCustomValidity("");
                  }}
                }}
              }}
              if (featureSetEl) {{
                featureSetEl.addEventListener("change", toggleDqnEpisodesField);
                toggleDqnEpisodesField();
              }}

              function loadManualWeightsFromState(featureNames) {{
                if (!manualFeatureWeightsInputEl) return null;
                try {{
                  const parsed = JSON.parse(manualFeatureWeightsInputEl.value || "[]");
                  if (!Array.isArray(parsed) || parsed.length !== featureNames.length) return null;
                  return parsed.map((value) => Number(value));
                }} catch (_err) {{
                  return null;
                }}
              }}

              function renderManualWeightInputs() {{
                if (!featureSetEl || !manualWeightsContainerEl) return;
                const featureNames = featureNameMap[featureSetEl.value] || [];
                const existingWeights = loadManualWeightsFromState(featureNames);
                const defaultWeight = featureNames.length ? (1 / featureNames.length) : 0;
                manualWeightsContainerEl.innerHTML = featureNames.map((name, idx) => {{
                  const value = existingWeights ? existingWeights[idx] : defaultWeight;
                  return `
                    <div class="manual-weight-item">
                      <code>${{name}}</code>
                      <input
                        type="number"
                        step="0.0001"
                        value="${{Number.isFinite(value) ? value.toFixed(4) : "0.0000"}}"
                        data-weight-index="${{idx}}"
                      />
                    </div>
                  `;
                }}).join("");
              }}

              function updateFeaturePipelineHint() {{
                if (!featureSetEl || !featurePipelineHintEl) return;
                featurePipelineHintEl.textContent = featurePipelineDescriptions[featureSetEl.value] || "";
              }}

              function syncManualWeightsJson() {{
                if (!manualFeatureWeightsInputEl || !manualWeightsContainerEl) return;
                const values = Array.from(manualWeightsContainerEl.querySelectorAll("input[data-weight-index]"))
                  .map((inputEl) => Number(inputEl.value || "0"));
                manualFeatureWeightsInputEl.value = JSON.stringify(values);
              }}

              function toggleManualWeightsField() {{
                if (!manualWeightsToggleEl || !manualWeightsWrapEl || !featureSetEl) return;
                const manualInputs = manualWeightsWrapEl.querySelectorAll("input[data-weight-index]");
                const isManual = manualWeightsToggleEl.value === "yes";
                const isDqn = featureSetEl.value === "dqn";
                const shouldShow = isManual && !isDqn;
                manualWeightsWrapEl.style.display = shouldShow ? "block" : "none";
                manualInputs.forEach((inputEl) => {{
                  inputEl.disabled = !shouldShow;
                  if (!shouldShow) {{
                    inputEl.setCustomValidity("");
                  }}
                }});
                if (isManual && isDqn) {{
                  manualWeightsToggleEl.setCustomValidity("Manual feature weights are unavailable for DQN.");
                }} else {{
                  manualWeightsToggleEl.setCustomValidity("");
                }}
                syncManualWeightsJson();
              }}

              if (manualWeightsContainerEl) {{
                manualWeightsContainerEl.addEventListener("input", syncManualWeightsJson);
              }}
              if (featureSetEl) {{
                featureSetEl.addEventListener("change", () => {{
                  renderManualWeightInputs();
                  toggleManualWeightsField();
                  updateFeaturePipelineHint();
                }});
              }}
              if (manualWeightsToggleEl) {{
                manualWeightsToggleEl.addEventListener("change", toggleManualWeightsField);
              }}
              renderManualWeightInputs();
              toggleManualWeightsField();
              updateFeaturePipelineHint();
              const presentStopLossStrategyEl = document.getElementById("presentStopLossStrategy");
              const presentFixedStopLossWrapEl = document.getElementById("presentFixedStopLossWrap");
              function togglePresentFixedStopField() {{
                if (!presentStopLossStrategyEl || !presentFixedStopLossWrapEl) return;
                const fixedStopInput = presentFixedStopLossWrapEl.querySelector('input[name="present_fixed_stop_pct"]');
                const isFixed = presentStopLossStrategyEl.value === "fixed_percentage" || presentStopLossStrategyEl.value === "trailing_stop";
                presentFixedStopLossWrapEl.style.display = isFixed ? "block" : "none";
                if (fixedStopInput) {{
                  fixedStopInput.disabled = !isFixed;
                  if (!isFixed) {{
                    fixedStopInput.setCustomValidity("");
                  }}
                }}
              }}
              if (presentStopLossStrategyEl) {{
                presentStopLossStrategyEl.addEventListener("change", togglePresentFixedStopField);
                togglePresentFixedStopField();
              }}

              function closeSavedEvaluationsModal() {{
                if (!savedEvalsModal) return;
                savedEvalsModal.style.display = "none";
                savedEvalsModal.setAttribute("aria-hidden", "true");
              }}

              if (openEvaluationsBtn && savedEvalsModal) {{
                openEvaluationsBtn.addEventListener("click", () => {{
                  savedEvalsModal.style.display = "flex";
                  savedEvalsModal.setAttribute("aria-hidden", "false");
                }});
              }}
              if (closeSavedEvalsBtn) {{
                closeSavedEvalsBtn.addEventListener("click", closeSavedEvaluationsModal);
              }}
              if (savedEvalsModal) {{
                savedEvalsModal.addEventListener("click", (evt) => {{
                  if (evt.target === savedEvalsModal) {{
                    closeSavedEvaluationsModal();
                  }}
                }});
              }}

              document.querySelectorAll(".saved-eval-item").forEach((btn) => {{
                btn.addEventListener("click", () => {{
                  if (!openSavedEvalForm || !openSavedEvalId) return;
                  openSavedEvalId.value = btn.dataset.id || "";
                  openSavedEvalForm.submit();
                }});
                btn.addEventListener("contextmenu", (evt) => {{
                  evt.preventDefault();
                  const id = btn.dataset.id || "";
                  if (!id || !savedEvalContextMenu) return;
                  contextTargetId = id;
                  savedEvalContextMenu.style.display = "block";
                  savedEvalContextMenu.style.left = `${{evt.clientX}}px`;
                  savedEvalContextMenu.style.top = `${{evt.clientY}}px`;
                }});
              }});

              if (deleteSavedEvalBtn) {{
                deleteSavedEvalBtn.addEventListener("click", () => {{
                  if (!contextTargetId || !deleteSavedEvalId || !deleteSavedEvalForm) return;
                  const target = savedEvaluations.find((item) => String(item.id) === String(contextTargetId));
                  const targetName = target?.name || "this evaluation";
                  const shouldDelete = window.confirm(`Delete "${{targetName}}"?`);
                  if (!shouldDelete) return;
                  deleteSavedEvalId.value = contextTargetId;
                  deleteSavedEvalForm.submit();
                }});
              }}

              document.addEventListener("click", () => {{
                if (savedEvalContextMenu) {{
                  savedEvalContextMenu.style.display = "none";
                }}
              }});

              if (saveEvaluationBtn) {{
                saveEvaluationBtn.addEventListener("click", () => {{
                  if (!currentEvaluationPayload) {{
                    window.alert("Run a single-ticker evaluation first, then save it.");
                    return;
                  }}
                  const defaultName = currentEvaluationPayload.form_state?.ticker
                    ? `${{currentEvaluationPayload.form_state.ticker}}_${{currentEvaluationPayload.form_state.interval}}_${{new Date().toISOString().slice(0, 16)}}`
                    : `evaluation_${{new Date().toISOString().slice(0, 16)}}`;
                  const name = window.prompt("Name this saved evaluation:", defaultName);
                  if (!name) return;
                  if (!evaluationNameInput || !evaluationPayloadInput || !saveEvalForm) return;
                  evaluationNameInput.value = name.trim();
                  evaluationPayloadInput.value = JSON.stringify(currentEvaluationPayload);
                  saveEvalForm.submit();
                }});
              }}
            </script>
          </body>
        </html>
        """

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quant strategy model with custom features.")
    parser.add_argument(
        "--csv",
        type=str,
        default="",
        help="Optional CSV path. Supports both new modular features (stoch/macd/returns/trend/vol/FVG/interactions + return_next) and legacy feature columns.",
    )
    parser.add_argument(
        "--feature-set",
        choices=[
            "feature2",
            "hybrid_sharpe_core",
            "hybrid_sharpe_core_no_stack",
            "hybrid_sharpe_momentum",
            "hybrid_sharpe_selective",
            "hybrid_sharpe_regime",
            "hybrid_sharpe_volume_flow",
            "hybrid_sharpe_volume_regime",
            "close_hold_reversion",
            "close_hold_momentum",
            "war_shock_reversion",
            "war_shock_momentum",
            "dqn",
            "fvg2",
            "fvg3",
            "rsi_thresholds",
            "stoch_rsi_thresholds",
            "derivative",
            "derivative2",
            "ema",
            "bollinger_bands",
            "vwap_anchor",
            "vwap_intraday_reversion",
            "vwap_intraday_momentum",
            "vwap_intraday_5m_session",
            "vwap_breakout_reversion_regime",
            "open15_orb_intraday",
            "open15_vwap_reclaim_intraday",
            "open15_trend_momentum_daytrade",
            "open15_dual_breakout_daytrade",
            "open15_dual_breakout_daytrade_plus",
            "open15_dual_breakout_daytrade_scalp",
            "vwap_momentum_trend_5m_conservative",
            "vwap_momentum_trend_5m_pullback",
            "vwap_volume_long_momentum_5m",
            "vwap_volume_regime_adaptive_5m",
            "vwap_volume_first5_trend_momentum_5m",
            "vwap_volume_profile_first5_trend_momentum_5m",
            "new",
            "legacy",
        ],
        default="feature2",
        help="Feature pipeline to use for CLI training/evaluation. Default: feature2.",
    )
    parser.add_argument("--ui", action="store_true", help="Run Flask UI.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Flask host when using --ui.")
    parser.add_argument("--port", type=int, default=5000, help="Flask port when using --ui.")
    parser.add_argument("--ngrok", action="store_true", help="Expose Flask UI through ngrok when using --ui.")
    parser.add_argument(
        "--ngrok-authtoken",
        type=str,
        default="",
        help="Optional ngrok auth token. Prefer NGROK_AUTHTOKEN env var to avoid shell history leaks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ui:
        app = create_app()
        ngrok_process: subprocess.Popen[bytes] | None = None
        try:
            if args.ngrok:
                authtoken = args.ngrok_authtoken or os.getenv("NGROK_AUTHTOKEN", "")
                ngrok_process, public_url = start_ngrok_tunnel(host=args.host, port=args.port, authtoken=authtoken)
                print(f"ngrok tunnel active: {public_url}")
            app.run(host=args.host, port=args.port, debug=False)
        finally:
            if ngrok_process and ngrok_process.poll() is None:
                ngrok_process.terminate()
        return

    if args.csv:
        rows = load_csv(args.csv)
        print(f"Loaded {len(rows)} rows from {args.csv}")
    else:
        rows = synthetic_data()
        print(f"No CSV provided. Using synthetic dataset ({len(rows)} rows).")

    run_model(rows, feature_set=args.feature_set)


if __name__ == "__main__":
    main()
