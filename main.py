#!/usr/bin/env python3
"""
Quant probability model for a momentum/reversal strategy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from datetime import datetime
from html import escape
from typing import TYPE_CHECKING, Dict
from zoneinfo import ZoneInfo

from quant.constants import OPTIONS_MODE, SPOT_MODE
from quant.data import fetch_market_rows, load_csv, synthetic_data
from quant.discord_notify import send_discord_webhook
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
from quant.stop_loss import MODEL_MAE_DEFAULT, StopLossConfig, StopLossStrategy, parse_stop_loss_strategy, stop_loss_price, validate_fixed_stop_pct
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
from quant.live_trading import (
    QuestradeAuthClient,
    get_accounts,
    get_balances,
    get_order_history,
    get_positions,
    get_quote,
    place_order,
)

if TYPE_CHECKING:
    from flask import Flask


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


def build_default_model_name(*, ticker: str, interval: str, row_count: int, feature_set: str, prediction_horizon: int) -> str:
    return sanitize_model_name(f"{ticker}_{interval}_{row_count}_{prediction_horizon}")


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


def evaluate_run_all_models(saved_models, model_configs, *, mode: str, long_only: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data_provider = str(model_configs.get("__ui_data_provider__", "yfinance")).strip().lower()
    twelve_api_key = str(model_configs.get("__ui_twelve_api_key__", "")).strip()
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
                    "provider_notice": "",
                    "error": str(exc),
                }
            )
    return rows


def build_run_all_rows(saved_models, model_configs, *, mode: str, long_only: bool) -> str:
    run_all_rows = ""
    interval_order = {"5m": 0, "15m": 1, "1h": 2, "1d": 3}
    interval_labels = {"5m": "5 min presets", "15m": "15 min presets", "1h": "1h presets", "1d": "1d presets"}
    results = evaluate_run_all_models(saved_models, model_configs, mode=mode, long_only=long_only)
    sorted_results = sorted(
        results,
        key=lambda item: (
            interval_order.get(str(item.get("interval", "1d")), 99),
            str(item.get("ticker", "")),
            str(item.get("model_name", "")),
        ),
    )

    current_interval = None
    for item in sorted_results:
        item_interval = str(item.get("interval", "1d"))
        if item_interval != current_interval:
            current_interval = item_interval
            run_all_rows += (
                "<tr class='run-all-group-row'>"
                f"<td colspan='10'><strong>{interval_labels.get(item_interval, f'{item_interval} presets')}</strong></td>"
                "</tr>"
            )
        if item["error"]:
            run_all_rows += (
                "<tr>"
                f"<td>{item['model_name']}</td>"
                f"<td>{item['ticker']}</td>"
                f"<td>{item['interval']}</td>"
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
        run_all_rows += (
            "<tr>"
            f"<td>{item['model_name']}</td>"
            f"<td>{item['ticker']}</td>"
            f"<td>{item['interval']}</td>"
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

    def start(self, *, mode: str, long_only: bool, data_provider: str, twelve_api_key: str, webhook_url: str) -> None:
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
            worker = threading.Thread(
                target=self._loop,
                kwargs={
                    "mode": mode,
                    "long_only": long_only,
                    "data_provider": data_provider,
                    "twelve_api_key": twelve_api_key,
                    "webhook_url": webhook_url,
                },
                daemon=True,
            )
            mode_state["thread"] = worker
            worker.start()

    def stop(self, mode: str) -> None:
        with self._lock:
            self._state[mode]["running"] = False
            self._state[mode]["worker_state"] = "stopped"

    def _loop(self, *, mode: str, long_only: bool, data_provider: str, twelve_api_key: str, webhook_url: str) -> None:
        try:
            while True:
                with self._lock:
                    if not bool(self._state[mode]["running"]):
                        self._state[mode]["thread"] = None
                        return
                market_open = is_us_market_open()
                with self._lock:
                    self._state[mode]["last_market_state"] = "open" if market_open else "closed"
                if market_open:
                    try:
                        configs = load_model_configs(mode)
                        configs["__ui_data_provider__"] = data_provider
                        configs["__ui_twelve_api_key__"] = twelve_api_key
                        models = list_saved_models(mode)
                        rows = evaluate_run_all_models(models, configs, mode=mode, long_only=long_only)
                        self._notify_action_changes(mode=mode, rows=rows, webhook_url=webhook_url)
                        with self._lock:
                            self._state[mode]["last_error"] = ""
                    except Exception as exc:
                        with self._lock:
                            self._state[mode]["last_error"] = str(exc)
                    finally:
                        with self._lock:
                            self._state[mode]["last_tick"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
                else:
                    with self._lock:
                        self._state[mode]["last_tick"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
                time.sleep(600)
        except Exception as exc:
            with self._lock:
                self._state[mode]["running"] = False
                self._state[mode]["thread"] = None
                self._state[mode]["worker_state"] = "crashed"
                self._state[mode]["last_error"] = str(exc)
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
                f"{prev} ➜ {action} | P(Up): {float(row['p_up']):.2%} | Exp Ret: {float(row['expected_return']):+.4%}"
            )
            send_discord_webhook(webhook_url, content)
        with self._lock:
            self._state[mode]["last_actions"] = new_known


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


def create_app() -> "Flask":
    from flask import Flask, redirect, request, url_for

    app = Flask(__name__)

    @app.route("/manage-models", methods=["GET", "POST"])
    @app.route("/spot/manage-models", methods=["GET", "POST"])
    def manage_models() -> str:
        is_spot = request.path.startswith("/spot")
        mode_key = SPOT_MODE if is_spot else OPTIONS_MODE
        home_href = "/spot" if is_spot else "/"
        manage_href = "/spot/manage-models" if is_spot else "/manage-models"
        live_href = "/spot/live-trading" if is_spot else "/live-trading"
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
                    fixed_stop_pct = 2.0
                    if stop_loss_strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
                        fixed_stop_pct = validate_fixed_stop_pct(float(fixed_stop_raw or "2.0"))
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
                f"data-fixed-stop='{float(cfg.get('fixed_stop_pct', 2.0)):.2f}'>"
                f"<td>{model_name}</td>"
                f"<td>{cfg.get('ticker')}</td>"
                f"<td>{cfg.get('interval')}</td>"
                f"<td>{int(cfg.get('rows', 250))}</td>"
                f"<td>{float(cfg.get('buy_threshold', 0.6)):.2f}</td>"
                f"<td>{float(cfg.get('sell_threshold', 0.4)):.2f}</td>"
                f"<td>{cfg.get('stop_loss_strategy', StopLossStrategy.NONE.value)}</td>"
                f"<td>{float(cfg.get('fixed_stop_pct', 2.0)):.2f}%</td>"
                f"<td>{include_badge}</td>"
                "</tr>"
            )

        return f"""
        <html>
          <head><title>Manage Models</title></head>
          <body>
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
            </style>
            <nav class="topbar">
              <div class="topbar-inner">
                <a href="{home_href}" class="brand">{brand_label}</a>
                <a href="{home_href}" class="tab-link">Model</a>
                <a href="{manage_href}" class="tab-link active">Manage Models</a>
                <a href="{live_href}" class="tab-link">Live Trading Bot</a>
                <a href="{home_href}#present-mode" class="tab-link">Present Mode</a>
                <a href="{mode_switch_href}" class="tab-link">{mode_switch_label}</a>
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
              let menuModelName = "";

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

              function openAllSettings() {{
                document.getElementById("allTicker").value = "";
                document.getElementById("allInterval").value = "";
                document.getElementById("allRows").value = "";
                document.getElementById("allBuyThreshold").value = "";
                document.getElementById("allSellThreshold").value = "";
                document.getElementById("allStopLossStrategy").value = "";
                document.getElementById("allFixedStopPct").value = "";
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

    @app.route("/live-trading", methods=["GET", "POST"])
    @app.route("/spot/live-trading", methods=["GET", "POST"])
    def live_trading() -> str:
        is_spot = request.path.startswith("/spot")
        mode_key = SPOT_MODE if is_spot else OPTIONS_MODE
        home_href = "/spot" if is_spot else "/"
        manage_href = "/spot/manage-models" if is_spot else "/manage-models"
        live_href = "/spot/live-trading" if is_spot else "/live-trading"
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

        message_html = ""
        error_html = ""
        details_html = ""
        selected_model = request.form.get("model_name", "").strip()
        symbol = request.form.get("symbol", "AAPL").strip().upper()
        quantity_raw = request.form.get("quantity", "1").strip()
        account_id = request.form.get("account_id", "").strip()
        order_type = request.form.get("order_type", "Market").strip().title()
        action = request.form.get("trade_action", "Buy").strip().title()
        limit_price_raw = request.form.get("limit_price", "").strip()
        interval = request.form.get("candle_interval", "OneDay").strip()
        is_paper = request.form.get("paper_mode", "1") == "1"

        saved_models = list_saved_models(mode_key)
        model_configs = load_model_configs(mode_key)

        if request.method == "POST":
            op = request.form.get("op", "").strip()
            try:
                auth_client = QuestradeAuthClient()
                if op == "login":
                    auth_client.refresh_access_token()
                    message_html = "<p style='color:#7bd88f;'>Authenticated with Questrade and refreshed OAuth2 token.</p>"
                elif op == "quote":
                    quote = get_quote(symbol, auth_client=auth_client)
                    details_html = f"<pre>{escape(json.dumps(quote, indent=2))}</pre>"
                    message_html = f"<p style='color:#7bd88f;'>Fetched quote for {escape(symbol)}.</p>"
                elif op == "accounts":
                    accounts = get_accounts(auth_client=auth_client)
                    details_html = f"<pre>{escape(json.dumps(accounts, indent=2))}</pre>"
                    message_html = "<p style='color:#7bd88f;'>Fetched account list.</p>"
                elif op == "balances":
                    balances = get_balances(account_id, auth_client=auth_client)
                    positions = get_positions(account_id, auth_client=auth_client)
                    details_html = "<h3>Balances</h3>" + f"<pre>{escape(json.dumps(balances, indent=2))}</pre>" + "<h3>Positions</h3>" + f"<pre>{escape(json.dumps(positions, indent=2))}</pre>"
                    message_html = f"<p style='color:#7bd88f;'>Fetched balances and positions for account {escape(account_id)}.</p>"
                elif op == "history":
                    orders = get_order_history(account_id, auth_client=auth_client)
                    details_html = f"<pre>{escape(json.dumps(orders, indent=2))}</pre>"
                    message_html = f"<p style='color:#7bd88f;'>Fetched order history for account {escape(account_id)}.</p>"
                elif op == "trade":
                    order: dict[str, object] = {
                        "accountId": account_id,
                        "symbol": symbol,
                        "quantity": int(quantity_raw or "0"),
                        "action": action,
                        "orderType": order_type,
                        "isPaper": is_paper,
                    }
                    if order_type == "Limit":
                        order["limitPrice"] = float(limit_price_raw or "0")
                    result = place_order(order, auth_client=auth_client)
                    details_html = f"<pre>{escape(json.dumps(result, indent=2))}</pre>"
                    message_html = "<p style='color:#7bd88f;'>Order request processed.</p>"
                elif op == "model_trade":
                    if selected_model not in saved_models:
                        raise ValueError("Select a valid saved model first.")
                    cfg = get_model_config(selected_model, model_configs)
                    bundle = load_model_bundle(mode_key, selected_model)
                    rows_payload, _ = fetch_market_rows(
                        ticker=str(cfg.get("ticker", "AAPL")),
                        interval=str(cfg.get("interval", "1d")),
                        row_count=int(cfg.get("rows", 250)),
                        provider="yfinance",
                        twelve_api_key="",
                        prediction_horizon=int(cfg.get("prediction_horizon", 5)),
                    )
                    prediction = predict_signal(
                        bundle,
                        rows_payload[-1],
                        buy_threshold=float(cfg.get("buy_threshold", 0.6)),
                        sell_threshold=float(cfg.get("sell_threshold", 0.4)),
                        long_only=is_spot,
                    )
                    mapped_action = str(prediction["action"]).upper()
                    if mapped_action not in {"BUY", "SELL"}:
                        details_html = f"<pre>{escape(json.dumps(prediction, indent=2))}</pre>"
                        message_html = f"<p style='color:#7bd88f;'>Model signal is {escape(mapped_action)}. No order was placed.</p>"
                    else:
                        order_request: dict[str, object] = {
                            "accountId": account_id,
                            "symbol": str(cfg.get('ticker', symbol)),
                            "quantity": int(quantity_raw or "0"),
                            "action": "Buy" if mapped_action == "BUY" else "Sell",
                            "orderType": order_type,
                            "isPaper": is_paper,
                        }
                        if order_type == "Limit":
                            order_request["limitPrice"] = float(limit_price_raw or "0")
                        trade_response = place_order(order_request, auth_client=auth_client)
                        details_html = "<h3>Model Prediction</h3>" + f"<pre>{escape(json.dumps(prediction, indent=2))}</pre>" + "<h3>Trade Response</h3>" + f"<pre>{escape(json.dumps(trade_response, indent=2))}</pre>"
                        message_html = f"<p style='color:#7bd88f;'>Executed {escape(mapped_action)} from model rule for {escape(selected_model)}.</p>"
            except Exception as exc:
                error_html = f"<p style='color:#ff7b7b;'><strong>Error:</strong> {escape(str(exc))}</p>"

        return f"""
        <html><head><title>Live Trading Bot</title></head><body>
        <style>
        :root {{--bg:{theme_bg};--panel:{theme_panel};--panel2:{theme_panel2};--border:{theme_border};--text:{theme_text};--muted:{theme_muted};--accent:{theme_accent};}}
        *{{box-sizing:border-box;}} body{{margin:0;background:var(--bg);color:var(--text);font-family:Inter,Segoe UI,Arial,sans-serif;}}
        .topbar{{position:sticky;top:0;z-index:50;background:{theme_topbar_bg};border-bottom:1px solid var(--border);}} .topbar-inner{{max-width:1100px;margin:0 auto;padding:0.9rem 2rem;display:flex;align-items:center;gap:1rem;}}
        .brand{{font-weight:700;color:{theme_brand};text-decoration:none;margin-right:auto;}} .tab-link{{color:{theme_tab};text-decoration:none;padding:.4rem .65rem;border-radius:8px;border:1px solid transparent;}}
        .tab-link:hover,.tab-link.active{{color:{theme_tab_active};border-color:var(--border);background:{theme_tab_hover_bg};}} .container{{max-width:1100px;margin:0 auto;padding:2rem;}}
        .card{{background:linear-gradient(180deg,var(--panel) 0%,var(--panel2) 100%);border:1px solid var(--border);border-radius:14px;padding:1rem 1.1rem;margin-bottom:1rem;}}
        .grid{{display:grid;grid-template-columns:repeat(2,minmax(240px,1fr));gap:.8rem;}} label{{display:block;color:var(--muted);font-size:.92rem;}}
        input,select{{width:100%;margin-top:.35rem;background:{theme_surface};color:var(--text);border:1px solid var(--border);border-radius:10px;padding:.58rem .65rem;}}
        .btn-row{{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:1rem;}} button{{border:none;border-radius:10px;padding:.62rem .8rem;cursor:pointer;background:{theme_accent};color:#111;font-weight:700;}}
        .secondary{{background:{theme_secondary_bg};color:{theme_secondary_text};border:1px solid var(--border);}}
        pre{{white-space:pre-wrap;background:{theme_surface};padding:.8rem;border-radius:10px;border:1px solid var(--border);}}
        </style>
        <nav class="topbar"><div class="topbar-inner">
            <a href="{home_href}" class="brand">{brand_label}</a>
            <a href="{home_href}" class="tab-link">Model</a>
            <a href="{manage_href}" class="tab-link">Manage Models</a>
            <a href="{live_href}" class="tab-link active">Live Trading Bot</a>
            <a href="{mode_switch_href}" class="tab-link">{mode_switch_label}</a>
        </div></nav>
        <div class="container">
          <h1>Live Trading Bot Mode</h1>
          <p style="color:var(--muted);">Safety guardrails: paper mode is on by default. Real orders require <code>ENABLE_LIVE_TRADING=true</code>.</p>
          {message_html}
          {error_html}
          <form method="post" class="card">
            <div class="grid">
              <label>Saved Model
                <select name="model_name"><option value="">-- Select model --</option>{''.join(f'<option value="{name}" {"selected" if selected_model == name else ""}>{name}</option>' for name in saved_models)}</select>
              </label>
              <label>Account ID<input name="account_id" value="{escape(account_id)}" /></label>
              <label>Symbol<input name="symbol" value="{escape(symbol)}" /></label>
              <label>Quantity<input type="number" min="1" name="quantity" value="{escape(quantity_raw)}" /></label>
              <label>Order Type<select name="order_type"><option value="Market" {"selected" if order_type == "Market" else ""}>Market</option><option value="Limit" {"selected" if order_type == "Limit" else ""}>Limit</option></select></label>
              <label>Limit Price<input type="number" step="0.01" min="0" name="limit_price" value="{escape(limit_price_raw)}" /></label>
              <label>Action<select name="trade_action"><option value="Buy" {"selected" if action == "Buy" else ""}>Buy</option><option value="Sell" {"selected" if action == "Sell" else ""}>Sell</option></select></label>
              <label>Candle Interval<select name="candle_interval"><option value="OneDay" {"selected" if interval == "OneDay" else ""}>OneDay</option><option value="OneHour" {"selected" if interval == "OneHour" else ""}>OneHour</option><option value="FiveMinutes" {"selected" if interval == "FiveMinutes" else ""}>FiveMinutes</option></select></label>
              <label>Paper Trading Mode<select name="paper_mode"><option value="1" {"selected" if is_paper else ""}>Enabled (no live orders)</option><option value="0" {"selected" if not is_paper else ""}>Disabled</option></select></label>
            </div>
            <div class="btn-row">
              <button type="submit" name="op" value="login" class="secondary">OAuth Login/Refresh</button>
              <button type="submit" name="op" value="quote" class="secondary">Get Quote</button>
              <button type="submit" name="op" value="accounts" class="secondary">Get Accounts</button>
              <button type="submit" name="op" value="balances" class="secondary">Get Balances/Positions</button>
              <button type="submit" name="op" value="history" class="secondary">Order History</button>
              <button type="submit" name="op" value="trade">Place Manual Order</button>
              <button type="submit" name="op" value="model_trade">Run Model Trade</button>
            </div>
          </form>
          <div class="card"><h2>API Output</h2>{details_html or '<p style="color:var(--muted);">Run an action to view responses.</p>'}</div>
        </div></body></html>
        """

    @app.route("/", methods=["GET", "POST"])
    @app.route("/spot", methods=["GET", "POST"])
    def index() -> str:
        is_spot = request.path.startswith("/spot")
        mode_key = SPOT_MODE if is_spot else OPTIONS_MODE
        allow_short = not is_spot
        home_href = "/spot" if is_spot else "/"
        manage_href = "/spot/manage-models" if is_spot else "/manage-models"
        live_href = "/spot/live-trading" if is_spot else "/live-trading"
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
        feature_set = normalize_feature_set(request.form.get("feature_set", "feature2"))
        dqn_episodes_raw = request.form.get("dqn_episodes", "120").strip()
        buy_threshold_raw = request.form.get("buy_threshold", "").strip()
        sell_threshold_raw = request.form.get("sell_threshold", "").strip()
        model_name = request.form.get("model_name", "").strip()
        stop_loss_strategy_raw = request.form.get("stop_loss_strategy", StopLossStrategy.NONE.value).strip()
        fixed_stop_pct_raw = request.form.get("fixed_stop_pct", "2.0").strip()
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
        mode = request.form.get("mode", "train")
        train_action = request.form.get("train_action", "train")
        evaluate_historical_only = train_action == "evaluate_historical"
        if evaluate_historical_only:
            split_style = "chronological"
        data_provider = request.form.get("data_provider", "yfinance").strip().lower()
        if data_provider not in ("yfinance", "twelvedata"):
            data_provider = "yfinance"
        twelve_api_key = os.getenv("TWELVE_DATA_API_KEY", "e90093c59e7a436d9436e34b56a6e6a5").strip()
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
                "new",
                "legacy",
            )
        }

        if request.method == "POST":
            try:
                if mode == "provider_toggle":
                    toggled_provider = request.form.get("toggle_to", "yfinance").strip().lower()
                    data_provider = toggled_provider if toggled_provider in ("yfinance", "twelvedata") else "yfinance"
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
                            webhook_url=webhook_url,
                        )
                        message_html = (
                            "<p style='color:#7bd88f;'><strong>Continuous Run All mode started.</strong> "
                            "Checks every 10 minutes while U.S. market hours are open (Mon-Fri 9:30-16:00 ET).</p>"
                        )
                    elif monitor_action == "stop":
                        RUN_ALL_MONITOR.stop(mode_key)
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
                            feature_set = normalize_feature_set(str(form_state.get("feature_set", feature_set)))
                            buy_threshold_raw = str(form_state.get("buy_threshold", buy_threshold_raw))
                            sell_threshold_raw = str(form_state.get("sell_threshold", sell_threshold_raw))
                            selected_model = str(form_state.get("selected_model", selected_model))
                            model_name = str(form_state.get("model_name", model_name))
                            stop_loss_strategy_raw = str(form_state.get("stop_loss_strategy", stop_loss_strategy_raw))
                            fixed_stop_pct_raw = str(form_state.get("fixed_stop_pct", fixed_stop_pct_raw))
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
                        feature_set = normalize_feature_set(str(form_state.get("feature_set", feature_set)))
                        buy_threshold_raw = str(form_state.get("buy_threshold", buy_threshold_raw))
                        sell_threshold_raw = str(form_state.get("sell_threshold", sell_threshold_raw))
                        selected_model = str(form_state.get("selected_model", selected_model))
                        model_name = str(form_state.get("model_name", model_name))
                        stop_loss_strategy_raw = str(form_state.get("stop_loss_strategy", stop_loss_strategy_raw))
                        fixed_stop_pct_raw = str(form_state.get("fixed_stop_pct", fixed_stop_pct_raw))
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
                    stop_loss_config = StopLossConfig(strategy=stop_loss_strategy, fixed_pct=fixed_stop_pct, model_mae=MODEL_MAE_DEFAULT, time_decay_bars=25)
                    if split_style not in ("shuffled", "chronological"):
                        raise ValueError("Split style must be either shuffled (legacy) or chronological (time-aware).")
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
                                    prediction_horizon=prediction_horizon,
                                )
                                if provider_notice:
                                    provider_notices.append(provider_notice)
                                if len(dataset) < row_count:
                                    multi_rows_used_notes.append(f"{ticker_symbol}: {len(dataset)} frames used")
                                bundle = train_strategy_models(dataset, split_style=split_style, feature_set=feature_set, dqn_episodes=dqn_episodes)
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
                            f"<p class='muted'>Tickers: {', '.join(tickers)} | Rows: {row_count} | Split: {split_style}</p>"
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
                        dataset, provider_notice = fetch_market_rows(
                            ticker=ticker,
                            interval=interval,
                            row_count=row_count,
                            provider=data_provider,
                            twelve_api_key=twelve_api_key,
                            prediction_horizon=prediction_horizon,
                        )
                        if provider_notice:
                            provider_notices.append(provider_notice)
                        if train_action in ("evaluate", "evaluate_historical"):
                            rows_used_note = "" if len(dataset) >= row_count else f"Only {len(dataset)} frames were available and used for evaluation."
                            if train_action == "evaluate_historical":
                                rows_used_note = (
                                    "Historical evaluation mode: forcing chronological split on downloaded market history."
                                    + (f" {rows_used_note}" if rows_used_note else "")
                                )
                            eval_rows = dataset
                        else:
                            rows_used_note = "" if len(dataset) >= row_count else f"Only {len(dataset)} frames were available and used for training."
                            _, eval_rows = train_test_split(dataset, split_style=split_style)
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
                                bundle = train_strategy_models(dataset, split_style=split_style, feature_set=feature_set, dqn_episodes=dqn_episodes)
                                x_test_raw = bundle["x_test_raw"]
                            metrics = evaluate_bundle(
                                bundle,
                                x_test_raw,
                                y_test_ret if use_manual_weights else bundle["y_test_ret"],
                                y_test_dir if use_manual_weights else bundle["y_test_dir"],
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
                                    "prediction_horizon": prediction_horizon,
                                }
                                save_model_configs(mode_key, model_configs)
                                metrics["saved_model"] = model_name_to_save
                                saved_models = list_saved_models(mode_key)
    
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
                        trade_log_rows = "".join(
                            "<tr>"
                            f"<td>{idx + 1}</td>"
                            f"<td>{escape(str(item.get('side', '')))}</td>"
                            f"<td>{escape(str(item.get('entry_label', 'n/a')))}</td>"
                            f"<td>{escape(str(item.get('exit_label', 'n/a')))}</td>"
                            f"<td>{float(item.get('entry_price', 0.0)):.4f}</td>"
                            f"<td>{float(item.get('exit_price', 0.0)):.4f}</td>"
                            f"<td>{float(item.get('bars_held', 0.0)):.0f}</td>"
                            f"<td>{float(item.get('gross_pnl', 0.0)):+.4%}</td>"
                            f"<td>{float(item.get('net_pnl', 0.0)):+.4%}</td>"
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
                                    f"<p><span class='muted'>Mean Sharpe</span> {float(mc_summary.get('mean_sharpe', 0.0)):.3f}</p>"
                                    f"<p><span class='muted'>Mean / Worst Drawdown</span> {float(mc_summary.get('mean_drawdown', 0.0)):.2%} / {float(mc_summary.get('worst_drawdown', 0.0)):.2%}</p>"
                                    f"<p><span class='muted'>Probability of Loss</span> <strong>{float(mc_summary.get('probability_of_loss', 0.0)):.2%}</strong></p>"
                                    f"<p><span class='muted'>P(Return &lt; -50%)</span> {float(mc_summary.get('probability_of_large_loss', 0.0)):.2%}</p>"
                                    f"<p><span class='muted'>P(Ruin &lt; -90%)</span> {float(mc_summary.get('probability_of_ruin', 0.0)):.2%}</p>"
                                    "<details>"
                                    "<summary>Hidden: Return Distribution Plot</summary>"
                                    f"{distribution_chart_html}"
                                    "</details>"
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
                        <p class="muted">{rows_used_note or f"Using requested {row_count} frames."}</p>
                        {model_msg}
                      </div>
                      <div class="card-grid model-overview-grid">
                        {model_cards_html}
                        <article class="card">
                          <h3>Decision Strategy</h3>
                          <p class="muted">{strategy_mode_text} · BUY P&gt;{metrics['strategy']['long_threshold']:.2f} · SELL P&lt;{metrics['strategy']['short_threshold']:.2f} · Stop {metrics['strategy']['stop_loss_strategy']} · Cost 0.05%</p>
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
                          <p class="muted">Includes entry/exit dates, prices, hold time, and P&amp;L after costs/slippage.</p>
                          <table>
                            <tr><th>#</th><th>Side</th><th>Date Bought/Opened</th><th>Date Sold/Closed</th><th>Entry</th><th>Exit</th><th>Bars</th><th>Gross PnL</th><th>Net PnL</th><th>Exit Reason</th></tr>
                            {trade_log_rows if trade_log_rows else "<tr><td colspan='10' class='muted'>No closed trades for this evaluation.</td></tr>"}
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
                                "feature_set": feature_set,
                                "buy_threshold": buy_threshold_raw,
                                "sell_threshold": sell_threshold_raw,
                                "selected_model": selected_model,
                                "model_name": model_name,
                                "stop_loss_strategy": stop_loss_strategy_raw,
                                "fixed_stop_pct": fixed_stop_pct_raw,
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
        monitor_started_at = escape(str(monitor_state.get("started_at", "")))
        monitor_last_tick = escape(str(monitor_state.get("last_tick", "")))
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
                f"data-updated-at='{escape(str(item['updated_at']))}'>"
                f"<strong>{escape(str(item['name']))}</strong>"
                f"<span>{escape(str(item['updated_at']))}</span>"
                "</button>"
            )
            for item in saved_evaluations
        )

        return f"""
        <html>
          <head><title>Quant Model Trainer</title></head>
          <body>
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
                <a href="{home_href}" class="tab-link active">Model</a>
                <a href="{manage_href}" class="tab-link">Manage Models</a>
                <a href="{live_href}" class="tab-link">Live Trading Bot</a>
                <a href="#present-mode" class="tab-link">Present Mode</a>
                <button type="button" id="openEvaluationsBtn" class="secondary topbar-btn">Saved</button>
                <form method="post" style="margin:0;">
                  <input type="hidden" name="mode" value="provider_toggle" />
                  <input type="hidden" name="toggle_to" value="{'yfinance' if data_provider == 'twelvedata' else 'twelvedata'}" />
                  <input type="hidden" name="data_provider" value="{data_provider}" />
                  <button type="submit" class="secondary provider-pill">Data: {'Twelve Data' if data_provider == 'twelvedata' else 'YFinance'}</button>
                </form>
                <a href="{mode_switch_href}" class="tab-link">{mode_switch_label}</a>
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
                </div>
              </label>
              </div>
            </form>
            <form method="post" class="card" id="present-mode">
              <input type="hidden" name="mode" value="present" />
              <input type="hidden" name="data_provider" value="{data_provider}" />
              <h2>Present Mode</h2>
              <p class="muted">Get current model call using the same thresholds used in testing (BUY &gt; 0.60, SELL &lt; 0.40, else HOLD).</p>
              <div class="form-grid">
              <label>Ticker:
                <input type="text" name="present_ticker" value="{present_ticker}" required />
              </label>
              <label>Candle Length:
                <select name="present_interval">
                  <option value="1d" {"selected" if present_interval == "1d" else ""}>Daily</option>
                  <option value="1h" {"selected" if present_interval == "1h" else ""}>1 hour</option>
                  <option value="15m" {"selected" if present_interval == "15m" else ""}>15 min</option>
                  <option value="5m" {"selected" if present_interval == "5m" else ""}>5 min</option>
                </select>
              </label>
              <label>Rows:
                <input type="number" min="50" name="present_rows" value="{present_rows}" required />
              </label>
              <label>Model:
                <select name="present_model">
                  <option value="__new__">Train new model</option>
                  {"".join(f'<option value="{name}" {"selected" if present_model == name else ""}>{name}</option>' for name in saved_models)}
                </select>
              </label>
              <label>BUY if P(Up) &gt; (optional):
                <input type="number" min="0" max="1" step="0.01" name="present_buy_threshold" value="{present_buy_raw}" placeholder="0.60" />
              </label>
              <label>SELL if P(Up) &lt; (optional):
                <input type="number" min="0" max="1" step="0.01" name="present_sell_threshold" value="{present_sell_raw}" placeholder="0.40" />
              </label>
              <label>Stop Loss Strategy:
                <select name="present_stop_loss_strategy" id="presentStopLossStrategy">
                  <option value="none" {"selected" if present_stop_loss_strategy_raw == "none" else ""}>None</option>
                  <option value="atr" {"selected" if present_stop_loss_strategy_raw == "atr" else ""}>Volatility Buffer (ATR-Based)</option>
                  <option value="model_invalidation" {"selected" if present_stop_loss_strategy_raw == "model_invalidation" else ""}>Model Invalidation (MAE-Linked)</option>
                  <option value="time_decay" {"selected" if present_stop_loss_strategy_raw == "time_decay" else ""}>Time-Decay (Temporal Exit)</option>
                  <option value="fixed_percentage" {"selected" if present_stop_loss_strategy_raw == "fixed_percentage" else ""}>Fixed Percentage</option>
                  <option value="trailing_stop" {"selected" if present_stop_loss_strategy_raw == "trailing_stop" else ""}>Trailing Stop Loss</option>
                </select>
              </label>
              <label id="presentFixedStopLossWrap">Present Fixed Stop Loss %:
                <input type="number" min="0.01" step="any" name="present_fixed_stop_pct" value="{present_fixed_stop_pct_raw}" placeholder="2.0" />
              </label>
              <label>&nbsp;
                <button type="submit">Run Present Mode</button>
              </label>
              </div>
            </form>
            <form method="post" class="card">
              <input type="hidden" name="mode" value="present_all" />
              <input type="hidden" name="data_provider" value="{data_provider}" />
              <h2>Run All Present Models</h2>
              <p class="muted">Runs each saved model in this mode only (isolated from the other mode) and shows the live prediction.</p>
              <button type="submit">Run All Present Models</button>
              {run_all_html}
              <table id="runAllTable">
                <tr>
                  <th data-sort-type="text">Model</th>
                  <th data-sort-type="text">Ticker</th>
                  <th data-sort-type="text">Candle</th>
                  <th data-sort-type="number">Rows</th>
                  <th data-sort-type="text">BUY/SELL</th>
                  <th data-sort-type="text">Stop Strategy</th>
                  <th data-sort-type="percent">Expected Return (next candle)</th>
                  <th data-sort-type="percent">P(Up)</th>
                  <th data-sort-type="number">Stop Price</th>
                  <th data-sort-type="text">Action</th>
                </tr>
                {run_all_rows if run_all_rows else "<tr><td colspan='10' class='muted'>Press 'Run All Present Models' to generate outputs.</td></tr>"}
              </table>
            </form>
            <form method="post" class="card">
              <input type="hidden" name="mode" value="run_all_monitor" />
              <input type="hidden" name="data_provider" value="{data_provider}" />
              <h2>Continuous Run All + Discord</h2>
              <p class="muted">When running, this polls every 10 minutes. It only evaluates during U.S. market hours (Mon-Fri 9:30-16:00 ET) and posts to Discord when a model action changes.</p>
              {monitor_status_html}
              <label>Discord Webhook URL:
                <input type="text" name="discord_webhook_url" value="{escape(webhook_url)}" placeholder="https://discord.com/api/webhooks/..." />
              </label>
              <div class="button-row" style="margin-top:0.8rem;">
                <button type="submit" name="monitor_action" value="save_webhook" class="secondary">Save Webhook</button>
                <button type="submit" name="monitor_action" value="start">Start Continuous Mode</button>
                <button type="submit" name="monitor_action" value="stop" class="secondary">Stop Continuous Mode</button>
              </div>
            </form>
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

                  if (mode === "train" && (action === "train" || action === "evaluate")) {{
                    const manualMode = (form.querySelector('select[name="use_manual_weights"]')?.value || "no") === "yes";
                    const seconds = estimateTrainOrEvalSeconds(form, action);
                    const title = action === "train"
                      ? (manualMode ? "Downloading data, saving manual-weight model, and evaluating..." : "Downloading data and training model...")
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
            "new",
            "legacy",
        ],
        default="feature2",
        help="Feature pipeline to use for CLI training/evaluation. Default: feature2.",
    )
    parser.add_argument("--ui", action="store_true", help="Run Flask UI.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Flask host when using --ui.")
    parser.add_argument("--port", type=int, default=5000, help="Flask port when using --ui.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ui:
        app = create_app()
        app.run(host=args.host, port=args.port, debug=False)
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
