#!/usr/bin/env python3
"""
Quant probability model for a momentum/reversal strategy.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import TYPE_CHECKING, Dict

from quant.constants import OPTIONS_MODE, SPOT_MODE
from quant.data import fetch_yahoo_rows, load_csv, synthetic_data
from quant.ml import (
    build_default_strategy_features,
    evaluate_bundle,
    parse_thresholds,
    predict_signal,
    run_model,
    train_strategy_models,
    train_test_split,
)
from quant.stop_loss import MODEL_MAE_DEFAULT, StopLossConfig, StopLossStrategy, parse_stop_loss_strategy, validate_fixed_stop_pct
from quant.storage import (
    list_saved_models,
    load_model_bundle,
    load_model_configs,
    mode_model_dir,
    sanitize_model_name,
    save_model_bundle,
    save_model_configs,
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


def build_run_all_rows(saved_models, model_configs, *, mode: str, long_only: bool) -> str:
    run_all_rows = ""
    for model_name in saved_models:
        cfg = get_model_config(model_name, model_configs)
        if not cfg.get("include_in_run_all", True):
            continue
        try:
            dataset = fetch_yahoo_rows(ticker=str(cfg.get("ticker", "AAPL")), interval=str(cfg.get("interval", "1d")), row_count=int(cfg.get("rows", 250)))
            latest_row = dataset[-1]
            bundle = load_model_bundle(mode, model_name)
            buy_threshold = float(cfg.get("buy_threshold", 0.6))
            sell_threshold = float(cfg.get("sell_threshold", 0.4))
            prediction = predict_signal(bundle, latest_row, buy_threshold=buy_threshold, sell_threshold=sell_threshold, long_only=long_only)
            run_all_rows += ("<tr>" f"<td>{model_name}</td>" f"<td>{cfg.get('ticker')}</td>" f"<td>{cfg.get('interval')}</td>" f"<td>{int(cfg.get('rows', 250))}</td>" f"<td>{buy_threshold:.2f} / {sell_threshold:.2f}</td>" f"<td>{prediction['expected_return']:+.4%}</td>" f"<td>{prediction['p_up']:.2%}</td>" f"<td><strong>{prediction['action']}</strong></td>" "</tr>")
        except Exception as exc:
            run_all_rows += ("<tr>" f"<td>{model_name}</td>" f"<td>{cfg.get('ticker')}</td>" f"<td>{cfg.get('interval')}</td>" f"<td>{int(cfg.get('rows', 250))}</td>" f"<td>{float(cfg.get('buy_threshold', 0.6)):.2f} / {float(cfg.get('sell_threshold', 0.4)):.2f}</td>" "<td colspan='3' style='color:#ff7b7b;'>" f"Run failed: {exc}" "</td>" "</tr>")
    return run_all_rows


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
                    fixed_stop_pct = validate_fixed_stop_pct(float(request.form.get("fixed_stop_pct", "2.0").strip() or "2.0"))
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
            except Exception as exc:
                error_html = f"<p style='color:#ff7b7b;'><strong>Error:</strong> {exc}</p>"
            saved_models = list_saved_models(mode_key)
            model_configs = load_model_configs(mode_key)
            model_configs = {name: get_model_config(name, model_configs) for name in saved_models}

        model_cards = ""
        for model_name in saved_models:
            cfg = get_model_config(model_name, model_configs)
            include_badge = "Included in Run All" if cfg.get("include_in_run_all", True) else "Excluded from Run All"
            model_cards += (
                f"<button type='button' class='model-card' "
                f"data-model='{model_name}' "
                f"data-ticker='{cfg.get('ticker')}' "
                f"data-interval='{cfg.get('interval')}' "
                f"data-rows='{int(cfg.get('rows', 250))}' "
                f"data-include='{1 if cfg.get('include_in_run_all', True) else 0}' "
                f"data-buy='{float(cfg.get('buy_threshold', 0.6)):.2f}' "
                f"data-sell='{float(cfg.get('sell_threshold', 0.4)):.2f}' "
                f"data-stop-loss='{cfg.get('stop_loss_strategy', StopLossStrategy.NONE.value)}' "
                f"data-fixed-stop='{float(cfg.get('fixed_stop_pct', 2.0)):.2f}'>"
                f"<strong>{model_name}</strong>"
                f"<span>{cfg.get('ticker')} • {cfg.get('interval')} • {int(cfg.get('rows', 250))} rows • "
                f"BUY>{float(cfg.get('buy_threshold', 0.6)):.2f} / SELL<{float(cfg.get('sell_threshold', 0.4)):.2f} • SL {cfg.get('stop_loss_strategy', StopLossStrategy.NONE.value)}</span>"
                f"<em>{include_badge}</em>"
                "</button>"
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
              .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.8rem; }}
              .model-card {{ text-align: left; border: 1px solid var(--border); border-radius: 12px; background: {theme_surface}; color: var(--text); padding: 0.8rem; cursor: pointer; display: flex; flex-direction: column; gap: 0.35rem; }}
              .model-card span {{ color: var(--muted); font-size: 0.92rem; }}
              .model-card em {{ color: {theme_badge}; font-style: normal; font-size: 0.82rem; }}
              table {{ width: 100%; border-collapse: collapse; }}
              th, td {{ border-bottom: 1px solid var(--border); padding: 0.45rem 0.35rem; text-align: left; }}
              th {{ color: {theme_table_head}; }}
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
                <a href="{home_href}#present-mode" class="tab-link">Present Mode</a>
                <a href="{mode_switch_href}" class="tab-link">{mode_switch_label}</a>
              </div>
            </nav>
            <div class="container">
              <h1>{heading_label}</h1>
              <p class="muted">Click a model to edit preset settings (ticker, candle length, rows, buy/sell thresholds, include in Run All). Right-click a model for rename/delete.</p>
              {message_html}
              {error_html}
              <div class="card">
                <h2>Saved Models</h2>
                <div class="model-grid">
                  {model_cards if model_cards else "<p class='muted'>No saved models yet. Train and save one from the main page.</p>"}
                </div>
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

            <form id="deleteForm" method="post" style="display:none;">
              <input type="hidden" name="action" value="delete_model" />
              <input type="hidden" name="model_name" id="deleteModelName" />
            </form>

            <script>
              const settingsModal = document.getElementById("settingsModal");
              const renameModal = document.getElementById("renameModal");
              const menu = document.getElementById("contextMenu");
              let menuModelName = "";

              function closeModals() {{
                settingsModal.style.display = "none";
                renameModal.style.display = "none";
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
                wrap.style.display = strategy === "fixed_percentage" ? "block" : "none";
              }}

              function openRename() {{
                if (!menuModelName) return;
                document.getElementById("renameModelName").value = menuModelName;
                document.getElementById("renameInput").value = menuModelName;
                renameModal.style.display = "flex";
                menu.style.display = "none";
              }}

              function deleteModel() {{
                if (!menuModelName) return;
                if (confirm(`Delete model "${{menuModelName}}"?`)) {{
                  document.getElementById("deleteModelName").value = menuModelName;
                  document.getElementById("deleteForm").submit();
                }}
              }}

              document.querySelectorAll(".model-card").forEach((card) => {{
                card.addEventListener("click", () => openModel(card));
                card.addEventListener("contextmenu", (evt) => {{
                  evt.preventDefault();
                  menuModelName = card.dataset.model;
                  menu.style.left = `${{evt.clientX}}px`;
                  menu.style.top = `${{evt.clientY}}px`;
                  menu.style.display = "block";
                }});
              }});
              document.getElementById("cfgStopLossStrategy").addEventListener("change", toggleCfgFixedStop);
              toggleCfgFixedStop();

              window.addEventListener("click", (evt) => {{
                if (evt.target === settingsModal || evt.target === renameModal) {{
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

    @app.route("/", methods=["GET", "POST"])
    @app.route("/spot", methods=["GET", "POST"])
    def index() -> str:
        is_spot = request.path.startswith("/spot")
        mode_key = SPOT_MODE if is_spot else OPTIONS_MODE
        allow_short = not is_spot
        home_href = "/spot" if is_spot else "/"
        manage_href = "/spot/manage-models" if is_spot else "/manage-models"
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
        split_style = request.form.get("split_style", "shuffled")
        buy_threshold_raw = request.form.get("buy_threshold", "").strip()
        sell_threshold_raw = request.form.get("sell_threshold", "").strip()
        model_name = request.form.get("model_name", "").strip()
        stop_loss_strategy_raw = request.form.get("stop_loss_strategy", StopLossStrategy.NONE.value).strip()
        fixed_stop_pct_raw = request.form.get("fixed_stop_pct", "2.0").strip()
        selected_model = request.form.get("selected_model", "__new__")
        present_ticker = request.form.get("present_ticker", ticker).upper().strip()
        present_interval = request.form.get("present_interval", interval)
        present_rows = request.form.get("present_rows", rows)
        present_buy_raw = request.form.get("present_buy_threshold", "").strip()
        present_sell_raw = request.form.get("present_sell_threshold", "").strip()
        present_model = request.form.get("present_model", selected_model)
        mode = request.form.get("mode", "train")
        train_action = request.form.get("train_action", "train")
        saved_models = list_saved_models(mode_key)
        present_html = ""
        run_all_html = ""
        run_all_rows = ""

        if request.method == "POST":
            try:
                if mode == "present":
                    present_row_count = int(present_rows)
                    present_buy_threshold, present_sell_threshold = parse_thresholds(present_buy_raw, present_sell_raw)
                    dataset = fetch_yahoo_rows(ticker=present_ticker, interval=present_interval, row_count=present_row_count)
                    present_rows_used_note = "" if len(dataset) >= present_row_count else f"Only {len(dataset)} frames were available and used for this run."
                    latest_row = dataset[-1]
                    if present_model == "__new__":
                        bundle = train_strategy_models(dataset, split_style=split_style)
                    else:
                        bundle = load_model_bundle(mode_key, present_model)
                    prediction = predict_signal(
                        bundle,
                        latest_row,
                        buy_threshold=present_buy_threshold,
                        sell_threshold=present_sell_threshold,
                        long_only=is_spot,
                    )
                    present_html = f"""
                    <section class="results">
                      <article class="card">
                        <h2>Present Mode • {present_ticker} ({present_interval})</h2>
                        <p class="muted">Model: {"Freshly trained on current dataset" if present_model == "__new__" else present_model}</p>
                        <p class="muted">{present_rule_text} (BUY&gt;{present_buy_threshold:.2f}, SELL&lt;{present_sell_threshold:.2f}).</p>
                        <p class="muted">{present_rows_used_note or f"Using requested {present_row_count} frames."}</p>
                        <p><span class="muted">Expected Return (next candle)</span> <strong>{prediction['expected_return']:+.4%}</strong></p>
                        <p><span class="muted">P(Up)</span> <strong>{prediction['p_up']:.2%}</strong></p>
                        <p><span class="muted">Action</span> <strong>{prediction['action']}</strong></p>
                      </article>
                    </section>
                    """
                elif mode == "present_all":
                    model_configs = load_model_configs(mode_key)
                    model_configs = {name: get_model_config(name, model_configs) for name in saved_models}
                    run_all_rows = build_run_all_rows(saved_models, model_configs, mode=mode_key, long_only=is_spot)
                    run_all_html = "<p class='muted'>Latest outputs for all models currently included in Run All.</p>"
                else:
                    row_count = int(rows)
                    buy_threshold, sell_threshold = parse_thresholds(buy_threshold_raw, sell_threshold_raw)
                    stop_loss_strategy = parse_stop_loss_strategy(stop_loss_strategy_raw)
                    fixed_stop_pct = validate_fixed_stop_pct(float(fixed_stop_pct_raw or "2.0"))
                    stop_loss_config = StopLossConfig(strategy=stop_loss_strategy, fixed_pct=fixed_stop_pct, model_mae=MODEL_MAE_DEFAULT, time_decay_bars=25)
                    if split_style not in ("shuffled", "chronological"):
                        raise ValueError("Split style must be either shuffled (legacy) or chronological (time-aware).")
                    tickers = parse_csv_values(ticker, uppercase=True)
                    if not tickers:
                        raise ValueError("Please enter at least one ticker symbol.")
                    model_names = parse_csv_values(model_name, uppercase=False) if model_name else []
                    if len(tickers) > 1:
                        if selected_model != "__new__":
                            raise ValueError("Multi-ticker run only supports training new models (not loading an existing saved model).")
                        if model_names and len(model_names) != len(tickers):
                            raise ValueError("When training multiple tickers, provide the same number of model names as tickers.")
                        multi_rows = []
                        model_configs = load_model_configs(mode_key)
                        multi_rows_used_notes = []
                        for idx, ticker_symbol in enumerate(tickers):
                            dataset = fetch_yahoo_rows(ticker=ticker_symbol, interval=interval, row_count=row_count)
                            if len(dataset) < row_count:
                                multi_rows_used_notes.append(f"{ticker_symbol}: {len(dataset)} frames used")
                            bundle = train_strategy_models(dataset, split_style=split_style)
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
                            )
                            trained_model_name = ""
                            if train_action == "train":
                                candidate_name = model_names[idx] if model_names else ""
                                if candidate_name:
                                    trained_model_name = sanitize_model_name(candidate_name)
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
                        if train_action == "train" and model_names:
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
                        dataset = fetch_yahoo_rows(ticker=ticker, interval=interval, row_count=row_count)
                        features = build_default_strategy_features()
                        if train_action == "evaluate":
                            rows_used_note = "" if len(dataset) >= row_count else f"Only {len(dataset)} frames were available and used for evaluation."
                            eval_rows = dataset
                        else:
                            rows_used_note = "" if len(dataset) >= row_count else f"Only {len(dataset)} frames were available and used for training."
                            _, eval_rows = train_test_split(dataset, split_style=split_style)
                        x_test_raw = features.transform(eval_rows)
                        y_test_ret = [r["return_next"] for r in eval_rows]
                        y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]
                        if selected_model != "__new__":
                            loaded = load_model_bundle(mode_key, selected_model)
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
                            )
                            metrics["train_size"] = "saved-model"
                            metrics["loaded_model"] = selected_model
                        else:
                            bundle = train_strategy_models(dataset, split_style=split_style)
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
                            )
                            metrics["train_size"] = bundle["train_size"]
                            if train_action == "train" and model_name:
                                save_model_bundle(mode_key, model_name, bundle)
                                model_configs = load_model_configs(mode_key)
                                model_configs[sanitize_model_name(model_name)] = {
                                    "ticker": ticker,
                                    "interval": interval,
                                    "rows": row_count,
                                    "include_in_run_all": True,
                                    "buy_threshold": buy_threshold,
                                    "sell_threshold": sell_threshold,
                                    "stop_loss_strategy": stop_loss_strategy.value,
                                    "fixed_stop_pct": fixed_stop_pct,
                                }
                                save_model_configs(mode_key, model_configs)
                                metrics["saved_model"] = model_name
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
                        hold_time_boxplot = render_hold_time_boxplot(
                            metrics["strategy"]["hold_time_stats"],
                            stroke=theme_border,
                            accent=theme_table_head,
                        )
                        result_html = f"""
                    <section class="results">
                      <div class="section-heading">
                        <h2>Results • {ticker} ({interval})</h2>
                        <p class="muted">Rows: {row_count} | Train: {metrics['train_size']} | Test: {metrics['test_size']} | Split: {metrics['split_style']}</p>
                        <p class="muted">{rows_used_note or f"Using requested {row_count} frames."}</p>
                        {model_msg}
                      </div>
                      <div class="card-grid">
                        <article class="card">
                          <h3>Linear Model</h3>
                          <p><span class="muted">Test MSE</span> <strong>{metrics['mse']:.8f}</strong></p>
                          <p><span class="muted">Test MAE</span> <strong>{metrics['mae']:.8f}</strong></p>
                          <p><span class="muted">Zero baseline (MSE/MAE)</span><br>{metrics['baseline_zero_mse']:.8f} / {metrics['baseline_zero_mae']:.8f}</p>
                          <p><span class="muted">Edge vs baseline (MSE/MAE)</span><br>{metrics['mse_vs_zero_baseline']:+.8f} / {metrics['mae_vs_zero_baseline']:+.8f}</p>
                        </article>
                        <article class="card">
                          <h3>Logistic Model</h3>
                          <p><span class="muted">Accuracy</span> <strong>{metrics['accuracy']:.4f}</strong></p>
                          <p><span class="muted">Always-UP baseline</span> {metrics['baseline_always_up_accuracy']:.4f} (edge {metrics['accuracy_vs_baseline']:+.4f})</p>
                          <p><span class="muted">Precision / Recall / F1</span><br>{metrics['precision']:.4f} / {metrics['recall']:.4f} / {metrics['f1']:.4f}</p>
                          <p><span class="muted">Confusion Matrix</span><br>TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}</p>
                        </article>
                        <article class="card">
                          <h3>Decision Strategy</h3>
                          <p class="muted">{strategy_mode_text} · BUY P&gt;{metrics['strategy']['long_threshold']:.2f} · SELL P&lt;{metrics['strategy']['short_threshold']:.2f} · Stop {metrics['strategy']['stop_loss_strategy']} · Cost 0.05%</p>
                          <p><span class="muted">Total Return</span> <strong>{metrics['strategy']['total_return']:+.2%}</strong></p>
                          <p><span class="muted">Buy &amp; Hold Return (test rows)</span> <strong>{metrics['strategy']['buy_hold_total_return']:+.2%}</strong></p>
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
                      </div>
    
                      <article class="card">
                        <h3>Feature Set</h3>
                        <p>{', '.join(metrics['features'])}</p>
                      </article>
    
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
                            <tr><td>P &gt; 0.6</td><td>{int(metrics['confidence_edge']['p_gt_0.6']['count'])}</td><td>{metrics['confidence_edge']['p_gt_0.6']['accuracy']:.4f}</td></tr>
                            <tr><td>P &gt; 0.7</td><td>{int(metrics['confidence_edge']['p_gt_0.7']['count'])}</td><td>{metrics['confidence_edge']['p_gt_0.7']['accuracy']:.4f}</td></tr>
                          </table>
                        </article>
                      </div>
    
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
    
                      <article class="card table-card">
                        <h3>Example Predictions</h3>
                        <table>
                          <tr><th>Row</th><th>Expected Return</th><th>P(Up)</th><th>Actual Return</th></tr>
                          {preview_rows}
                        </table>
                      </article>
                    </section>
                    """
            except Exception as exc:
                error_html = f"<p style='color:red;'><strong>Error:</strong> {exc}</p>"

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
            </style>
            <nav class="topbar">
              <div class="topbar-inner">
                <a href="{home_href}" class="brand">{brand_label}</a>
                <a href="{home_href}" class="tab-link active">Model</a>
                <a href="{manage_href}" class="tab-link">Manage Models</a>
                <a href="#present-mode" class="tab-link">Present Mode</a>
                <a href="{mode_switch_href}" class="tab-link">{mode_switch_label}</a>
              </div>
            </nav>
            <div class="container">
            <h1>{trainer_heading}</h1>
            <form method="post" class="card">
              <input type="hidden" name="mode" value="train" />
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
              <label>Split Style:
                <select name="split_style">
                  <option value="shuffled" {"selected" if split_style == "shuffled" else ""}>Legacy (shuffled)</option>
                  <option value="chronological" {"selected" if split_style == "chronological" else ""}>Time-aware (chronological)</option>
                </select>
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
                </select>
              </label>
              <label id="fixedStopLossWrap">Fixed Stop Loss %:
                <input type="number" min="0.01" step="0.1" name="fixed_stop_pct" value="{fixed_stop_pct_raw}" placeholder="2.0" />
              </label>
              <label>&nbsp;
                <div class="button-row">
                  <button type="submit" name="train_action" value="train">Download + Train</button>
                  <button type="submit" name="train_action" value="evaluate" class="secondary">Evaluate Only</button>
                </div>
              </label>
              </div>
            </form>
            <form method="post" class="card" id="present-mode">
              <input type="hidden" name="mode" value="present" />
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
              <label>&nbsp;
                <button type="submit">Run Present Mode</button>
              </label>
              </div>
            </form>
            <form method="post" class="card">
              <input type="hidden" name="mode" value="present_all" />
              <h2>Run All Present Models</h2>
              <p class="muted">Runs each saved model in this mode only (isolated from the other mode) and shows the live prediction.</p>
              <button type="submit">Run All Present Models</button>
              {run_all_html}
              <table>
                <tr><th>Model</th><th>Ticker</th><th>Candle</th><th>Rows</th><th>BUY/SELL</th><th>Expected Return (next candle)</th><th>P(Up)</th><th>Action</th></tr>
                {run_all_rows if run_all_rows else "<tr><td colspan='8' class='muted'>Press 'Run All Present Models' to generate outputs.</td></tr>"}
              </table>
            </form>
            {error_html}
            {present_html}
            {result_html}
            </div>
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
              let loadingTimer = null;

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

              document.querySelectorAll("form").forEach((form) => {{
                form.addEventListener("submit", (evt) => {{
                  const mode = form.querySelector('input[name="mode"]')?.value || "";
                  const submitter = evt.submitter;
                  const action = submitter?.value || "";

                  if (mode === "train" && action === "train") {{
                    const rows = Number(form.querySelector('input[name="rows"]')?.value || "250");
                    const seconds = Math.min(95, Math.max(12, Math.round(rows / 7)));
                    showLoading("Downloading data and training model...", seconds);
                  }} else if (mode === "train" && action === "evaluate") {{
                    const rows = Number(form.querySelector('input[name="rows"]')?.value || "250");
                    const seconds = Math.min(70, Math.max(8, Math.round(rows / 10)));
                    showLoading("Downloading data and evaluating model...", seconds);
                  }} else if (mode === "present_all") {{
                    const modelCount = Math.max(1, document.querySelectorAll('table tr').length - 1);
                    const seconds = Math.min(120, Math.max(10, modelCount * 8));
                    showLoading("Running all present-mode models...", seconds);
                  }}
                }});
              }});

              const stopLossStrategyEl = document.getElementById("stopLossStrategy");
              const fixedStopLossWrapEl = document.getElementById("fixedStopLossWrap");
              function toggleFixedStopField() {{
                if (!stopLossStrategyEl || !fixedStopLossWrapEl) return;
                fixedStopLossWrapEl.style.display = stopLossStrategyEl.value === "fixed_percentage" ? "block" : "none";
              }}
              if (stopLossStrategyEl) {{
                stopLossStrategyEl.addEventListener("change", toggleFixedStopField);
                toggleFixedStopField();
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
        help="Optional CSV path with columns: stoch_rsi, macd_hist, macd_hist_delta, fvg_green_size, fvg_red_size, fvg_red_above_green, first_green_fvg_dip, first_red_fvg_touch, return_next",
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

    run_model(rows)


if __name__ == "__main__":
    main()
