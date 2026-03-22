from __future__ import annotations

import csv
import random
from typing import List, Sequence

from .types import Row


def load_csv(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            rows.append(
                {
                    "stoch_rsi": float(record["stoch_rsi"]),
                    "macd_hist": float(record["macd_hist"]),
                    "macd_hist_delta": float(record.get("macd_hist_delta", 0.0)),
                    "fvg_green_size": float(record["fvg_green_size"]),
                    "fvg_red_size": float(record.get("fvg_red_size", 0.0)),
                    "fvg_red_above_green": float(record["fvg_red_above_green"]),
                    "first_green_fvg_dip": float(record["first_green_fvg_dip"]),
                    "first_red_fvg_touch": float(record.get("first_red_fvg_touch", 0.0)),
                    "return_next": float(record["return_next"]),
                }
            )
    return rows


def synthetic_data(n: int = 1200, seed: int = 42) -> List[Row]:
    random.seed(seed)
    rows: List[Row] = []
    for _ in range(n):
        stoch_rsi = random.uniform(0, 100)
        macd_hist = random.uniform(-1.0, 1.0)
        macd_hist_delta = random.uniform(-0.4, 0.4)
        fvg_green_size = max(0.0, random.gauss(0.5, 0.6))
        fvg_red_size = max(0.0, random.gauss(0.45, 0.6))
        fvg_red_above_green = 1.0 if random.random() < 0.35 else 0.0
        first_green_fvg_dip = 1.0 if random.random() < 0.25 else 0.0
        first_red_fvg_touch = 1.0 if random.random() < 0.23 else 0.0
        stoch_extreme = 1.0 if (stoch_rsi > 80 or stoch_rsi < 20) else 0.0
        macd_green_increasing = 1.0 if (macd_hist > 0 and macd_hist_delta > 0) else 0.0
        macd_red_recovering = 1.0 if (macd_hist < 0 and macd_hist_delta > 0) else 0.0
        macd_green_falling = 1.0 if (macd_hist > 0 and macd_hist_delta < 0) else 0.0
        macd_red_deepening = 1.0 if (macd_hist < 0 and macd_hist_delta < 0) else 0.0
        alpha = (
            0.0035 * stoch_extreme
            + 0.0032 * macd_green_increasing
            + 0.0012 * macd_red_recovering
            - 0.0025 * macd_green_falling
            - 0.0040 * macd_red_deepening
            + 0.0030 * first_green_fvg_dip * min(fvg_green_size, 3.0)
            - 0.0020 * fvg_red_above_green * min(fvg_green_size, 3.0)
            - 0.0030 * first_red_fvg_touch * min(fvg_red_size, 3.0)
            + random.gauss(0, 0.0018)
        )
        rows.append({"stoch_rsi": stoch_rsi, "macd_hist": macd_hist, "macd_hist_delta": macd_hist_delta, "fvg_green_size": fvg_green_size, "fvg_red_size": fvg_red_size, "fvg_red_above_green": fvg_red_above_green, "first_green_fvg_dip": first_green_fvg_dip, "first_red_fvg_touch": first_red_fvg_touch, "return_next": alpha})
    return rows


def ema(values: Sequence[float], span: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    result = [values[0]]
    for i in range(1, len(values)):
        result.append(alpha * values[i] + (1.0 - alpha) * result[-1])
    return result


def compute_strategy_rows_from_prices(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> List[Row]:
    n = len(closes)
    if n < 40:
        raise ValueError("Not enough rows to compute indicators. Need at least 40 rows.")
    deltas = [0.0] + [closes[i] - closes[i - 1] for i in range(1, n)]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = ema(gains, 14)
    avg_loss = ema(losses, 14)
    rsi: List[float] = []
    for g, l in zip(avg_gain, avg_loss):
        rs = g / l if l > 1e-12 else 100.0
        rsi.append(100.0 - (100.0 / (1.0 + rs)))
    stoch_rsi: List[float] = []
    lookback = 14
    for i in range(n):
        start = max(0, i - lookback + 1)
        window = rsi[start : i + 1]
        lo = min(window)
        hi = max(window)
        stoch_rsi.append(50.0 if hi - lo < 1e-12 else 100.0 * ((rsi[i] - lo) / (hi - lo)))
    ema_12 = ema(closes, 12)
    ema_26 = ema(closes, 26)
    macd = [a - b for a, b in zip(ema_12, ema_26)]
    signal = ema(macd, 9)
    macd_hist = [m - s for m, s in zip(macd, signal)]
    macd_hist_delta = [0.0] + [macd_hist[i] - macd_hist[i - 1] for i in range(1, len(macd_hist))]
    close_returns = [0.0] + [((closes[i] - closes[i - 1]) / closes[i - 1]) if closes[i - 1] != 0 else 0.0 for i in range(1, n)]
    atr_frac: List[float] = []
    atr_window = 14
    abs_ret_window: List[float] = []
    for value in close_returns:
        abs_ret_window.append(abs(value))
        if len(abs_ret_window) > atr_window:
            abs_ret_window.pop(0)
        atr_frac.append(sum(abs_ret_window) / len(abs_ret_window))
    rows: List[Row] = []
    for i in range(2, n - 1):
        bullish_gap = max(0.0, lows[i] - highs[i - 2])
        bearish_gap = max(0.0, lows[i - 2] - highs[i])
        rows.append(
            {
                "stoch_rsi": stoch_rsi[i],
                "macd_hist": macd_hist[i],
                "macd_hist_delta": macd_hist_delta[i],
                "fvg_green_size": bullish_gap,
                "fvg_red_size": bearish_gap,
                "fvg_red_above_green": 1.0 if bearish_gap > 0 else 0.0,
                "first_green_fvg_dip": 1.0 if (bullish_gap > 0 and lows[i + 1] <= highs[i - 2]) else 0.0,
                "first_red_fvg_touch": 1.0 if (bearish_gap > 0 and highs[i + 1] >= lows[i - 2]) else 0.0,
                "return_next": (closes[i + 1] - closes[i]) / closes[i] if closes[i] != 0 else 0.0,
                "close": closes[i],
                "atr_frac": atr_frac[i],
            }
        )
    if not rows:
        raise ValueError("Could not build feature rows from downloaded candles.")
    return rows


def fetch_yahoo_rows(ticker: str, interval: str, row_count: int) -> List[Row]:
    import yfinance as yf

    interval_periods = {"1d": ["1y", "5y", "10y", "max"], "1h": ["730d"], "15m": ["60d"], "5m": ["60d"]}
    if interval not in interval_periods:
        raise ValueError("Interval must be one of: 1d, 1h, 15m, 5m")
    if row_count < 50:
        raise ValueError("Please request at least 50 rows.")
    ticker_obj = yf.Ticker(ticker)
    rows: List[Row] = []
    best_rows: List[Row] = []
    for period in interval_periods[interval]:
        history = ticker_obj.history(period=period, interval=interval, auto_adjust=False)
        if history.empty:
            continue
        highs = [float(v) for v in history["High"].tolist()]
        lows = [float(v) for v in history["Low"].tolist()]
        closes = [float(v) for v in history["Close"].tolist()]
        rows = compute_strategy_rows_from_prices(highs=highs, lows=lows, closes=closes)
        if len(rows) > len(best_rows):
            best_rows = rows
        if len(rows) >= row_count:
            return rows[-row_count:]
    if not best_rows:
        raise ValueError(f"No Yahoo Finance data returned for ticker '{ticker}'.")
    return best_rows
