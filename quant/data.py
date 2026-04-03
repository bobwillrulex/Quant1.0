from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta, timezone
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
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


def compute_strategy_rows_from_prices(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    prediction_horizon: int = 5,
    timestamps: Sequence[object] | None = None,
) -> List[Row]:
    n = len(closes)
    if prediction_horizon < 1:
        raise ValueError("Prediction horizon must be at least 1.")
    if n < (40 + prediction_horizon):
        raise ValueError("Not enough rows to compute indicators. Need at least 40 + prediction horizon rows.")
    if timestamps is not None and len(timestamps) != n:
        raise ValueError("timestamps must match closes length when provided.")
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
    ema_3 = ema(closes, 3)
    ema_9 = ema(closes, 9)
    ema_21 = ema(closes, 21)
    ema_12 = ema(closes, 12)
    ema_26 = ema(closes, 26)
    macd = [a - b for a, b in zip(ema_12, ema_26)]
    signal = ema(macd, 9)
    macd_hist = [m - s for m, s in zip(macd, signal)]
    macd_hist_delta = [0.0] + [macd_hist[i] - macd_hist[i - 1] for i in range(1, len(macd_hist))]
    ema9_derivative_1 = [0.0] + [ema_9[i] - ema_9[i - 1] for i in range(1, len(ema_9))]
    ema9_derivative_2 = [0.0] + [ema9_derivative_1[i] - ema9_derivative_1[i - 1] for i in range(1, len(ema9_derivative_1))]
    ema9_derivative_3 = [0.0] + [ema9_derivative_2[i] - ema9_derivative_2[i - 1] for i in range(1, len(ema9_derivative_2))]
    ema3_derivative_1 = [0.0] + [ema_3[i] - ema_3[i - 1] for i in range(1, len(ema_3))]
    ema21_derivative_1 = [0.0] + [ema_21[i] - ema_21[i - 1] for i in range(1, len(ema_21))]
    ema26_derivative_1 = [0.0] + [ema_26[i] - ema_26[i - 1] for i in range(1, len(ema_26))]
    ema26_derivative_2 = [0.0] + [ema26_derivative_1[i] - ema26_derivative_1[i - 1] for i in range(1, len(ema26_derivative_1))]
    ema26_derivative_3 = [0.0] + [ema26_derivative_2[i] - ema26_derivative_2[i - 1] for i in range(1, len(ema26_derivative_2))]
    ema_derivative_1_diff = [a - b for a, b in zip(ema9_derivative_1, ema26_derivative_1)]
    ema_derivative_2_diff = [a - b for a, b in zip(ema9_derivative_2, ema26_derivative_2)]
    ema_derivative_3_diff = [a - b for a, b in zip(ema9_derivative_3, ema26_derivative_3)]
    close_returns = [0.0] + [((closes[i] - closes[i - 1]) / closes[i - 1]) if closes[i - 1] != 0 else 0.0 for i in range(1, n)]
    rolling_window = 20
    bb_middle: List[float] = []
    bb_std: List[float] = []
    for i in range(n):
        start = max(0, i - rolling_window + 1)
        window = closes[start : i + 1]
        mean = sum(window) / len(window)
        variance = sum((v - mean) ** 2 for v in window) / len(window)
        bb_middle.append(mean)
        bb_std.append(variance ** 0.5)
    bb_upper = [mid + (2.0 * sd) for mid, sd in zip(bb_middle, bb_std)]
    bb_lower = [mid - (2.0 * sd) for mid, sd in zip(bb_middle, bb_std)]
    hlc3 = [(highs[i] + lows[i] + closes[i]) / 3.0 for i in range(n)]
    range_weight = [max(1e-9, highs[i] - lows[i]) for i in range(n)]
    atr_frac: List[float] = []
    atr_window = 14
    abs_ret_window: List[float] = []
    for value in close_returns:
        abs_ret_window.append(abs(value))
        if len(abs_ret_window) > atr_window:
            abs_ret_window.pop(0)
        atr_frac.append(sum(abs_ret_window) / len(abs_ret_window))
    vwap_lookback = 60
    vwap_anchor_high: List[float] = []
    vwap_anchor_low: List[float] = []
    for i in range(n):
        start = max(0, i - vwap_lookback + 1)
        high_anchor_idx = max(range(start, i + 1), key=lambda idx: highs[idx])
        low_anchor_idx = min(range(start, i + 1), key=lambda idx: lows[idx])
        high_weights = range_weight[high_anchor_idx : i + 1]
        high_prices = hlc3[high_anchor_idx : i + 1]
        low_weights = range_weight[low_anchor_idx : i + 1]
        low_prices = hlc3[low_anchor_idx : i + 1]
        high_denom = sum(high_weights)
        low_denom = sum(low_weights)
        vwap_anchor_high.append(sum(p * w for p, w in zip(high_prices, high_weights)) / (high_denom if high_denom != 0 else 1.0))
        vwap_anchor_low.append(sum(p * w for p, w in zip(low_prices, low_weights)) / (low_denom if low_denom != 0 else 1.0))
    bars_per_regular_session_5m = 78
    session_vwap_5m: List[float] = []
    session_vwap_delta_5m: List[float] = []
    session_vwap_std_1_5m: List[float] = []
    session_vwap_std_2_5m: List[float] = []
    session_vwap_std_1_upper_5m: List[float] = []
    session_vwap_std_1_lower_5m: List[float] = []
    session_vwap_std_2_upper_5m: List[float] = []
    session_vwap_std_2_lower_5m: List[float] = []
    price_to_session_vwap_std_1_upper_5m: List[float] = []
    price_to_session_vwap_std_1_lower_5m: List[float] = []
    session_vwap_std_1_range_5m: List[float] = []
    price_to_session_vwap_std_2_upper_5m: List[float] = []
    price_to_session_vwap_std_2_lower_5m: List[float] = []
    session_vwap_std_2_range_5m: List[float] = []
    session_vwap_delta_to_mean_5m: List[float] = []
    cumulative_weight = 0.0
    cumulative_weighted_price = 0.0
    session_count = 0
    session_mean_price = 0.0
    session_m2 = 0.0
    for i in range(n):
        if i % bars_per_regular_session_5m == 0:
            cumulative_weight = 0.0
            cumulative_weighted_price = 0.0
            session_count = 0
            session_mean_price = 0.0
            session_m2 = 0.0
        weight = range_weight[i]
        cumulative_weight += weight
        cumulative_weighted_price += hlc3[i] * weight
        vwap_now = cumulative_weighted_price / (cumulative_weight if cumulative_weight != 0 else 1.0)
        session_vwap_5m.append(vwap_now)
        if i % bars_per_regular_session_5m == 0:
            session_vwap_delta_5m.append(0.0)
        else:
            session_vwap_delta_5m.append(vwap_now - session_vwap_5m[i - 1])
        session_count += 1
        delta = hlc3[i] - session_mean_price
        session_mean_price += delta / session_count
        delta2 = hlc3[i] - session_mean_price
        session_m2 += delta * delta2
        session_variance = (session_m2 / session_count) if session_count > 0 else 0.0
        session_std = session_variance ** 0.5
        std_1 = session_std
        std_2 = 2.0 * session_std
        std_1_upper = vwap_now + std_1
        std_1_lower = vwap_now - std_1
        std_2_upper = vwap_now + std_2
        std_2_lower = vwap_now - std_2
        session_vwap_std_1_5m.append(std_1)
        session_vwap_std_2_5m.append(std_2)
        session_vwap_std_1_upper_5m.append(std_1_upper)
        session_vwap_std_1_lower_5m.append(std_1_lower)
        session_vwap_std_2_upper_5m.append(std_2_upper)
        session_vwap_std_2_lower_5m.append(std_2_lower)
        price_to_session_vwap_std_1_upper_5m.append(closes[i] - std_1_upper)
        price_to_session_vwap_std_1_lower_5m.append(closes[i] - std_1_lower)
        session_vwap_std_1_range_5m.append(std_1_upper - std_1_lower)
        price_to_session_vwap_std_2_upper_5m.append(closes[i] - std_2_upper)
        price_to_session_vwap_std_2_lower_5m.append(closes[i] - std_2_lower)
        session_vwap_std_2_range_5m.append(std_2_upper - std_2_lower)
        session_vwap_delta_to_mean_5m.append(closes[i] - vwap_now)
    rows: List[Row] = []
    last_bull_gap_low = 0.0
    last_bull_gap_high = 0.0
    last_bear_gap_low = 0.0
    last_bear_gap_high = 0.0
    # Build each row at index i using only information available by candle close i.
    # We intentionally lag FVG features one bar so "first touch/dip" is observable at i
    # without peeking into i+1.
    for i in range(3, n - prediction_horizon):
        session_bar_index = i % bars_per_regular_session_5m
        opening_range_start = i - session_bar_index
        opening_range_end = min(opening_range_start + 2, i)
        opening_range_high_15m = max(highs[j] for j in range(opening_range_start, opening_range_end + 1))
        opening_range_low_15m = min(lows[j] for j in range(opening_range_start, opening_range_end + 1))
        opening_range_width_15m = max(1e-9, opening_range_high_15m - opening_range_low_15m)
        opening_range_mid_15m = (opening_range_high_15m + opening_range_low_15m) / 2.0
        bars_remaining_in_session_5m = float(max(0, bars_per_regular_session_5m - (session_bar_index + 1)))
        intraday_trade_window_open = 1.0 if (session_bar_index >= 3 and session_bar_index <= 63) else 0.0
        near_session_close_5m = 1.0 if session_bar_index >= 72 else 0.0
        post_opening_range_window_15m = 1.0 if session_bar_index >= 3 else 0.0
        stoch_now = stoch_rsi[i] / 100.0
        stoch_prev = stoch_rsi[i - 1] / 100.0 if i > 0 else stoch_now
        gap_up = lows[i] > highs[i - 1]
        gap_down = highs[i] < lows[i - 1]
        if gap_up:
            last_bull_gap_low = highs[i - 1]
            last_bull_gap_high = lows[i]
        if gap_down:
            last_bear_gap_low = highs[i]
            last_bear_gap_high = lows[i - 1]
        close_now = closes[i]
        inside_bull_fvg = 1.0 if (last_bull_gap_low > 0.0 and last_bull_gap_low <= close_now <= last_bull_gap_high) else 0.0
        inside_bear_fvg = 1.0 if (last_bear_gap_low > 0.0 and last_bear_gap_low <= close_now <= last_bear_gap_high) else 0.0
        ret_window = close_returns[max(0, i - 19) : i + 1]
        mean_ret = sum(ret_window) / max(1, len(ret_window))
        ret_var = sum((value - mean_ret) ** 2 for value in ret_window) / max(1, len(ret_window))
        ma_window = closes[max(0, i - 19) : i + 1]
        ma20 = sum(ma_window) / max(1, len(ma_window))
        prev_bullish_gap = max(0.0, lows[i - 1] - highs[i - 3])
        prev_bearish_gap = max(0.0, lows[i - 3] - highs[i - 1])
        macd_green_increasing = 1.0 if (macd_hist[i] > 0 and macd_hist_delta[i] > 0) else 0.0
        macd_red_recovering = 1.0 if (macd_hist[i] < 0 and macd_hist_delta[i] > 0) else 0.0
        macd_green_fading = 1.0 if (macd_hist[i] > 0 and macd_hist_delta[i] < 0) else 0.0
        macd_red_deepening = 1.0 if (macd_hist[i] < 0 and macd_hist_delta[i] < 0) else 0.0
        deriv_1_diff_now = ema_derivative_1_diff[i]
        deriv_1_diff_prev = ema_derivative_1_diff[i - 1]
        deriv_1_cross = 1.0 if (deriv_1_diff_now == 0.0 or (deriv_1_diff_now * deriv_1_diff_prev) < 0.0) else 0.0
        deriv_1_cross_positive = 1.0 if (deriv_1_cross > 0.0 and deriv_1_diff_now >= 0.0) else 0.0
        deriv_1_cross_negative = 1.0 if (deriv_1_cross > 0.0 and deriv_1_diff_now <= 0.0) else 0.0
        deriv_2_diff_now = ema_derivative_2_diff[i]
        deriv_2_diff_prev = ema_derivative_2_diff[i - 1]
        deriv_2_cross = 1.0 if (deriv_2_diff_now == 0.0 or (deriv_2_diff_now * deriv_2_diff_prev) < 0.0) else 0.0
        deriv_2_cross_positive = 1.0 if (deriv_2_cross > 0.0 and deriv_2_diff_now >= 0.0) else 0.0
        deriv_2_cross_negative = 1.0 if (deriv_2_cross > 0.0 and deriv_2_diff_now <= 0.0) else 0.0
        deriv_3_diff_now = ema_derivative_3_diff[i]
        deriv_3_diff_prev = ema_derivative_3_diff[i - 1]
        deriv_3_cross = 1.0 if (deriv_3_diff_now == 0.0 or (deriv_3_diff_now * deriv_3_diff_prev) < 0.0) else 0.0
        deriv_3_cross_positive = 1.0 if (deriv_3_cross > 0.0 and deriv_3_diff_now >= 0.0) else 0.0
        deriv_3_cross_negative = 1.0 if (deriv_3_cross > 0.0 and deriv_3_diff_now <= 0.0) else 0.0
        row: Row = {
                "stoch_rsi": stoch_rsi[i],
                "stoch_velocity": stoch_now - stoch_prev,
                "stoch_low_zone": max(0.0, 0.2 - stoch_now),
                "stoch_high_zone": max(0.0, stoch_now - 0.8),
                "macd_hist": macd_hist[i],
                "macd_hist_delta": macd_hist_delta[i],
                "macd_delta": macd_hist_delta[i],
                "macd_green_increasing": macd_green_increasing,
                "macd_red_recovering": macd_red_recovering,
                "macd_green_fading": macd_green_fading,
                "macd_red_deepening": macd_red_deepening,
                "ema9": ema_9[i],
                "ema3": ema_3[i],
                "ema21": ema_21[i],
                "ema26": ema_26[i],
                "ema3_derivative_1": ema3_derivative_1[i],
                "ema9_derivative_1": ema9_derivative_1[i],
                "ema21_derivative_1": ema21_derivative_1[i],
                "ema9_derivative_2": ema9_derivative_2[i],
                "ema9_derivative_3": ema9_derivative_3[i],
                "ema26_derivative_1": ema26_derivative_1[i],
                "ema26_derivative_2": ema26_derivative_2[i],
                "ema26_derivative_3": ema26_derivative_3[i],
                "ema_derivative_1_diff": deriv_1_diff_now,
                "ema_derivative_2_diff": deriv_2_diff_now,
                "ema_derivative_3_diff": deriv_3_diff_now,
                "ema_derivative_1_cross": deriv_1_cross,
                "ema_derivative_1_cross_positive": deriv_1_cross_positive,
                "ema_derivative_1_cross_negative": deriv_1_cross_negative,
                "ema_derivative_2_cross": deriv_2_cross,
                "ema_derivative_2_cross_positive": deriv_2_cross_positive,
                "ema_derivative_2_cross_negative": deriv_2_cross_negative,
                "ema_derivative_3_cross": deriv_3_cross,
                "ema_derivative_3_cross_positive": deriv_3_cross_positive,
                "ema_derivative_3_cross_negative": deriv_3_cross_negative,
                "ret_1": close_returns[i],
                "ret_3": ((closes[i] / closes[i - 3]) - 1.0) if closes[i - 3] != 0 else 0.0,
                "ret_5": ((closes[i] / closes[i - 5]) - 1.0) if (i >= 5 and closes[i - 5] != 0) else 0.0,
                "ma20": ma20,
                "trend_20": ((close_now - ma20) / ma20) if ma20 != 0 else 0.0,
                "vol_20": ret_var ** 0.5,
                "gap_up": 1.0 if gap_up else 0.0,
                "gap_down": 1.0 if gap_down else 0.0,
                "last_bull_gap_low": last_bull_gap_low,
                "last_bull_gap_high": last_bull_gap_high,
                "last_bear_gap_low": last_bear_gap_low,
                "last_bear_gap_high": last_bear_gap_high,
                "dist_to_bull_fvg": ((close_now - last_bull_gap_low) / close_now) if (close_now != 0 and last_bull_gap_low > 0.0) else 0.0,
                "dist_to_bear_fvg": ((last_bear_gap_high - close_now) / close_now) if (close_now != 0 and last_bear_gap_high > 0.0) else 0.0,
                "inside_bull_fvg": inside_bull_fvg,
                "inside_bear_fvg": inside_bear_fvg,
                "fvg_green_size": prev_bullish_gap,
                "fvg_red_size": prev_bearish_gap,
                "fvg_red_above_green": 1.0 if prev_bearish_gap > 0 else 0.0,
                "first_green_fvg_dip": 1.0 if (prev_bullish_gap > 0 and lows[i] <= highs[i - 3]) else 0.0,
                "first_red_fvg_touch": 1.0 if (prev_bearish_gap > 0 and highs[i] >= lows[i - 3]) else 0.0,
                "rsi": rsi[i],
                "return_next": (closes[i + prediction_horizon] - closes[i]) / closes[i] if closes[i] != 0 else 0.0,
                "close": closes[i],
                "bb_upper": bb_upper[i],
                "bb_middle": bb_middle[i],
                "bb_lower": bb_lower[i],
                "bb_percent_b": ((closes[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])) if abs(bb_upper[i] - bb_lower[i]) > 1e-12 else 0.5,
                "vwap_anchor_high": vwap_anchor_high[i],
                "vwap_anchor_low": vwap_anchor_low[i],
                "session_vwap_5m": session_vwap_5m[i],
                "session_vwap_delta_5m": session_vwap_delta_5m[i],
                "price_vs_session_vwap_5m": close_now - session_vwap_5m[i],
                "session_vwap_delta_to_mean_5m": session_vwap_delta_to_mean_5m[i],
                "session_vwap_std_1_5m": session_vwap_std_1_5m[i],
                "session_vwap_std_2_5m": session_vwap_std_2_5m[i],
                "session_vwap_std_1_upper_5m": session_vwap_std_1_upper_5m[i],
                "session_vwap_std_1_lower_5m": session_vwap_std_1_lower_5m[i],
                "session_vwap_std_2_upper_5m": session_vwap_std_2_upper_5m[i],
                "session_vwap_std_2_lower_5m": session_vwap_std_2_lower_5m[i],
                "price_to_session_vwap_std_1_upper_5m": price_to_session_vwap_std_1_upper_5m[i],
                "price_to_session_vwap_std_1_lower_5m": price_to_session_vwap_std_1_lower_5m[i],
                "session_vwap_std_1_range_5m": session_vwap_std_1_range_5m[i],
                "price_to_session_vwap_std_2_upper_5m": price_to_session_vwap_std_2_upper_5m[i],
                "price_to_session_vwap_std_2_lower_5m": price_to_session_vwap_std_2_lower_5m[i],
                "session_vwap_std_2_range_5m": session_vwap_std_2_range_5m[i],
                "session_bar_index_5m": float(session_bar_index),
                "bars_remaining_in_session_5m": bars_remaining_in_session_5m,
                "intraday_trade_window_open": intraday_trade_window_open,
                "near_session_close_5m": near_session_close_5m,
                "opening_range_high_15m": opening_range_high_15m,
                "opening_range_low_15m": opening_range_low_15m,
                "opening_range_mid_15m": opening_range_mid_15m,
                "opening_range_width_15m": opening_range_high_15m - opening_range_low_15m,
                "opening_range_width_pct_15m": opening_range_width_15m / close_now if close_now != 0 else 0.0,
                "price_vs_opening_range_high_15m": close_now - opening_range_high_15m,
                "price_vs_opening_range_low_15m": close_now - opening_range_low_15m,
                "price_vs_opening_range_mid_15m": close_now - opening_range_mid_15m,
                "opening_range_position_pct_15m": (close_now - opening_range_low_15m) / opening_range_width_15m,
                "opening_range_breakout_up_15m": 1.0 if (session_bar_index >= 3 and close_now > opening_range_high_15m) else 0.0,
                "opening_range_breakdown_15m": 1.0 if (session_bar_index >= 3 and close_now < opening_range_low_15m) else 0.0,
                "post_opening_range_window_15m": post_opening_range_window_15m,
                "session_vwap_reversion_signal_5m": 1.0 if abs(close_now - session_vwap_5m[i]) <= session_vwap_std_1_5m[i] else 0.0,
                "vwap_reclaim_long_signal_5m": 1.0 if (session_vwap_delta_to_mean_5m[i] > 0.0 and closes[i - 1] <= session_vwap_5m[i - 1]) else 0.0,
                "vwap_reclaim_short_signal_5m": 1.0 if (session_vwap_delta_to_mean_5m[i] < 0.0 and closes[i - 1] >= session_vwap_5m[i - 1]) else 0.0,
                "atr_frac": atr_frac[i],
            }
        if timestamps is not None:
            row["timestamp"] = str(timestamps[i])
        rows.append(row)
    if not rows:
        raise ValueError("Could not build feature rows from downloaded candles.")
    return rows


def fetch_yahoo_rows(ticker: str, interval: str, row_count: int, prediction_horizon: int = 5) -> List[Row]:
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
        timestamps = [str(v) for v in history.index.tolist()]
        rows = compute_strategy_rows_from_prices(
            highs=highs,
            lows=lows,
            closes=closes,
            prediction_horizon=prediction_horizon,
            timestamps=timestamps,
        )
        if len(rows) > len(best_rows):
            best_rows = rows
        if len(rows) >= row_count:
            return rows[-row_count:]
    if not best_rows:
        raise ValueError(f"No Yahoo Finance data returned for ticker '{ticker}'.")
    return best_rows


def _twelve_interval(interval: str) -> str:
    interval_map = {"1d": "1day", "1h": "1h", "15m": "15min", "5m": "5min"}
    if interval not in interval_map:
        raise ValueError("Interval must be one of: 1d, 1h, 15m, 5m")
    return interval_map[interval]


def _ticker_is_unavailable_for_twelve_data(ticker: str) -> bool:
    upper = ticker.upper().strip()
    if "-" in upper:
        return True
    non_us_equity_suffixes = (".TO", ".V", ".CN", ".NE", ".AX", ".L")
    return upper.endswith(non_us_equity_suffixes)


def fetch_twelve_data_rows(ticker: str, interval: str, row_count: int, api_key: str, prediction_horizon: int = 5) -> List[Row]:
    if row_count < 50:
        raise ValueError("Please request at least 50 rows.")
    query = urlencode(
        {
            "symbol": ticker,
            "interval": _twelve_interval(interval),
            "outputsize": min(5000, max(50, row_count + 100)),
            "order": "asc",
            "timezone": "Exchange",
            "apikey": api_key,
        }
    )
    endpoint = f"https://api.twelvedata.com/time_series?{query}"
    try:
        with urlopen(endpoint, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise ValueError(f"Twelve Data request failed: {exc}") from exc
    if payload.get("status") == "error":
        message = str(payload.get("message", "Unknown Twelve Data API error"))
        raise ValueError(message)
    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError("Twelve Data returned no candles.")
    highs = [float(item["high"]) for item in values]
    lows = [float(item["low"]) for item in values]
    closes = [float(item["close"]) for item in values]
    timestamps = [str(point.get("datetime", "")) for point in values]
    rows = compute_strategy_rows_from_prices(
        highs=highs,
        lows=lows,
        closes=closes,
        prediction_horizon=prediction_horizon,
        timestamps=timestamps,
    )
    return rows[-row_count:] if len(rows) >= row_count else rows


def _massive_interval(interval: str) -> tuple[int, str]:
    interval_map = {"1d": (1, "day"), "1h": (1, "hour"), "15m": (15, "minute"), "5m": (5, "minute")}
    if interval not in interval_map:
        raise ValueError("Interval must be one of: 1d, 1h, 15m, 5m")
    return interval_map[interval]


def fetch_massive_rows(ticker: str, interval: str, row_count: int, api_key: str, prediction_horizon: int = 5) -> List[Row]:
    if row_count < 50:
        raise ValueError("Please request at least 50 rows.")
    multiplier, timespan = _massive_interval(interval)
    range_days = {"1d": 3650, "1h": 730, "15m": 90, "5m": 60}
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=range_days[interval])
    query = urlencode(
        {
            "adjusted": "true",
            "sort": "asc",
            "limit": min(50000, max(500, row_count + 150)),
            "apiKey": api_key,
        }
    )
    endpoint = (
        "https://api.polygon.io/v2/aggs/ticker/"
        f"{ticker}/range/{multiplier}/{timespan}/{start.date().isoformat()}/{now.date().isoformat()}?{query}"
    )
    try:
        with urlopen(endpoint, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise ValueError(f"Massive request failed: {exc}") from exc
    if payload.get("status") == "ERROR":
        message = str(payload.get("error", "Unknown Massive API error"))
        raise ValueError(message)
    values = payload.get("results")
    if not isinstance(values, list) or not values:
        raise ValueError("Massive returned no candles.")
    highs = [float(item["h"]) for item in values]
    lows = [float(item["l"]) for item in values]
    closes = [float(item["c"]) for item in values]
    timestamps = [datetime.fromtimestamp(float(point["t"]) / 1000.0, tz=timezone.utc).isoformat() for point in values]
    rows = compute_strategy_rows_from_prices(
        highs=highs,
        lows=lows,
        closes=closes,
        prediction_horizon=prediction_horizon,
        timestamps=timestamps,
    )
    return rows[-row_count:] if len(rows) >= row_count else rows


def fetch_market_rows(
    ticker: str,
    interval: str,
    row_count: int,
    provider: str,
    twelve_api_key: str,
    massive_api_key: str,
    prediction_horizon: int = 5,
) -> tuple[List[Row], str | None]:
    selected_provider = provider.strip().lower()
    if selected_provider not in ("yfinance", "twelvedata", "massive"):
        raise ValueError("Data provider must be one of: yfinance, twelvedata, massive.")
    if selected_provider == "yfinance":
        return fetch_yahoo_rows(ticker=ticker, interval=interval, row_count=row_count, prediction_horizon=prediction_horizon), None
    if selected_provider == "massive":
        try:
            rows = fetch_massive_rows(
                ticker=ticker,
                interval=interval,
                row_count=row_count,
                api_key=massive_api_key,
                prediction_horizon=prediction_horizon,
            )
            return rows, None
        except Exception as exc:
            rows = fetch_yahoo_rows(ticker=ticker, interval=interval, row_count=row_count, prediction_horizon=prediction_horizon)
            return rows, f"Massive failed for {ticker} ({exc}). Fell back to yfinance."
    if _ticker_is_unavailable_for_twelve_data(ticker):
        rows = fetch_yahoo_rows(ticker=ticker, interval=interval, row_count=row_count, prediction_horizon=prediction_horizon)
        return rows, (
            f"Twelve Data does not support this instrument in the current setup ({ticker}). "
            "Fell back to yfinance."
        )
    try:
        rows = fetch_twelve_data_rows(ticker=ticker, interval=interval, row_count=row_count, api_key=twelve_api_key, prediction_horizon=prediction_horizon)
        return rows, None
    except Exception as exc:
        rows = fetch_yahoo_rows(ticker=ticker, interval=interval, row_count=row_count, prediction_horizon=prediction_horizon)
        return rows, f"Twelve Data failed for {ticker} ({exc}). Fell back to yfinance."
