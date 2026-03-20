#!/usr/bin/env python3
"""
Quant probability model for a momentum/reversal strategy.

Strategy summary:
This strategy treats indicators as probabilistic inputs (not direct buy/sell signals).
It combines three ideas: (1) Stoch RSI extremes (>80 or <20) indicate a potential expansion
move where timing matters, (2) MACD state and momentum change indicate whether trend pressure
is weakening or improving, and (3) Fair Value Gaps (FVGs) provide location/context where the
first dip into a green FVG is considered high-probability for reversal, while overlapping red
FVG above green FVG can cap upside and increase reversal risk. Larger FVGs are modeled as more
influential by scaling their feature contribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:
    from flask import Flask


Row = Dict[str, float]
FeatureFn = Callable[[Row], float]
MODEL_DIR = "saved_models"


@dataclass
class FeatureSpec:
    name: str
    fn: FeatureFn


class StrategyFeatureBuilder:
    """Composable feature builder so you can add/replace strategy logic easily."""

    def __init__(self) -> None:
        self._features: List[FeatureSpec] = []

    def add(self, name: str, fn: FeatureFn) -> "StrategyFeatureBuilder":
        self._features.append(FeatureSpec(name=name, fn=fn))
        return self

    def names(self) -> List[str]:
        return [f.name for f in self._features]

    def transform(self, rows: Sequence[Row]) -> List[List[float]]:
        matrix: List[List[float]] = []
        for row in rows:
            matrix.append([feature.fn(row) for feature in self._features])
        return matrix


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class LinearRegressionGD:
    def __init__(self, learning_rate: float = 0.01, epochs: int = 2000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias = 0.0

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        n = len(x)
        d = len(x[0]) if n else 0
        self.weights = [0.0] * d
        self.bias = 0.0

        for _ in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0

            for i in range(n):
                pred = self.predict_one(x[i])
                err = pred - y[i]
                for j in range(d):
                    grad_w[j] += err * x[i][j]
                grad_b += err

            inv_n = 1.0 / max(1, n)
            for j in range(d):
                self.weights[j] -= self.learning_rate * grad_w[j] * inv_n
            self.bias -= self.learning_rate * grad_b * inv_n

    def predict_one(self, x: Sequence[float]) -> float:
        return sum(w * v for w, v in zip(self.weights, x)) + self.bias

    def predict(self, x: Sequence[Sequence[float]]) -> List[float]:
        return [self.predict_one(row) for row in x]


class LogisticRegressionGD:
    def __init__(self, learning_rate: float = 0.05, epochs: int = 2000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias = 0.0

    def fit(self, x: Sequence[Sequence[float]], y: Sequence[int]) -> None:
        n = len(x)
        d = len(x[0]) if n else 0
        self.weights = [0.0] * d
        self.bias = 0.0

        for _ in range(self.epochs):
            grad_w = [0.0] * d
            grad_b = 0.0

            for i in range(n):
                pred = self.predict_proba_one(x[i])
                err = pred - y[i]
                for j in range(d):
                    grad_w[j] += err * x[i][j]
                grad_b += err

            inv_n = 1.0 / max(1, n)
            for j in range(d):
                self.weights[j] -= self.learning_rate * grad_w[j] * inv_n
            self.bias -= self.learning_rate * grad_b * inv_n

    def predict_proba_one(self, x: Sequence[float]) -> float:
        z = sum(w * v for w, v in zip(self.weights, x)) + self.bias
        return sigmoid(z)

    def predict_proba(self, x: Sequence[Sequence[float]]) -> List[float]:
        return [self.predict_proba_one(row) for row in x]


def train_test_split(rows: Sequence[Row], test_ratio: float = 0.25) -> Tuple[List[Row], List[Row]]:
    data = list(rows)
    random.shuffle(data)
    split = max(1, int(len(data) * (1 - test_ratio)))
    return data[:split], data[split:]


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    n = len(y_true)
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / max(1, n)


def accuracy(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> float:
    preds = [1 if p >= threshold else 0 for p in y_prob]
    correct = sum(int(a == b) for a, b in zip(y_true, preds))
    return correct / max(1, len(y_true))


def classification_metrics(y_true: Sequence[int], y_prob: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    preds = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, preds) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, preds) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, preds) if a == 1 and b == 0)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2.0 * precision * recall) / max(1e-12, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))


def load_csv(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            row: Row = {
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
            rows.append(row)
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

        rows.append(
            {
                "stoch_rsi": stoch_rsi,
                "macd_hist": macd_hist,
                "macd_hist_delta": macd_hist_delta,
                "fvg_green_size": fvg_green_size,
                "fvg_red_size": fvg_red_size,
                "fvg_red_above_green": fvg_red_above_green,
                "first_green_fvg_dip": first_green_fvg_dip,
                "first_red_fvg_touch": first_red_fvg_touch,
                "return_next": alpha,
            }
        )
    return rows


def build_default_strategy_features() -> StrategyFeatureBuilder:
    builder = StrategyFeatureBuilder()

    # 1) Stoch RSI extremes imply setup urgency / likely expansion move.
    builder.add("stoch_extreme", lambda r: 1.0 if (r["stoch_rsi"] > 80 or r["stoch_rsi"] < 20) else 0.0)

    # 2) MACD histogram-only regime features (best -> worst):
    # green increasing > red improving > green fading > red deepening.
    builder.add("macd_hist", lambda r: r["macd_hist"])
    builder.add("macd_hist_delta", lambda r: r["macd_hist_delta"])
    builder.add("macd_green_increasing", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_red_recovering", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_green_fading", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] < 0) else 0.0)
    builder.add("macd_red_deepening", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] < 0) else 0.0)

    # 3) FVG context and size weighting for both green and red gaps.
    builder.add("first_green_fvg_dip", lambda r: r["first_green_fvg_dip"])
    builder.add("first_red_fvg_touch", lambda r: r["first_red_fvg_touch"])
    builder.add("fvg_green_size", lambda r: min(r["fvg_green_size"], 3.0))
    builder.add("fvg_red_size", lambda r: min(r["fvg_red_size"], 3.0))
    builder.add(
        "fvg_bull_signal",
        lambda r: r["first_green_fvg_dip"] * min(r["fvg_green_size"], 3.0),
    )
    builder.add(
        "fvg_bear_signal",
        lambda r: r["first_red_fvg_touch"] * min(r["fvg_red_size"], 3.0),
    )
    builder.add(
        "fvg_conflict_penalty",
        lambda r: r["fvg_red_above_green"] * min(r["fvg_green_size"], 3.0),
    )

    return builder


def standardize_fit(x: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    n = len(x)
    d = len(x[0]) if n else 0
    means = [0.0] * d
    stds = [1.0] * d

    for j in range(d):
        col = [x[i][j] for i in range(n)]
        mu = sum(col) / max(1, n)
        var = sum((v - mu) ** 2 for v in col) / max(1, n)
        sd = math.sqrt(var) if var > 1e-12 else 1.0
        means[j] = mu
        stds[j] = sd

    x_scaled = [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in x]
    return x_scaled, means, stds


def standardize_apply(x: Sequence[Sequence[float]], means: Sequence[float], stds: Sequence[float]) -> List[List[float]]:
    return [[(row[j] - means[j]) / stds[j] for j in range(len(row))] for row in x]


def ensure_model_dir() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)


def train_strategy_models(rows: Sequence[Row]) -> Dict[str, object]:
    train_rows, test_rows = train_test_split(rows)
    features = build_default_strategy_features()
    x_train_raw = features.transform(train_rows)
    x_test_raw = features.transform(test_rows)
    x_train, means, stds = standardize_fit(x_train_raw)
    x_test = standardize_apply(x_test_raw, means, stds)

    y_train_ret = [r["return_next"] for r in train_rows]
    y_test_ret = [r["return_next"] for r in test_rows]
    y_train_dir = [1 if r > 0 else 0 for r in y_train_ret]
    y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]

    lin = LinearRegressionGD(learning_rate=0.03, epochs=3000)
    lin.fit(x_train, y_train_ret)

    logit = LogisticRegressionGD(learning_rate=0.05, epochs=2500)
    logit.fit(x_train, y_train_dir)

    return {
        "feature_names": features.names(),
        "means": means,
        "stds": stds,
        "lin_weights": lin.weights,
        "lin_bias": lin.bias,
        "logit_weights": logit.weights,
        "logit_bias": logit.bias,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
        "x_test_raw": x_test_raw,
        "y_test_ret": y_test_ret,
        "y_test_dir": y_test_dir,
    }


def evaluate_bundle(bundle: Dict[str, object], x_test_raw: Sequence[Sequence[float]], y_test_ret: Sequence[float], y_test_dir: Sequence[int]) -> Dict[str, object]:
    if not x_test_raw:
        raise ValueError("No rows available for evaluation.")
    if len(x_test_raw[0]) != len(bundle["feature_names"]):
        raise ValueError("Saved model feature size does not match current strategy feature set.")
    x_test = standardize_apply(x_test_raw, bundle["means"], bundle["stds"])
    ret_pred = [
        sum(w * v for w, v in zip(bundle["lin_weights"], row)) + bundle["lin_bias"]
        for row in x_test
    ]
    up_prob = [
        sigmoid(sum(w * v for w, v in zip(bundle["logit_weights"], row)) + bundle["logit_bias"])
        for row in x_test
    ]
    cls = classification_metrics(y_test_dir, up_prob)

    preview = []
    for i in range(min(5, len(x_test))):
        preview.append({"expected_return": ret_pred[i], "p_up": up_prob[i], "actual_return": y_test_ret[i]})

    return {
        "features": bundle["feature_names"],
        "mse": mse(y_test_ret, ret_pred),
        "mae": mae(y_test_ret, ret_pred),
        "accuracy": cls["accuracy"],
        "precision": cls["precision"],
        "recall": cls["recall"],
        "f1": cls["f1"],
        "tp": int(cls["tp"]),
        "tn": int(cls["tn"]),
        "fp": int(cls["fp"]),
        "fn": int(cls["fn"]),
        "lin_weights": list(zip(bundle["feature_names"], bundle["lin_weights"])),
        "lin_bias": bundle["lin_bias"],
        "logit_weights": list(zip(bundle["feature_names"], bundle["logit_weights"])),
        "logit_bias": bundle["logit_bias"],
        "preview": preview,
        "test_size": len(y_test_ret),
    }


def save_model_bundle(model_name: str, bundle: Dict[str, object]) -> str:
    ensure_model_dir()
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name).strip("_")
    if not safe_name:
        raise ValueError("Model name must include letters or numbers.")
    path = os.path.join(MODEL_DIR, f"{safe_name}.json")
    payload = {k: bundle[k] for k in ["feature_names", "means", "stds", "lin_weights", "lin_bias", "logit_weights", "logit_bias"]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def list_saved_models() -> List[str]:
    ensure_model_dir()
    return sorted([f[:-5] for f in os.listdir(MODEL_DIR) if f.endswith(".json")])


def load_model_bundle(model_name: str) -> Dict[str, object]:
    path = os.path.join(MODEL_DIR, f"{model_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_model(rows: Sequence[Row]) -> None:
    bundle = train_strategy_models(rows)
    metrics = evaluate_bundle(bundle, bundle["x_test_raw"], bundle["y_test_ret"], bundle["y_test_dir"])
    print("=== Strategy Feature Set ===")
    print(", ".join(metrics["features"]))
    print("\n=== Linear Regression (predict next return) ===")
    print(f"Test MSE: {metrics['mse']:.8f}")
    print(f"Test MAE: {metrics['mae']:.8f}")
    print("\n=== Logistic Regression (predict P(up)) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")


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
        if hi - lo < 1e-12:
            stoch_rsi.append(50.0)
        else:
            stoch_rsi.append(100.0 * ((rsi[i] - lo) / (hi - lo)))

    ema_12 = ema(closes, 12)
    ema_26 = ema(closes, 26)
    macd = [a - b for a, b in zip(ema_12, ema_26)]
    signal = ema(macd, 9)
    macd_hist = [m - s for m, s in zip(macd, signal)]
    macd_hist_delta = [0.0] + [macd_hist[i] - macd_hist[i - 1] for i in range(1, len(macd_hist))]

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
            }
        )
    if not rows:
        raise ValueError("Could not build feature rows from downloaded candles.")
    return rows


def fetch_yahoo_rows(ticker: str, interval: str, row_count: int) -> List[Row]:
    import yfinance as yf

    interval_periods = {
        "1d": ["1y", "5y", "10y", "max"],
        "1h": ["730d"],
        "15m": ["60d"],
        "5m": ["60d"],
    }
    if interval not in interval_periods:
        raise ValueError("Interval must be one of: 1d, 1h, 15m, 5m")
    if row_count < 50:
        raise ValueError("Please request at least 50 rows.")

    ticker_obj = yf.Ticker(ticker)
    rows: List[Row] = []
    last_non_empty_count = 0

    for period in interval_periods[interval]:
        history = ticker_obj.history(period=period, interval=interval, auto_adjust=False)
        if history.empty:
            continue

        highs = [float(v) for v in history["High"].tolist()]
        lows = [float(v) for v in history["Low"].tolist()]
        closes = [float(v) for v in history["Close"].tolist()]
        rows = compute_strategy_rows_from_prices(highs=highs, lows=lows, closes=closes)
        last_non_empty_count = max(last_non_empty_count, len(rows))

        if len(rows) >= row_count:
            return rows[-row_count:]

    if last_non_empty_count == 0:
        raise ValueError(f"No Yahoo Finance data returned for ticker '{ticker}'.")

    raise ValueError(
        f"Requested {row_count} rows, but only {last_non_empty_count} are available for {ticker} ({interval}). "
        "Try requesting fewer rows or using a larger timeframe."
    )


def run_model_metrics(rows: Sequence[Row]) -> Dict[str, object]:
    bundle = train_strategy_models(rows)
    metrics = evaluate_bundle(bundle, bundle["x_test_raw"], bundle["y_test_ret"], bundle["y_test_dir"])
    metrics["train_size"] = bundle["train_size"]
    return metrics


def create_app() -> "Flask":
    from flask import Flask, request

    app = Flask(__name__)

    @app.route("/", methods=["GET", "POST"])
    def index() -> str:
        result_html = ""
        error_html = ""
        ticker = request.form.get("ticker", "AAPL").upper().strip()
        interval = request.form.get("interval", "1d")
        rows = request.form.get("rows", "250")
        model_name = request.form.get("model_name", "").strip()
        selected_model = request.form.get("selected_model", "__new__")
        saved_models = list_saved_models()

        if request.method == "POST":
            try:
                row_count = int(rows)
                dataset = fetch_yahoo_rows(ticker=ticker, interval=interval, row_count=row_count)
                if selected_model != "__new__":
                    loaded = load_model_bundle(selected_model)
                    features = build_default_strategy_features()
                    x_raw = features.transform(dataset)
                    y_ret = [r["return_next"] for r in dataset]
                    y_dir = [1 if r > 0 else 0 for r in y_ret]
                    metrics = evaluate_bundle(loaded, x_raw, y_ret, y_dir)
                    metrics["train_size"] = "saved-model"
                    metrics["loaded_model"] = selected_model
                else:
                    bundle = train_strategy_models(dataset)
                    metrics = evaluate_bundle(bundle, bundle["x_test_raw"], bundle["y_test_ret"], bundle["y_test_dir"])
                    metrics["train_size"] = bundle["train_size"]
                    if model_name:
                        save_model_bundle(model_name, bundle)
                        metrics["saved_model"] = model_name
                        saved_models = list_saved_models()

                preview_rows = "".join(
                    f"<tr><td>{idx + 1}</td><td>{p['expected_return']:+.4%}</td><td>{p['p_up']:.2%}</td><td>{p['actual_return']:+.4%}</td></tr>"
                    for idx, p in enumerate(metrics["preview"])
                )
                model_msg = ""
                if "saved_model" in metrics:
                    model_msg += f"<p><strong>Saved model:</strong> {metrics['saved_model']}</p>"
                if "loaded_model" in metrics:
                    model_msg += f"<p><strong>Loaded model:</strong> {metrics['loaded_model']}</p>"
                result_html = f"""
                <h2>Results for {ticker} ({interval})</h2>
                <p>Rows used: {row_count} (train={metrics['train_size']}, test={metrics['test_size']})</p>
                {model_msg}
                <p><strong>Linear Test MSE:</strong> {metrics['mse']:.8f}</p>
                <p><strong>Linear Test MAE:</strong> {metrics['mae']:.8f}</p>
                <p><strong>Logistic Accuracy:</strong> {metrics['accuracy']:.4f}</p>
                <p><strong>Precision:</strong> {metrics['precision']:.4f} | <strong>Recall:</strong> {metrics['recall']:.4f} | <strong>F1:</strong> {metrics['f1']:.4f}</p>
                <p><strong>Confusion Matrix:</strong> TP={metrics['tp']}, FP={metrics['fp']}, TN={metrics['tn']}, FN={metrics['fn']}</p>
                <h3>Feature Set</h3>
                <p>{', '.join(metrics['features'])}</p>
                <h3>Example Predictions</h3>
                <table border="1" cellpadding="6">
                  <tr><th>Row</th><th>Expected Return</th><th>P(Up)</th><th>Actual Return</th></tr>
                  {preview_rows}
                </table>
                """
            except Exception as exc:
                error_html = f"<p style='color:red;'><strong>Error:</strong> {exc}</p>"

        return f"""
        <html>
          <head><title>Quant Model Trainer</title></head>
          <body style="font-family: Arial, sans-serif; margin: 2rem;">
            <h1>Train Quant Model from Yahoo Data</h1>
            <form method="post">
              <label>Ticker:
                <input type="text" name="ticker" value="{ticker}" required />
              </label>
              <br/><br/>
              <label>Interval:
                <select name="interval">
                  <option value="1d" {"selected" if interval == "1d" else ""}>Daily</option>
                  <option value="1h" {"selected" if interval == "1h" else ""}>Hourly</option>
                  <option value="15m" {"selected" if interval == "15m" else ""}>15 min</option>
                  <option value="5m" {"selected" if interval == "5m" else ""}>5 min</option>
                </select>
              </label>
              <br/><br/>
              <label>Rows:
                <input type="number" min="50" name="rows" value="{rows}" required />
              </label>
              <br/><br/>
              <label>Saved Model:
                <select name="selected_model">
                  <option value="__new__">Train new model</option>
                  {"".join(f'<option value="{name}" {"selected" if selected_model == name else ""}>{name}</option>' for name in saved_models)}
                </select>
              </label>
              <br/><br/>
              <label>New Model Name (optional):
                <input type="text" name="model_name" value="{model_name}" placeholder="momentum_v1" />
              </label>
              <br/><br/>
              <button type="submit">Download + Train</button>
            </form>
            {error_html}
            {result_html}
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
