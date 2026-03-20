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
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Sequence, Tuple

if TYPE_CHECKING:
    from flask import Flask


Row = Dict[str, float]
FeatureFn = Callable[[Row], float]


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


def load_csv(path: str) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for record in reader:
            row: Row = {
                "stoch_rsi": float(record["stoch_rsi"]),
                "macd": float(record["macd"]),
                "macd_hist": float(record["macd_hist"]),
                "fvg_green_size": float(record["fvg_green_size"]),
                "fvg_red_above_green": float(record["fvg_red_above_green"]),
                "first_green_fvg_dip": float(record["first_green_fvg_dip"]),
                "return_next": float(record["return_next"]),
            }
            rows.append(row)
    return rows


def synthetic_data(n: int = 1200, seed: int = 42) -> List[Row]:
    random.seed(seed)
    rows: List[Row] = []
    for _ in range(n):
        stoch_rsi = random.uniform(0, 100)
        macd = random.uniform(-2.0, 2.0)
        macd_hist = random.uniform(-1.0, 1.0)
        fvg_green_size = max(0.0, random.gauss(0.5, 0.6))
        fvg_red_above_green = 1.0 if random.random() < 0.35 else 0.0
        first_green_fvg_dip = 1.0 if random.random() < 0.25 else 0.0

        stoch_extreme = 1.0 if (stoch_rsi > 80 or stoch_rsi < 20) else 0.0
        macd_weakening = 1.0 if (macd_hist < 0 and macd > 0) or (macd_hist < -0.2) else 0.0
        macd_improving_red = 1.0 if (macd < 0 and macd_hist > 0) else 0.0

        alpha = (
            0.0035 * stoch_extreme
            - 0.0025 * macd_weakening
            + 0.0017 * macd_improving_red
            + 0.0030 * first_green_fvg_dip * min(fvg_green_size, 3.0)
            - 0.0020 * fvg_red_above_green * min(fvg_green_size, 3.0)
            + random.gauss(0, 0.0018)
        )

        rows.append(
            {
                "stoch_rsi": stoch_rsi,
                "macd": macd,
                "macd_hist": macd_hist,
                "fvg_green_size": fvg_green_size,
                "fvg_red_above_green": fvg_red_above_green,
                "first_green_fvg_dip": first_green_fvg_dip,
                "return_next": alpha,
            }
        )
    return rows


def build_default_strategy_features() -> StrategyFeatureBuilder:
    builder = StrategyFeatureBuilder()

    # 1) Stoch RSI extremes imply setup urgency / likely expansion move.
    builder.add("stoch_extreme", lambda r: 1.0 if (r["stoch_rsi"] > 80 or r["stoch_rsi"] < 20) else 0.0)

    # 2) MACD states: weakening green is a sell pressure, improving red is acceptable.
    builder.add("macd_green", lambda r: 1.0 if r["macd"] > 0 else 0.0)
    builder.add("macd_weakening", lambda r: 1.0 if (r["macd"] > 0 and r["macd_hist"] < 0) else 0.0)
    builder.add("macd_red_improving", lambda r: 1.0 if (r["macd"] < 0 and r["macd_hist"] > 0) else 0.0)

    # 3) FVG context and size weighting.
    builder.add("first_green_fvg_dip", lambda r: r["first_green_fvg_dip"])
    builder.add("fvg_size", lambda r: min(r["fvg_green_size"], 3.0))
    builder.add(
        "fvg_bull_signal",
        lambda r: r["first_green_fvg_dip"] * min(r["fvg_green_size"], 3.0),
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


def run_model(rows: Sequence[Row]) -> None:
    train_rows, test_rows = train_test_split(rows)

    features = build_default_strategy_features()
    x_train_raw = features.transform(train_rows)
    x_test_raw = features.transform(test_rows)

    x_train, means, stds = standardize_fit(x_train_raw)
    x_test = standardize_apply(x_test_raw, means, stds)

    y_train_ret = [r["return_next"] for r in train_rows]
    y_test_ret = [r["return_next"] for r in test_rows]

    lin = LinearRegressionGD(learning_rate=0.03, epochs=3000)
    lin.fit(x_train, y_train_ret)
    ret_pred = lin.predict(x_test)
    ret_mse = mse(y_test_ret, ret_pred)

    y_train_dir = [1 if r > 0 else 0 for r in y_train_ret]
    y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]

    logit = LogisticRegressionGD(learning_rate=0.05, epochs=2500)
    logit.fit(x_train, y_train_dir)
    up_prob = logit.predict_proba(x_test)
    acc = accuracy(y_test_dir, up_prob)

    print("=== Strategy Feature Set ===")
    print(", ".join(features.names()))

    print("\n=== Linear Regression (predict next return) ===")
    print(f"Test MSE: {ret_mse:.8f}")
    print("Weights:")
    for name, w in zip(features.names(), lin.weights):
        print(f"  {name:>22}: {w:+.6f}")
    print(f"  {'bias':>22}: {lin.bias:+.6f}")

    print("\n=== Logistic Regression (predict P(up)) ===")
    print(f"Test Accuracy: {acc:.4f}")
    print("Weights:")
    for name, w in zip(features.names(), logit.weights):
        print(f"  {name:>22}: {w:+.6f}")
    print(f"  {'bias':>22}: {logit.bias:+.6f}")

    print("\n=== Example Predictions (first 5 test rows) ===")
    for i in range(min(5, len(x_test))):
        print(
            f"Row {i+1}: expected_return={ret_pred[i]:+.4%}, "
            f"p_up={up_prob[i]:.2%}, actual_return={y_test_ret[i]:+.4%}"
        )


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

    rows: List[Row] = []
    for i in range(2, n - 1):
        bullish_gap = max(0.0, lows[i] - highs[i - 2])
        bearish_gap = max(0.0, lows[i - 2] - highs[i])
        rows.append(
            {
                "stoch_rsi": stoch_rsi[i],
                "macd": macd[i],
                "macd_hist": macd_hist[i],
                "fvg_green_size": bullish_gap,
                "fvg_red_above_green": 1.0 if bearish_gap > 0 else 0.0,
                "first_green_fvg_dip": 1.0 if (bullish_gap > 0 and lows[i + 1] <= highs[i - 2]) else 0.0,
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
    train_rows, test_rows = train_test_split(rows)
    features = build_default_strategy_features()
    x_train_raw = features.transform(train_rows)
    x_test_raw = features.transform(test_rows)
    x_train, means, stds = standardize_fit(x_train_raw)
    x_test = standardize_apply(x_test_raw, means, stds)

    y_train_ret = [r["return_next"] for r in train_rows]
    y_test_ret = [r["return_next"] for r in test_rows]
    lin = LinearRegressionGD(learning_rate=0.03, epochs=3000)
    lin.fit(x_train, y_train_ret)
    ret_pred = lin.predict(x_test)
    ret_mse = mse(y_test_ret, ret_pred)

    y_train_dir = [1 if r > 0 else 0 for r in y_train_ret]
    y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]
    logit = LogisticRegressionGD(learning_rate=0.05, epochs=2500)
    logit.fit(x_train, y_train_dir)
    up_prob = logit.predict_proba(x_test)
    acc = accuracy(y_test_dir, up_prob)
    preview = []
    for i in range(min(5, len(x_test))):
        preview.append(
            {
                "expected_return": ret_pred[i],
                "p_up": up_prob[i],
                "actual_return": y_test_ret[i],
            }
        )
    return {
        "features": features.names(),
        "mse": ret_mse,
        "accuracy": acc,
        "lin_weights": list(zip(features.names(), lin.weights)),
        "lin_bias": lin.bias,
        "logit_weights": list(zip(features.names(), logit.weights)),
        "logit_bias": logit.bias,
        "preview": preview,
        "train_size": len(train_rows),
        "test_size": len(test_rows),
    }


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

        if request.method == "POST":
            try:
                row_count = int(rows)
                dataset = fetch_yahoo_rows(ticker=ticker, interval=interval, row_count=row_count)
                metrics = run_model_metrics(dataset)

                preview_rows = "".join(
                    f"<tr><td>{idx + 1}</td><td>{p['expected_return']:+.4%}</td><td>{p['p_up']:.2%}</td><td>{p['actual_return']:+.4%}</td></tr>"
                    for idx, p in enumerate(metrics["preview"])
                )
                result_html = f"""
                <h2>Results for {ticker} ({interval})</h2>
                <p>Rows used: {row_count} (train={metrics['train_size']}, test={metrics['test_size']})</p>
                <p><strong>Linear Test MSE:</strong> {metrics['mse']:.8f}</p>
                <p><strong>Logistic Test Accuracy:</strong> {metrics['accuracy']:.4f}</p>
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
        help="Optional CSV path with columns: stoch_rsi, macd, macd_hist, fvg_green_size, fvg_red_above_green, first_green_fvg_dip, return_next",
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
