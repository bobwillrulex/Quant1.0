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
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Sequence, Tuple

if TYPE_CHECKING:
    from flask import Flask


Row = Dict[str, float]
FeatureFn = Callable[[Row], float]
MODEL_DIR = "saved_models"
MODEL_CONFIGS_FILE = "model_configs.json"


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
    def __init__(self, learning_rate: float = 0.01, epochs: int = 800) -> None:
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
    def __init__(self, learning_rate: float = 0.05, epochs: int = 700) -> None:
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


SplitStyle = Literal["shuffled", "chronological"]


def train_test_split(
    rows: Sequence[Row],
    test_ratio: float = 0.25,
    split_style: SplitStyle = "shuffled",
) -> Tuple[List[Row], List[Row]]:
    data = list(rows)
    if split_style == "shuffled":
        random.shuffle(data)
    elif split_style != "chronological":
        raise ValueError("split_style must be either 'shuffled' or 'chronological'.")
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


def stddev(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def compounded_return(returns: Sequence[float]) -> float:
    equity = 1.0
    for ret in returns:
        equity *= 1.0 + ret
    return equity - 1.0


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


def model_configs_path() -> str:
    ensure_model_dir()
    return os.path.join(MODEL_DIR, MODEL_CONFIGS_FILE)


def load_model_configs() -> Dict[str, Dict[str, object]]:
    path = model_configs_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    return {}


def save_model_configs(configs: Dict[str, Dict[str, object]]) -> None:
    with open(model_configs_path(), "w", encoding="utf-8") as f:
        json.dump(configs, f, indent=2)


def default_model_config() -> Dict[str, object]:
    return {
        "ticker": "AAPL",
        "interval": "1d",
        "rows": 250,
        "include_in_run_all": True,
        "buy_threshold": 0.6,
        "sell_threshold": 0.4,
    }


def get_model_config(model_name: str, configs: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    merged = dict(default_model_config())
    stored = configs.get(model_name, {})
    if isinstance(stored, dict):
        merged.update(stored)
    return merged


def parse_thresholds(buy_raw: str, sell_raw: str, *, default_buy: float = 0.6, default_sell: float = 0.4) -> Tuple[float, float]:
    buy_text = buy_raw.strip()
    sell_text = sell_raw.strip()
    buy_threshold = default_buy if buy_text == "" else float(buy_text)
    sell_threshold = default_sell if sell_text == "" else float(sell_text)
    if not (0.0 <= sell_threshold < buy_threshold <= 1.0):
        raise ValueError("Thresholds must satisfy: 0.0 <= sell < buy <= 1.0.")
    return buy_threshold, sell_threshold


def train_strategy_models(rows: Sequence[Row], split_style: SplitStyle = "shuffled") -> Dict[str, object]:
    train_rows, test_rows = train_test_split(rows, split_style=split_style)
    features = build_default_strategy_features()
    x_train_raw = features.transform(train_rows)
    x_test_raw = features.transform(test_rows)
    x_train, means, stds = standardize_fit(x_train_raw)
    x_test = standardize_apply(x_test_raw, means, stds)

    y_train_ret = [r["return_next"] for r in train_rows]
    y_test_ret = [r["return_next"] for r in test_rows]
    y_train_dir = [1 if r > 0 else 0 for r in y_train_ret]
    y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]

    lin = LinearRegressionGD(learning_rate=0.03, epochs=800)
    lin.fit(x_train, y_train_ret)

    logit = LogisticRegressionGD(learning_rate=0.05, epochs=700)
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
        "split_style": split_style,
        "x_test_raw": x_test_raw,
        "y_test_ret": y_test_ret,
        "y_test_dir": y_test_dir,
    }


def evaluate_bundle(
    bundle: Dict[str, object],
    x_test_raw: Sequence[Sequence[float]],
    y_test_ret: Sequence[float],
    y_test_dir: Sequence[int],
    eval_rows: Sequence[Row] | None = None,
    split_style: SplitStyle = "shuffled",
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
) -> Dict[str, object]:
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
    baseline_up_accuracy = sum(y_test_dir) / max(1, len(y_test_dir))
    baseline_zero = [0.0] * len(y_test_ret)
    calibration = calibration_buckets(y_test_dir, up_prob)
    confidence_edges = confidence_edge_analysis(y_test_dir, up_prob)
    strategy = strategy_metrics(
        y_test_ret,
        up_prob,
        long_threshold=buy_threshold,
        short_threshold=sell_threshold,
        trade_cost=0.0005,
        buy_hold_returns=y_test_ret,
    )
    pnl_by_signal = pnl_signal_strength_breakdown(y_test_ret, up_prob, trade_cost=0.0005)
    pnl_by_regime = pnl_market_regime_breakdown(y_test_ret, up_prob, trade_cost=0.0005)
    walk_forward = walk_forward_validation_rows(rows=eval_rows, max_windows=4) if eval_rows else []
    ablation = feature_ablation_analysis(eval_rows, bundle["feature_names"], split_style=split_style) if eval_rows else []
    errors = error_analysis(y_test_ret, up_prob, ret_pred, top_n=5)

    preview = []
    for i in range(min(5, len(x_test))):
        preview.append({"expected_return": ret_pred[i], "p_up": up_prob[i], "actual_return": y_test_ret[i]})

    return {
        "features": bundle["feature_names"],
        "mse": mse(y_test_ret, ret_pred),
        "mae": mae(y_test_ret, ret_pred),
        "accuracy": cls["accuracy"],
        "baseline_always_up_accuracy": baseline_up_accuracy,
        "accuracy_vs_baseline": cls["accuracy"] - baseline_up_accuracy,
        "precision": cls["precision"],
        "recall": cls["recall"],
        "f1": cls["f1"],
        "baseline_zero_mse": mse(y_test_ret, baseline_zero),
        "baseline_zero_mae": mae(y_test_ret, baseline_zero),
        "mse_vs_zero_baseline": mse(y_test_ret, baseline_zero) - mse(y_test_ret, ret_pred),
        "mae_vs_zero_baseline": mae(y_test_ret, baseline_zero) - mae(y_test_ret, ret_pred),
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
        "split_style": split_style,
        "calibration": calibration,
        "confidence_edge": confidence_edges,
        "strategy": strategy,
        "pnl_by_signal_strength": pnl_by_signal,
        "pnl_by_regime": pnl_by_regime,
        "walk_forward": walk_forward,
        "feature_ablation": ablation,
        "error_analysis": errors,
    }


def calibration_buckets(y_true: Sequence[int], y_prob: Sequence[float], bucket_size: float = 0.05) -> List[Dict[str, float]]:
    bins: Dict[Tuple[float, float], List[int]] = {}
    prob_bins: Dict[Tuple[float, float], List[float]] = {}
    steps = int(1.0 / bucket_size)
    for i in range(steps):
        lo = i * bucket_size
        hi = (i + 1) * bucket_size
        bins[(lo, hi)] = []
        prob_bins[(lo, hi)] = []
    for actual, p in zip(y_true, y_prob):
        idx = min(steps - 1, int(p / bucket_size))
        lo = idx * bucket_size
        hi = (idx + 1) * bucket_size
        bins[(lo, hi)].append(int(actual))
        prob_bins[(lo, hi)].append(float(p))
    result: List[Dict[str, float]] = []
    for (lo, hi), values in bins.items():
        if not values:
            continue
        probs = prob_bins[(lo, hi)]
        result.append(
            {
                "bucket_low": lo,
                "bucket_high": hi,
                "count": float(len(values)),
                "predicted_mean": sum(probs) / len(probs),
                "actual_win_rate": sum(values) / len(values),
            }
        )
    return result


def confidence_edge_analysis(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for threshold in [0.6, 0.7]:
        preds = [(a, p) for a, p in zip(y_true, y_prob) if p > threshold]
        if preds:
            acc = sum(1 for a, _ in preds if a == 1) / len(preds)
            out[f"p_gt_{threshold:.1f}"] = {"count": float(len(preds)), "accuracy": acc}
        else:
            out[f"p_gt_{threshold:.1f}"] = {"count": 0.0, "accuracy": 0.0}
    return out


def strategy_metrics(
    returns: Sequence[float],
    probs: Sequence[float],
    long_threshold: float = 0.6,
    short_threshold: float = 0.4,
    trade_cost: float = 0.0005,
    buy_hold_returns: Sequence[float] | None = None,
) -> Dict[str, float]:
    positions: List[int] = []
    for p in probs:
        if p > long_threshold:
            positions.append(1)
        elif p < short_threshold:
            positions.append(-1)
        else:
            positions.append(0)

    pnl: List[float] = []
    prev_pos = 0
    wins = 0
    trades = 0
    for pos, ret in zip(positions, returns):
        turnover = abs(pos - prev_pos)
        cost = turnover * trade_cost
        day_pnl = pos * ret - cost
        pnl.append(day_pnl)
        if pos != 0:
            trades += 1
            if day_pnl > 0:
                wins += 1
        prev_pos = pos
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for r in pnl:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

    sharpe = 0.0
    sd = stddev(pnl)
    if sd > 1e-12:
        sharpe = (sum(pnl) / len(pnl)) / sd * math.sqrt(252.0)
    buy_hold_source = buy_hold_returns if buy_hold_returns is not None else returns
    buy_hold_total_return = compounded_return(buy_hold_source)
    return {
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "trade_cost": trade_cost,
        "total_return": equity - 1.0,
        "buy_hold_total_return": buy_hold_total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": (wins / trades) if trades else 0.0,
        "trade_count": float(trades),
    }


def pnl_signal_strength_breakdown(returns: Sequence[float], probs: Sequence[float], trade_cost: float = 0.0005) -> List[Dict[str, float]]:
    buckets = {
        "weak_0.50_0.55": (0.50, 0.55),
        "medium_0.55_0.65": (0.55, 0.65),
        "strong_0.65_1.00": (0.65, 1.00),
    }
    out: List[Dict[str, float]] = []
    for name, (lo, hi) in buckets.items():
        pnl = []
        for p, r in zip(probs, returns):
            confidence = max(p, 1.0 - p)
            if lo <= confidence < hi:
                pos = 1 if p >= 0.5 else -1
                pnl.append(pos * r - trade_cost)
        if pnl:
            out.append({"bucket": name, "count": float(len(pnl)), "avg_pnl": sum(pnl) / len(pnl), "total_pnl": sum(pnl)})
    return out


def pnl_market_regime_breakdown(returns: Sequence[float], probs: Sequence[float], trade_cost: float = 0.0005) -> List[Dict[str, float]]:
    out = {"trending": [], "sideways": [], "high_volatility": []}
    window = 20
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        r_win = returns[start : i + 1]
        trend = abs(sum(r_win) / max(1, len(r_win)))
        vol = stddev(r_win)
        p = probs[i]
        pos = 1 if p > 0.6 else (-1 if p < 0.4 else 0)
        pnl = pos * returns[i] - (trade_cost if pos != 0 else 0.0)
        if vol > 0.02:
            out["high_volatility"].append(pnl)
        elif trend > 0.002:
            out["trending"].append(pnl)
        else:
            out["sideways"].append(pnl)
    result: List[Dict[str, float]] = []
    for regime, pnl_list in out.items():
        if pnl_list:
            result.append(
                {
                    "regime": regime,
                    "count": float(len(pnl_list)),
                    "avg_pnl": sum(pnl_list) / len(pnl_list),
                    "total_pnl": sum(pnl_list),
                }
            )
    return result


def walk_forward_validation_rows(rows: Sequence[Row], max_windows: int = 4) -> List[Dict[str, float]]:
    if len(rows) < 120:
        return []
    features = build_default_strategy_features()
    chunk = len(rows) // (max_windows + 2)
    results: List[Dict[str, float]] = []
    for idx in range(max_windows):
        train_end = chunk * (idx + 2)
        test_end = min(len(rows), train_end + chunk)
        train_rows = rows[:train_end]
        test_rows = rows[train_end:test_end]
        if len(test_rows) < 20:
            continue
        x_train_raw = features.transform(train_rows)
        x_test_raw = features.transform(test_rows)
        x_train, means, stds = standardize_fit(x_train_raw)
        x_test = standardize_apply(x_test_raw, means, stds)
        y_train_ret = [r["return_next"] for r in train_rows]
        y_test_ret = [r["return_next"] for r in test_rows]
        y_train_dir = [1 if r > 0 else 0 for r in y_train_ret]
        y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]

        lin = LinearRegressionGD(learning_rate=0.03, epochs=800)
        lin.fit(x_train, y_train_ret)
        logit = LogisticRegressionGD(learning_rate=0.05, epochs=700)
        logit.fit(x_train, y_train_dir)
        ret_pred = lin.predict(x_test)
        up_prob = logit.predict_proba(x_test)
        results.append(
            {
                "window": float(idx + 1),
                "train_size": float(len(train_rows)),
                "test_size": float(len(test_rows)),
                "accuracy": accuracy(y_test_dir, up_prob),
                "mse": mse(y_test_ret, ret_pred),
            }
        )
    return results


def feature_ablation_analysis(
    rows: Sequence[Row],
    feature_names: Sequence[str],
    split_style: SplitStyle = "shuffled",
) -> List[Dict[str, float]]:
    if len(rows) < 100:
        return []
    if len(rows) > 600:
        rows = list(rows)[-600:]
    train_rows, test_rows = train_test_split(rows, split_style=split_style)
    features = build_default_strategy_features()
    x_train_raw_full = features.transform(train_rows)
    x_test_raw_full = features.transform(test_rows)
    x_train_full, means_full, stds_full = standardize_fit(x_train_raw_full)
    x_test_full = standardize_apply(x_test_raw_full, means_full, stds_full)
    y_train_ret = [r["return_next"] for r in train_rows]
    y_test_ret = [r["return_next"] for r in test_rows]
    y_train_dir = [1 if r > 0 else 0 for r in y_train_ret]
    y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]

    lin_full = LinearRegressionGD(learning_rate=0.03, epochs=800)
    lin_full.fit(x_train_full, y_train_ret)
    logit_full = LogisticRegressionGD(learning_rate=0.05, epochs=700)
    logit_full.fit(x_train_full, y_train_dir)
    full_ret_pred = lin_full.predict(x_test_full)
    full_prob = logit_full.predict_proba(x_test_full)
    full_accuracy = accuracy(y_test_dir, full_prob)
    full_mse = mse(y_test_ret, full_ret_pred)

    out: List[Dict[str, float]] = []
    for removed in feature_names:
        builder = StrategyFeatureBuilder()
        default_builder = build_default_strategy_features()
        for feature in default_builder._features:
            if feature.name != removed:
                builder.add(feature.name, feature.fn)

        train_rows, test_rows = train_test_split(rows, split_style=split_style)
        x_train_raw = builder.transform(train_rows)
        x_test_raw = builder.transform(test_rows)
        x_train, means, stds = standardize_fit(x_train_raw)
        x_test = standardize_apply(x_test_raw, means, stds)
        y_train_ret_loop = [r["return_next"] for r in train_rows]
        y_test_ret_loop = [r["return_next"] for r in test_rows]
        y_train_dir_loop = [1 if r > 0 else 0 for r in y_train_ret_loop]
        y_test_dir_loop = [1 if r > 0 else 0 for r in y_test_ret_loop]
        lin = LinearRegressionGD(learning_rate=0.03, epochs=800)
        lin.fit(x_train, y_train_ret_loop)
        logit = LogisticRegressionGD(learning_rate=0.05, epochs=700)
        logit.fit(x_train, y_train_dir_loop)

        ret_pred = lin.predict(x_test)
        up_prob = logit.predict_proba(x_test)
        out.append(
            {
                "removed_feature": removed,
                "accuracy_delta": accuracy(y_test_dir_loop, up_prob) - full_accuracy,
                "mse_delta": mse(y_test_ret_loop, ret_pred) - full_mse,
            }
        )
    return out


def error_analysis(y_test_ret: Sequence[float], up_prob: Sequence[float], ret_pred: Sequence[float], top_n: int = 5) -> Dict[str, List[Dict[str, float]]]:
    largest_errors = sorted(
        [{"index": float(i), "abs_error": abs(y_test_ret[i] - ret_pred[i]), "actual_return": y_test_ret[i], "pred_return": ret_pred[i]} for i in range(len(y_test_ret))],
        key=lambda x: x["abs_error"],
        reverse=True,
    )[:top_n]
    high_conf_wrong = []
    for i, (ret, p) in enumerate(zip(y_test_ret, up_prob)):
        actual_up = 1 if ret > 0 else 0
        pred_up = 1 if p >= 0.5 else 0
        confidence = max(p, 1.0 - p)
        if actual_up != pred_up and confidence >= 0.7:
            high_conf_wrong.append({"index": float(i), "p_up": p, "actual_return": ret, "confidence": confidence})
    high_conf_wrong = sorted(high_conf_wrong, key=lambda x: x["confidence"], reverse=True)[:top_n]
    return {"largest_return_errors": largest_errors, "high_confidence_wrong_calls": high_conf_wrong}


def save_model_bundle(model_name: str, bundle: Dict[str, object]) -> str:
    ensure_model_dir()
    safe_name = sanitize_model_name(model_name)
    if not safe_name:
        raise ValueError("Model name must include letters or numbers.")
    path = os.path.join(MODEL_DIR, f"{safe_name}.json")
    payload = {k: bundle[k] for k in ["feature_names", "means", "stds", "lin_weights", "lin_bias", "logit_weights", "logit_bias"]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def sanitize_model_name(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in model_name).strip("_")


def list_saved_models() -> List[str]:
    ensure_model_dir()
    return sorted([f[:-5] for f in os.listdir(MODEL_DIR) if f.endswith(".json")])


def load_model_bundle(model_name: str) -> Dict[str, object]:
    path = os.path.join(MODEL_DIR, f"{model_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_model(rows: Sequence[Row]) -> None:
    bundle = train_strategy_models(rows)
    metrics = evaluate_bundle(
        bundle,
        bundle["x_test_raw"],
        bundle["y_test_ret"],
        bundle["y_test_dir"],
        eval_rows=rows,
        split_style=bundle["split_style"],
    )
    print("=== Strategy Feature Set ===")
    print(", ".join(metrics["features"]))
    print("\n=== Linear Regression (predict next return) ===")
    print(f"Test MSE: {metrics['mse']:.8f}")
    print(f"Test MAE: {metrics['mae']:.8f}")
    print("\n=== Logistic Regression (predict P(up)) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"Always-UP baseline accuracy: {metrics['baseline_always_up_accuracy']:.4f} (edge: {metrics['accuracy_vs_baseline']:+.4f})")
    print(f"Zero-return baseline MSE/MAE: {metrics['baseline_zero_mse']:.8f} / {metrics['baseline_zero_mae']:.8f}")
    print(f"Model improvement vs zero baseline (MSE/MAE): {metrics['mse_vs_zero_baseline']:+.8f} / {metrics['mae_vs_zero_baseline']:+.8f}")
    strat = metrics["strategy"]
    print("\n=== Decision Strategy (long > 0.60, short < 0.40) ===")
    print(
        f"Total Return: {strat['total_return']:+.2%}, Sharpe: {strat['sharpe']:.3f}, "
        f"Max Drawdown: {strat['max_drawdown']:.2%}, Win Rate: {strat['win_rate']:.2%}, Trades: {int(strat['trade_count'])}"
    )
    print(f"Buy & Hold Return (test rows): {strat['buy_hold_total_return']:+.2%}")


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
    metrics = evaluate_bundle(
        bundle,
        bundle["x_test_raw"],
        bundle["y_test_ret"],
        bundle["y_test_dir"],
        eval_rows=rows,
        split_style=bundle["split_style"],
    )
    metrics["train_size"] = bundle["train_size"]
    return metrics


def predict_signal(
    bundle: Dict[str, object],
    row: Row,
    *,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
) -> Dict[str, float | str]:
    feature_builder = build_default_strategy_features()
    row_features = feature_builder.transform([row])
    if len(row_features[0]) != len(bundle["feature_names"]):
        raise ValueError("Saved model feature size does not match current strategy feature set.")
    x_scaled = standardize_apply(row_features, bundle["means"], bundle["stds"])[0]
    expected_return = sum(w * v for w, v in zip(bundle["lin_weights"], x_scaled)) + bundle["lin_bias"]
    p_up = sigmoid(sum(w * v for w, v in zip(bundle["logit_weights"], x_scaled)) + bundle["logit_bias"])
    action = "HOLD"
    if p_up > buy_threshold:
        action = "BUY"
    elif p_up < sell_threshold:
        action = "SELL"
    return {"expected_return": expected_return, "p_up": p_up, "action": action}


def build_run_all_rows(saved_models: Sequence[str], model_configs: Dict[str, Dict[str, object]]) -> str:
    run_all_rows = ""
    for model_name in saved_models:
        cfg = get_model_config(model_name, model_configs)
        if not cfg.get("include_in_run_all", True):
            continue
        try:
            dataset = fetch_yahoo_rows(
                ticker=str(cfg.get("ticker", "AAPL")),
                interval=str(cfg.get("interval", "1d")),
                row_count=int(cfg.get("rows", 250)),
            )
            latest_row = dataset[-1]
            bundle = load_model_bundle(model_name)
            buy_threshold = float(cfg.get("buy_threshold", 0.6))
            sell_threshold = float(cfg.get("sell_threshold", 0.4))
            prediction = predict_signal(
                bundle,
                latest_row,
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
            )
            run_all_rows += (
                "<tr>"
                f"<td>{model_name}</td>"
                f"<td>{cfg.get('ticker')}</td>"
                f"<td>{cfg.get('interval')}</td>"
                f"<td>{int(cfg.get('rows', 250))}</td>"
                f"<td>{buy_threshold:.2f} / {sell_threshold:.2f}</td>"
                f"<td>{prediction['expected_return']:+.4%}</td>"
                f"<td>{prediction['p_up']:.2%}</td>"
                f"<td><strong>{prediction['action']}</strong></td>"
                "</tr>"
            )
        except Exception as exc:
            run_all_rows += (
                "<tr>"
                f"<td>{model_name}</td>"
                f"<td>{cfg.get('ticker')}</td>"
                f"<td>{cfg.get('interval')}</td>"
                f"<td>{int(cfg.get('rows', 250))}</td>"
                f"<td>{float(cfg.get('buy_threshold', 0.6)):.2f} / {float(cfg.get('sell_threshold', 0.4)):.2f}</td>"
                "<td colspan='3' style='color:#ff7b7b;'>"
                f"Run failed: {exc}"
                "</td>"
                "</tr>"
            )
    return run_all_rows


def create_app() -> "Flask":
    from flask import Flask, redirect, request, url_for

    app = Flask(__name__)

    @app.route("/manage-models", methods=["GET", "POST"])
    def manage_models() -> str:
        message_html = ""
        error_html = ""
        model_configs = load_model_configs()
        saved_models = list_saved_models()
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
                    }
                    save_model_configs(model_configs)
                    message_html = f"<p style='color:#7bd88f;'><strong>Saved settings for:</strong> {model_name}</p>"
                elif action == "rename_model":
                    new_name = sanitize_model_name(request.form.get("new_name", "").strip())
                    if model_name not in saved_models:
                        raise ValueError("Model not found.")
                    if not new_name:
                        raise ValueError("New model name cannot be empty.")
                    if new_name in saved_models:
                        raise ValueError("A model with that name already exists.")
                    old_path = os.path.join(MODEL_DIR, f"{model_name}.json")
                    new_path = os.path.join(MODEL_DIR, f"{new_name}.json")
                    os.rename(old_path, new_path)
                    if model_name in model_configs:
                        model_configs[new_name] = model_configs.pop(model_name)
                    save_model_configs(model_configs)
                    return redirect(url_for("manage_models"))
                elif action == "delete_model":
                    if model_name not in saved_models:
                        raise ValueError("Model not found.")
                    path = os.path.join(MODEL_DIR, f"{model_name}.json")
                    if os.path.exists(path):
                        os.remove(path)
                    if model_name in model_configs:
                        model_configs.pop(model_name)
                        save_model_configs(model_configs)
                    return redirect(url_for("manage_models"))
            except Exception as exc:
                error_html = f"<p style='color:#ff7b7b;'><strong>Error:</strong> {exc}</p>"
            saved_models = list_saved_models()
            model_configs = load_model_configs()
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
                f"data-sell='{float(cfg.get('sell_threshold', 0.4)):.2f}'>"
                f"<strong>{model_name}</strong>"
                f"<span>{cfg.get('ticker')} • {cfg.get('interval')} • {int(cfg.get('rows', 250))} rows • "
                f"BUY>{float(cfg.get('buy_threshold', 0.6)):.2f} / SELL<{float(cfg.get('sell_threshold', 0.4)):.2f}</span>"
                f"<em>{include_badge}</em>"
                "</button>"
            )

        return f"""
        <html>
          <head><title>Manage Models</title></head>
          <body>
            <style>
              :root {{
                --bg: #0a0a0f;
                --panel: #15161f;
                --panel-2: #1d1f2b;
                --border: #2a2d3a;
                --text: #e7e9f1;
                --muted: #9aa1b2;
                --accent: #66a3ff;
              }}
              * {{ box-sizing: border-box; }}
              body {{ margin: 0; background: radial-gradient(circle at top, #101425 0%, var(--bg) 50%); color: var(--text); font-family: Inter, Segoe UI, Arial, sans-serif; }}
              .container {{ max-width: 1100px; margin: 0 auto; padding: 2rem; }}
              .topbar {{ position: sticky; top: 0; z-index: 50; background: rgba(10, 10, 15, 0.92); border-bottom: 1px solid var(--border); backdrop-filter: blur(6px); }}
              .topbar-inner {{ max-width: 1100px; margin: 0 auto; padding: 0.9rem 2rem; display: flex; align-items: center; gap: 1rem; }}
              .brand {{ font-weight: 700; color: #d8e6ff; text-decoration: none; margin-right: auto; }}
              .tab-link {{ color: #9fb9ea; text-decoration: none; padding: 0.4rem 0.65rem; border-radius: 8px; border: 1px solid transparent; }}
              .tab-link:hover, .tab-link.active {{ color: #e6f0ff; border-color: var(--border); background: #121827; }}
              .card {{ background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.1rem; margin-bottom: 1rem; }}
              .muted {{ color: var(--muted); }}
              .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 0.8rem; }}
              .model-card {{ text-align: left; border: 1px solid var(--border); border-radius: 12px; background: #0f1118; color: var(--text); padding: 0.8rem; cursor: pointer; display: flex; flex-direction: column; gap: 0.35rem; }}
              .model-card span {{ color: var(--muted); font-size: 0.92rem; }}
              .model-card em {{ color: #a8c9ff; font-style: normal; font-size: 0.82rem; }}
              table {{ width: 100%; border-collapse: collapse; }}
              th, td {{ border-bottom: 1px solid var(--border); padding: 0.45rem 0.35rem; text-align: left; }}
              th {{ color: #bfc8df; }}
              a, button {{ color: inherit; }}
              .btn-link {{ display: inline-block; margin-bottom: 0.8rem; color: #91bbff; text-decoration: none; }}
              #contextMenu {{ position: fixed; display: none; z-index: 40; min-width: 170px; background: #111521; border: 1px solid var(--border); border-radius: 10px; box-shadow: 0 16px 30px rgba(0,0,0,0.45); }}
              #contextMenu button {{ width: 100%; border: none; background: transparent; color: var(--text); text-align: left; padding: 0.65rem 0.75rem; cursor: pointer; }}
              #contextMenu button:hover {{ background: #20293f; }}
              .modal-backdrop {{ position: fixed; inset: 0; background: rgba(0,0,0,0.55); display: none; align-items: center; justify-content: center; z-index: 30; }}
              .modal {{ width: min(560px, 92vw); background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%); border: 1px solid var(--border); border-radius: 14px; padding: 1rem; }}
              .form-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }}
              label {{ display: block; color: var(--muted); font-size: 0.92rem; }}
              input, select {{ width: 100%; margin-top: 0.35rem; background: #0f1118; color: var(--text); border: 1px solid var(--border); border-radius: 10px; padding: 0.58rem 0.65rem; }}
              .row-actions {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-top: 0.9rem; }}
              .row-actions button {{ border: none; border-radius: 10px; padding: 0.62rem 0.65rem; cursor: pointer; font-weight: 600; }}
              .primary {{ background: var(--accent); color: #081121; }}
              .secondary {{ background: #2c3555; color: #e8ecff; }}
            </style>
            <nav class="topbar">
              <div class="topbar-inner">
                <a href="/" class="brand">Quant Trader</a>
                <a href="/" class="tab-link">Model</a>
                <a href="/manage-models" class="tab-link active">Manage Models</a>
                <a href="/#present-mode" class="tab-link">Present Mode</a>
              </div>
            </nav>
            <div class="container">
              <h1>Manage Models</h1>
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
                document.getElementById("cfgInclude").value = card.dataset.include || "1";
                settingsModal.style.display = "flex";
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
    def index() -> str:
        result_html = ""
        error_html = ""
        ticker = request.form.get("ticker", "AAPL").upper().strip()
        interval = request.form.get("interval", "1d")
        rows = request.form.get("rows", "250")
        split_style = request.form.get("split_style", "shuffled")
        buy_threshold_raw = request.form.get("buy_threshold", "").strip()
        sell_threshold_raw = request.form.get("sell_threshold", "").strip()
        model_name = request.form.get("model_name", "").strip()
        selected_model = request.form.get("selected_model", "__new__")
        present_ticker = request.form.get("present_ticker", ticker).upper().strip()
        present_interval = request.form.get("present_interval", interval)
        present_rows = request.form.get("present_rows", rows)
        present_buy_raw = request.form.get("present_buy_threshold", "").strip()
        present_sell_raw = request.form.get("present_sell_threshold", "").strip()
        present_model = request.form.get("present_model", selected_model)
        mode = request.form.get("mode", "train")
        train_action = request.form.get("train_action", "train")
        saved_models = list_saved_models()
        present_html = ""
        run_all_html = ""
        run_all_rows = ""

        if request.method == "POST":
            try:
                if mode == "present":
                    present_row_count = int(present_rows)
                    present_buy_threshold, present_sell_threshold = parse_thresholds(present_buy_raw, present_sell_raw)
                    dataset = fetch_yahoo_rows(ticker=present_ticker, interval=present_interval, row_count=present_row_count)
                    latest_row = dataset[-1]
                    if present_model == "__new__":
                        bundle = train_strategy_models(dataset, split_style=split_style)
                    else:
                        bundle = load_model_bundle(present_model)
                    prediction = predict_signal(
                        bundle,
                        latest_row,
                        buy_threshold=present_buy_threshold,
                        sell_threshold=present_sell_threshold,
                    )
                    present_html = f"""
                    <section class="results">
                      <article class="card">
                        <h2>Present Mode • {present_ticker} ({present_interval})</h2>
                        <p class="muted">Model: {"Freshly trained on current dataset" if present_model == "__new__" else present_model}</p>
                        <p class="muted">Decision rule: BUY if P(Up)&gt;{present_buy_threshold:.2f}, SELL if P(Up)&lt;{present_sell_threshold:.2f}, else HOLD.</p>
                        <p><span class="muted">Expected Return (next candle)</span> <strong>{prediction['expected_return']:+.4%}</strong></p>
                        <p><span class="muted">P(Up)</span> <strong>{prediction['p_up']:.2%}</strong></p>
                        <p><span class="muted">Action</span> <strong>{prediction['action']}</strong></p>
                      </article>
                    </section>
                    """
                elif mode == "present_all":
                    model_configs = load_model_configs()
                    model_configs = {name: get_model_config(name, model_configs) for name in saved_models}
                    run_all_rows = build_run_all_rows(saved_models, model_configs)
                    run_all_html = "<p class='muted'>Latest outputs for all models currently included in Run All.</p>"
                else:
                    row_count = int(rows)
                    buy_threshold, sell_threshold = parse_thresholds(buy_threshold_raw, sell_threshold_raw)
                    if split_style not in ("shuffled", "chronological"):
                        raise ValueError("Split style must be either shuffled (legacy) or chronological (time-aware).")
                    dataset = fetch_yahoo_rows(ticker=ticker, interval=interval, row_count=row_count)
                    train_rows, test_rows = train_test_split(dataset, split_style=split_style)
                    features = build_default_strategy_features()
                    x_test_raw = features.transform(test_rows)
                    y_test_ret = [r["return_next"] for r in test_rows]
                    y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]
                    if selected_model != "__new__":
                        loaded = load_model_bundle(selected_model)
                        metrics = evaluate_bundle(
                            loaded,
                            x_test_raw,
                            y_test_ret,
                            y_test_dir,
                            eval_rows=dataset,
                            split_style=split_style,
                            buy_threshold=buy_threshold,
                            sell_threshold=sell_threshold,
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
                        )
                        metrics["train_size"] = bundle["train_size"]
                        if train_action == "train" and model_name:
                            save_model_bundle(model_name, bundle)
                            model_configs = load_model_configs()
                            model_configs[sanitize_model_name(model_name)] = {
                                "ticker": ticker,
                                "interval": interval,
                                "rows": row_count,
                                "include_in_run_all": True,
                                "buy_threshold": buy_threshold,
                                "sell_threshold": sell_threshold,
                            }
                            save_model_configs(model_configs)
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
                    result_html = f"""
                <section class="results">
                  <div class="section-heading">
                    <h2>Results • {ticker} ({interval})</h2>
                    <p class="muted">Rows: {row_count} | Train: {metrics['train_size']} | Test: {metrics['test_size']} | Split: {metrics['split_style']}</p>
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
                      <p class="muted">Long P&gt;{metrics['strategy']['long_threshold']:.2f} · Short P&lt;{metrics['strategy']['short_threshold']:.2f} · Cost 0.05%</p>
                      <p><span class="muted">Total Return</span> <strong>{metrics['strategy']['total_return']:+.2%}</strong></p>
                      <p><span class="muted">Buy &amp; Hold Return (test rows)</span> <strong>{metrics['strategy']['buy_hold_total_return']:+.2%}</strong></p>
                      <p><span class="muted">Sharpe</span> <strong>{metrics['strategy']['sharpe']:.3f}</strong></p>
                      <p><span class="muted">Max Drawdown</span> {metrics['strategy']['max_drawdown']:.2%}</p>
                      <p><span class="muted">Win Rate / Trades</span> {metrics['strategy']['win_rate']:.2%} / {int(metrics['strategy']['trade_count'])}</p>
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
                --bg: #0a0a0f;
                --panel: #15161f;
                --panel-2: #1d1f2b;
                --border: #2a2d3a;
                --text: #e7e9f1;
                --muted: #9aa1b2;
                --accent: #66a3ff;
              }}
              * {{ box-sizing: border-box; }}
              body {{
                margin: 0;
                background: radial-gradient(circle at top, #101425 0%, var(--bg) 50%);
                color: var(--text);
                font-family: Inter, Segoe UI, Arial, sans-serif;
              }}
              .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
              }}
              .topbar {{ position: sticky; top: 0; z-index: 50; background: rgba(10, 10, 15, 0.92); border-bottom: 1px solid var(--border); backdrop-filter: blur(6px); }}
              .topbar-inner {{ max-width: 1200px; margin: 0 auto; padding: 0.9rem 2rem; display: flex; align-items: center; gap: 1rem; }}
              .brand {{ font-weight: 700; color: #d8e6ff; text-decoration: none; margin-right: auto; }}
              .tab-link {{ color: #9fb9ea; text-decoration: none; padding: 0.4rem 0.65rem; border-radius: 8px; border: 1px solid transparent; }}
              .tab-link:hover, .tab-link.active {{ color: #e6f0ff; border-color: var(--border); background: #121827; }}
              h1, h2, h3 {{ margin-top: 0; }}
              .card {{
                background: linear-gradient(180deg, var(--panel) 0%, var(--panel-2) 100%);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 1rem 1.1rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
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
                background: #0f1118;
                color: var(--text);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.65rem 0.7rem;
              }}
              button {{
                background: var(--accent);
                color: #081121;
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
                background: #2b3250;
                color: #e7e9f1;
                border: 1px solid #3a4467;
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
              th {{ color: #bfc8df; font-weight: 600; }}
              pre {{
                white-space: pre-wrap;
                background: #0f1118;
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.8rem;
              }}
              details {{
                background: #10131c;
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 0.7rem 0.9rem;
                margin-bottom: 1rem;
              }}
              summary {{
                cursor: pointer;
                color: #d5dcf4;
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
                background: rgba(6, 8, 14, 0.78);
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
                color: #dbe7ff;
              }}
              .progress-track {{
                width: 100%;
                height: 14px;
                border-radius: 999px;
                background: #0f1118;
                border: 1px solid var(--border);
                overflow: hidden;
              }}
              .progress-fill {{
                width: 0%;
                height: 100%;
                background: linear-gradient(90deg, #66a3ff 0%, #7bd88f 100%);
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
                <a href="/" class="brand">Quant Trader</a>
                <a href="/" class="tab-link active">Model</a>
                <a href="/manage-models" class="tab-link">Manage Models</a>
                <a href="#present-mode" class="tab-link">Present Mode</a>
              </div>
            </nav>
            <div class="container">
            <h1>Quant Model Trainer</h1>
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
              <p class="muted">Runs each saved model that is enabled in Manage Models and shows the live prediction.</p>
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
