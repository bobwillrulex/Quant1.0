from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

from .types import FeatureFn, Row


@dataclass
class FeatureSpec:
    name: str
    fn: FeatureFn


class StrategyFeatureBuilder:
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


def train_test_split(rows: Sequence[Row], test_ratio: float = 0.25, split_style: SplitStyle = "shuffled") -> Tuple[List[Row], List[Row]]:
    data = list(rows)
    if split_style == "shuffled":
        random.shuffle(data)
    elif split_style != "chronological":
        raise ValueError("split_style must be either 'shuffled' or 'chronological'.")
    split = max(1, int(len(data) * (1 - test_ratio)))
    return data[:split], data[split:]


def mse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / max(1, len(y_true))


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
    return {"accuracy": (tp + tn) / max(1, len(y_true)), "precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)}


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


def build_default_strategy_features() -> StrategyFeatureBuilder:
    builder = StrategyFeatureBuilder()
    builder.add("stoch_extreme", lambda r: 1.0 if (r["stoch_rsi"] > 80 or r["stoch_rsi"] < 20) else 0.0)
    builder.add("macd_hist", lambda r: r["macd_hist"])
    builder.add("macd_hist_delta", lambda r: r["macd_hist_delta"])
    builder.add("macd_green_increasing", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_red_recovering", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] > 0) else 0.0)
    builder.add("macd_green_fading", lambda r: 1.0 if (r["macd_hist"] > 0 and r["macd_hist_delta"] < 0) else 0.0)
    builder.add("macd_red_deepening", lambda r: 1.0 if (r["macd_hist"] < 0 and r["macd_hist_delta"] < 0) else 0.0)
    builder.add("first_green_fvg_dip", lambda r: r["first_green_fvg_dip"])
    builder.add("first_red_fvg_touch", lambda r: r["first_red_fvg_touch"])
    builder.add("fvg_green_size", lambda r: min(r["fvg_green_size"], 3.0))
    builder.add("fvg_red_size", lambda r: min(r["fvg_red_size"], 3.0))
    builder.add("fvg_bull_signal", lambda r: r["first_green_fvg_dip"] * min(r["fvg_green_size"], 3.0))
    builder.add("fvg_bear_signal", lambda r: r["first_red_fvg_touch"] * min(r["fvg_red_size"], 3.0))
    builder.add("fvg_conflict_penalty", lambda r: r["fvg_red_above_green"] * min(r["fvg_green_size"], 3.0))
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
    y_train_ret = [r["return_next"] for r in train_rows]
    y_test_ret = [r["return_next"] for r in test_rows]
    y_train_dir = [1 if r > 0 else 0 for r in y_train_ret]
    y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]
    lin = LinearRegressionGD(learning_rate=0.03, epochs=800)
    lin.fit(x_train, y_train_ret)
    logit = LogisticRegressionGD(learning_rate=0.05, epochs=700)
    logit.fit(x_train, y_train_dir)
    return {"feature_names": features.names(), "means": means, "stds": stds, "lin_weights": lin.weights, "lin_bias": lin.bias, "logit_weights": logit.weights, "logit_bias": logit.bias, "train_size": len(train_rows), "test_size": len(test_rows), "split_style": split_style, "x_test_raw": x_test_raw, "y_test_ret": y_test_ret, "y_test_dir": y_test_dir}


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
        result.append({"bucket_low": lo, "bucket_high": hi, "count": float(len(values)), "predicted_mean": sum(probs) / len(probs), "actual_win_rate": sum(values) / len(values)})
    return result


def confidence_edge_analysis(y_true: Sequence[int], y_prob: Sequence[float]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for threshold in [0.6, 0.7]:
        preds = [(a, p) for a, p in zip(y_true, y_prob) if p > threshold]
        out[f"p_gt_{threshold:.1f}"] = {"count": float(len(preds)), "accuracy": (sum(1 for a, _ in preds if a == 1) / len(preds)) if preds else 0.0}
    return out


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac)


def strategy_metrics(
    returns: Sequence[float],
    probs: Sequence[float],
    long_threshold: float = 0.6,
    short_threshold: float = 0.4,
    trade_cost: float = 0.0005,
    buy_hold_returns: Sequence[float] | None = None,
    allow_short: bool = True,
    min_hold_bars: int = 0,
    prob_smoothing_window: int = 3,
) -> Dict[str, object]:
    smooth_window = max(1, int(prob_smoothing_window))
    smoothed_probs: List[float] = []
    for i in range(len(probs)):
        start = max(0, i - smooth_window + 1)
        p_window = probs[start : i + 1]
        smoothed_probs.append(sum(p_window) / len(p_window))

    sell_threshold = short_threshold
    positions: List[int] = []
    current_pos = 0
    bars_in_position = 0
    for p in smoothed_probs:
        if current_pos == 0:
            if p > long_threshold:
                current_pos = 1
                bars_in_position = 1
            elif p < short_threshold and allow_short:
                current_pos = -1
                bars_in_position = 1
        elif current_pos == 1:
            if bars_in_position >= min_hold_bars and p < sell_threshold:
                current_pos = 0
                bars_in_position = 0
            else:
                bars_in_position += 1
        elif current_pos == -1:
            if bars_in_position >= min_hold_bars and p > long_threshold:
                current_pos = 0
                bars_in_position = 0
            else:
                bars_in_position += 1
        positions.append(current_pos)
    pnl: List[float] = []
    prev_pos = 0
    wins = 0
    trades = 0
    in_trade = False
    trade_pnl = 0.0
    closed_trade_pnls: List[float] = []
    for pos, ret in zip(positions, returns):
        turnover = abs(pos - prev_pos)
        day_pnl = pos * ret - turnover * trade_cost
        pnl.append(day_pnl)
        if prev_pos == 0 and pos != 0:
            in_trade = True
            trade_pnl = 0.0
        if in_trade:
            trade_pnl += day_pnl
        if prev_pos != 0 and pos == 0 and in_trade:
            trades += 1
            closed_trade_pnls.append(trade_pnl)
            if trade_pnl > 0:
                wins += 1
            in_trade = False
            trade_pnl = 0.0
        prev_pos = pos
    if in_trade:
        trades += 1
        closed_trade_pnls.append(trade_pnl)
        if trade_pnl > 0:
            wins += 1
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    drawdowns: List[float] = []
    for r in pnl:
        equity *= (1.0 + r)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        drawdowns.append(dd)
        max_drawdown = max(max_drawdown, dd)
    avg_drawdown = (sum(drawdowns) / len(drawdowns)) if drawdowns else 0.0
    hold_lengths: List[int] = []
    active_pos = 0
    active_len = 0
    for pos in positions:
        if pos != 0 and pos == active_pos:
            active_len += 1
            continue
        if active_pos != 0 and active_len > 0:
            hold_lengths.append(active_len)
        if pos != 0:
            active_pos = pos
            active_len = 1
        else:
            active_pos = 0
            active_len = 0
    if active_pos != 0 and active_len > 0:
        hold_lengths.append(active_len)
    hold_lengths_float = [float(x) for x in hold_lengths]
    sorted_holds = sorted(hold_lengths_float)
    hold_time_stats = {
        "count": float(len(hold_lengths_float)),
        "min": float(sorted_holds[0]) if sorted_holds else 0.0,
        "q1": _quantile(sorted_holds, 0.25),
        "median": _quantile(sorted_holds, 0.5),
        "q3": _quantile(sorted_holds, 0.75),
        "max": float(sorted_holds[-1]) if sorted_holds else 0.0,
    }
    sd = stddev(pnl)
    sharpe = ((sum(pnl) / len(pnl)) / sd * math.sqrt(252.0)) if (sd > 1e-12 and pnl) else 0.0
    buy_hold_source = buy_hold_returns if buy_hold_returns is not None else returns
    return {
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "allow_short": 1.0 if allow_short else 0.0,
        "trade_cost": trade_cost,
        "min_hold_bars": float(min_hold_bars),
        "prob_smoothing_window": float(smooth_window),
        "total_return": equity - 1.0,
        "buy_hold_total_return": compounded_return(buy_hold_source),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "win_rate": (wins / trades) if trades else 0.0,
        "trade_count": float(trades),
        "avg_gain_per_trade": (sum(closed_trade_pnls) / len(closed_trade_pnls)) if closed_trade_pnls else 0.0,
        "max_loss_per_trade": min(closed_trade_pnls) if closed_trade_pnls else 0.0,
        "hold_time_stats": hold_time_stats,
    }


def pnl_signal_strength_breakdown(returns: Sequence[float], probs: Sequence[float], trade_cost: float = 0.0005, allow_short: bool = True) -> List[Dict[str, float]]:
    buckets = {"weak_0.50_0.55": (0.50, 0.55), "medium_0.55_0.65": (0.55, 0.65), "strong_0.65_1.00": (0.65, 1.00)}
    out: List[Dict[str, float]] = []
    for name, (lo, hi) in buckets.items():
        pnl = []
        for p, r in zip(probs, returns):
            confidence = max(p, 1.0 - p)
            if lo <= confidence < hi:
                pos = 1 if p >= 0.5 else (-1 if allow_short else 0)
                pnl.append(pos * r - trade_cost)
        if pnl:
            out.append({"bucket": name, "count": float(len(pnl)), "avg_pnl": sum(pnl) / len(pnl), "total_pnl": sum(pnl)})
    return out


def pnl_market_regime_breakdown(returns: Sequence[float], probs: Sequence[float], trade_cost: float = 0.0005, allow_short: bool = True) -> List[Dict[str, float]]:
    out = {"trending": [], "sideways": [], "high_volatility": []}
    window = 20
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        r_win = returns[start : i + 1]
        trend = abs(sum(r_win) / max(1, len(r_win)))
        vol = stddev(r_win)
        p = probs[i]
        pos = 1 if p > 0.6 else (-1 if (p < 0.4 and allow_short) else 0)
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
            result.append({"regime": regime, "count": float(len(pnl_list)), "avg_pnl": sum(pnl_list) / len(pnl_list), "total_pnl": sum(pnl_list)})
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
        results.append({"window": float(idx + 1), "train_size": float(len(train_rows)), "test_size": float(len(test_rows)), "accuracy": accuracy(y_test_dir, up_prob), "mse": mse(y_test_ret, ret_pred)})
    return results


def feature_ablation_analysis(rows: Sequence[Row], feature_names: Sequence[str], split_style: SplitStyle = "shuffled") -> List[Dict[str, float]]:
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
        out.append({"removed_feature": removed, "accuracy_delta": accuracy(y_test_dir_loop, up_prob) - full_accuracy, "mse_delta": mse(y_test_ret_loop, ret_pred) - full_mse})
    return out


def error_analysis(y_test_ret: Sequence[float], up_prob: Sequence[float], ret_pred: Sequence[float], top_n: int = 5) -> Dict[str, List[Dict[str, float]]]:
    largest_errors = sorted([{"index": float(i), "abs_error": abs(y_test_ret[i] - ret_pred[i]), "actual_return": y_test_ret[i], "pred_return": ret_pred[i]} for i in range(len(y_test_ret))], key=lambda x: x["abs_error"], reverse=True)[:top_n]
    high_conf_wrong = []
    for i, (ret, p) in enumerate(zip(y_test_ret, up_prob)):
        actual_up = 1 if ret > 0 else 0
        pred_up = 1 if p >= 0.5 else 0
        confidence = max(p, 1.0 - p)
        if actual_up != pred_up and confidence >= 0.7:
            high_conf_wrong.append({"index": float(i), "p_up": p, "actual_return": ret, "confidence": confidence})
    high_conf_wrong = sorted(high_conf_wrong, key=lambda x: x["confidence"], reverse=True)[:top_n]
    return {"largest_return_errors": largest_errors, "high_confidence_wrong_calls": high_conf_wrong}


def evaluate_bundle(bundle: Dict[str, object], x_test_raw: Sequence[Sequence[float]], y_test_ret: Sequence[float], y_test_dir: Sequence[int], eval_rows: Sequence[Row] | None = None, split_style: SplitStyle = "shuffled", buy_threshold: float = 0.6, sell_threshold: float = 0.4, allow_short: bool = True) -> Dict[str, object]:
    if not x_test_raw:
        raise ValueError("No rows available for evaluation.")
    if len(x_test_raw[0]) != len(bundle["feature_names"]):
        raise ValueError("Saved model feature size does not match current strategy feature set.")
    x_test = standardize_apply(x_test_raw, bundle["means"], bundle["stds"])
    ret_pred = [sum(w * v for w, v in zip(bundle["lin_weights"], row)) + bundle["lin_bias"] for row in x_test]
    up_prob = [sigmoid(sum(w * v for w, v in zip(bundle["logit_weights"], row)) + bundle["logit_bias"]) for row in x_test]
    cls = classification_metrics(y_test_dir, up_prob)
    baseline_up_accuracy = sum(y_test_dir) / max(1, len(y_test_dir))
    baseline_zero = [0.0] * len(y_test_ret)
    strategy = strategy_metrics(y_test_ret, up_prob, long_threshold=buy_threshold, short_threshold=sell_threshold, trade_cost=0.0005, buy_hold_returns=y_test_ret, allow_short=allow_short)
    preview = [{"expected_return": ret_pred[i], "p_up": up_prob[i], "actual_return": y_test_ret[i]} for i in range(min(5, len(x_test)))]
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
        "tp": int(cls["tp"]), "tn": int(cls["tn"]), "fp": int(cls["fp"]), "fn": int(cls["fn"]),
        "lin_weights": list(zip(bundle["feature_names"], bundle["lin_weights"])), "lin_bias": bundle["lin_bias"],
        "logit_weights": list(zip(bundle["feature_names"], bundle["logit_weights"])), "logit_bias": bundle["logit_bias"],
        "preview": preview, "test_size": len(y_test_ret), "split_style": split_style,
        "calibration": calibration_buckets(y_test_dir, up_prob),
        "confidence_edge": confidence_edge_analysis(y_test_dir, up_prob),
        "strategy": strategy,
        "pnl_by_signal_strength": pnl_signal_strength_breakdown(y_test_ret, up_prob, trade_cost=0.0005, allow_short=allow_short),
        "pnl_by_regime": pnl_market_regime_breakdown(y_test_ret, up_prob, trade_cost=0.0005, allow_short=allow_short),
        "walk_forward": walk_forward_validation_rows(rows=eval_rows, max_windows=4) if eval_rows else [],
        "feature_ablation": feature_ablation_analysis(eval_rows, bundle["feature_names"], split_style=split_style) if eval_rows else [],
        "error_analysis": error_analysis(y_test_ret, up_prob, ret_pred, top_n=5),
    }


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
        f"Max Drawdown: {strat['max_drawdown']:.2%}, Avg Drawdown: {strat['avg_drawdown']:.2%}, "
        f"Win Rate: {strat['win_rate']:.2%}, Trades: {int(strat['trade_count'])}, "
        f"Avg Gain/Trade: {strat['avg_gain_per_trade']:+.4%}, Max Loss/Trade: {strat['max_loss_per_trade']:+.4%}"
    )
    print(f"Buy & Hold Return (test rows): {strat['buy_hold_total_return']:+.2%}")


def run_model_metrics(rows: Sequence[Row]) -> Dict[str, object]:
    bundle = train_strategy_models(rows)
    metrics = evaluate_bundle(bundle, bundle["x_test_raw"], bundle["y_test_ret"], bundle["y_test_dir"], eval_rows=rows, split_style=bundle["split_style"])
    metrics["train_size"] = bundle["train_size"]
    return metrics


def predict_signal(bundle: Dict[str, object], row: Row, *, buy_threshold: float = 0.6, sell_threshold: float = 0.4, long_only: bool = False) -> Dict[str, float | str]:
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
    if long_only and action == "SELL":
        action = "SELL"
    return {"expected_return": expected_return, "p_up": p_up, "action": action}
