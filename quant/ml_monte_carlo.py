from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Sequence, Tuple


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


def quantile(sorted_values: Sequence[float], q: float) -> float:
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


def summarize_distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p5": 0.0,
            "p95": 0.0,
        }
    ordered = sorted(float(v) for v in values)
    mean_val = sum(ordered) / len(ordered)
    variance = sum((val - mean_val) ** 2 for val in ordered) / len(ordered)
    return {
        "mean": mean_val,
        "median": quantile(ordered, 0.5),
        "std": math.sqrt(variance),
        "min": ordered[0],
        "max": ordered[-1],
        "p5": quantile(ordered, 0.05),
        "p95": quantile(ordered, 0.95),
    }


def cvar(values: List[float], alpha: float = 0.05) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    cutoff = max(1, int(len(sorted_vals) * alpha))
    return sum(sorted_vals[:cutoff]) / cutoff


def distribution_shape(values: List[float], mean_val: float, std_val: float) -> Tuple[float, float]:
    if not values or std_val <= 1e-12:
        return 0.0, 0.0
    n = len(values)
    skew = sum(((value - mean_val) / std_val) ** 3 for value in values) / n
    kurtosis_excess = (sum(((value - mean_val) / std_val) ** 4 for value in values) / n) - 3.0
    return skew, kurtosis_excess


def run_monte_carlo_backtest(
    returns: List[float],
    probs: List[float],
    strategy_fn: Callable[..., Dict[str, object]],
    n_sim: int = 500,
    method: str = "block",
    block_size: int = 20,
    seed: int | None = None,
    return_equity_curves: int = 0,
    **strategy_kwargs: Any,
) -> Dict[str, Any]:
    if len(returns) != len(probs):
        raise ValueError("returns and probs must have the same length.")
    if n_sim < 1:
        raise ValueError("n_sim must be at least 1.")
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if method not in {"bootstrap", "shuffle", "block"}:
        raise ValueError("method must be one of: bootstrap, shuffle, block.")

    rng = random.Random(seed)
    base_metrics = strategy_fn(returns, probs, **strategy_kwargs)
    sampled_unit_returns = [float(value) for value in base_metrics.get("bar_returns", [])]
    if not sampled_unit_returns:
        sampled_unit_returns = [float(value) for value in base_metrics.get("trade_returns", [])]
    if not sampled_unit_returns:
        empty_summary = {
            "mean_return": 0.0,
            "median_return": 0.0,
            "std_return": 0.0,
            "min_return": 0.0,
            "max_return": 0.0,
            "p5_return": 0.0,
            "p95_return": 0.0,
            "cvar_5_return": 0.0,
            "log_mean_return": 0.0,
            "log_median_return": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "mean_sharpe": 0.0,
            "mean_drawdown": 0.0,
            "worst_drawdown": 0.0,
            "probability_of_loss": 0.0,
            "probability_of_large_loss": 0.0,
            "probability_of_ruin": 0.0,
        }
        return {"raw_results": [], "summary": empty_summary}

    raw_results: List[Dict[str, float]] = []
    equity_curves: List[List[float]] = []
    n = len(sampled_unit_returns)
    log_total_returns: List[float] = []

    for _ in range(n_sim):
        if method == "bootstrap":
            sampled_returns = [sampled_unit_returns[rng.randrange(n)] for _ in range(n)]
        elif method == "shuffle":
            sampled_returns = list(sampled_unit_returns)
            rng.shuffle(sampled_returns)
        else:
            sampled_returns = []
            while len(sampled_returns) < n:
                start = rng.randrange(n)
                end = min(n, start + block_size)
                sampled_returns.extend(sampled_unit_returns[start:end])
            sampled_returns = sampled_returns[:n]

        equity = 1.0
        peak = 1.0
        max_drawdown = 0.0
        for trade_ret in sampled_returns:
            equity *= 1.0 + trade_ret
            equity = max(equity, 0.0)
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, ((peak - equity) / peak) if peak > 0 else 0.0)
        total_return = equity - 1.0
        mean_ret = (sum(sampled_returns) / len(sampled_returns)) if sampled_returns else 0.0
        sd_ret = stddev(sampled_returns)
        sharpe = (mean_ret / sd_ret * math.sqrt(252.0)) if sd_ret > 1e-12 else 0.0
        win_rate = (sum(1 for value in sampled_returns if value > 0.0) / len(sampled_returns)) if sampled_returns else 0.0
        total_log_return = sum(math.log1p(value) if value > -1.0 else float("-inf") for value in sampled_returns)
        log_total_return = -1.0 if math.isinf(total_log_return) and total_log_return < 0 else (math.exp(total_log_return) - 1.0)
        log_total_returns.append(log_total_return)
        raw_results.append(
            {
                "total_return": float(total_return),
                "sharpe": float(sharpe),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate),
                "log_total_return": float(log_total_return),
            }
        )

        if len(equity_curves) < max(0, int(return_equity_curves)):
            curve: List[float] = []
            equity = 1.0
            for ret in sampled_returns:
                equity *= 1.0 + ret
                curve.append(equity)
            equity_curves.append(curve)

    total_returns = [item["total_return"] for item in raw_results]
    sharpes = [item["sharpe"] for item in raw_results]
    drawdowns = [item["max_drawdown"] for item in raw_results]
    returns_summary = summarize_distribution(total_returns)
    log_returns_summary = summarize_distribution(log_total_returns)
    skewness, kurtosis = distribution_shape(total_returns, returns_summary["mean"], returns_summary["std"])
    summary = {
        "mean_return": returns_summary["mean"],
        "median_return": returns_summary["median"],
        "std_return": returns_summary["std"],
        "min_return": returns_summary["min"],
        "max_return": returns_summary["max"],
        "p5_return": returns_summary["p5"],
        "p95_return": returns_summary["p95"],
        "cvar_5_return": cvar(total_returns, 0.05),
        "log_mean_return": log_returns_summary["mean"],
        "log_median_return": log_returns_summary["median"],
        "skewness": skewness,
        "kurtosis": kurtosis,
        "mean_sharpe": (sum(sharpes) / len(sharpes)) if sharpes else 0.0,
        "mean_drawdown": (sum(drawdowns) / len(drawdowns)) if drawdowns else 0.0,
        "worst_drawdown": max(drawdowns) if drawdowns else 0.0,
        "probability_of_loss": (sum(1 for value in total_returns if value < 0.0) / len(total_returns)) if total_returns else 0.0,
        "probability_of_large_loss": (sum(1 for value in total_returns if value < -0.5) / len(total_returns)) if total_returns else 0.0,
        "probability_of_ruin": (sum(1 for value in total_returns if value < -0.9) / len(total_returns)) if total_returns else 0.0,
    }

    out: Dict[str, Any] = {"raw_results": raw_results, "summary": summary}
    if equity_curves:
        out["equity_curves"] = equity_curves
    if summary["mean_sharpe"] > 3.0:
        print("WARNING: Monte Carlo mean Sharpe exceeds 3.0; results may be unrealistic.")
    if summary["median_return"] > 5.0:
        print("WARNING: Monte Carlo median return exceeds 500%; results may be unrealistic.")
    if summary["probability_of_loss"] < 0.05:
        print("WARNING: Monte Carlo probability of loss below 5%; results may be unrealistic.")
    return out
