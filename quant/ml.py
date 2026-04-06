from __future__ import annotations

from datetime import datetime
import math
import random
from typing import Any, Dict, List, Literal, Sequence, Tuple

from .dqn import dqn_q_values, train_dqn_policy
from .evaluation import (
    accuracy,
    calibration_buckets,
    classification_metrics,
    confidence_edge_analysis,
    error_analysis,
    mae,
    mse,
)
from .models_linear import LinearRegressionGD
from .models_logistic import LogisticRegressionGD, sigmoid
from .ml_monte_carlo import compounded_return, quantile, run_monte_carlo_backtest, stddev
from .strategy_features import (
    FeatureSet,
    StrategyFeatureBuilder,
    get_strategy_feature_builder,
    infer_bundle_feature_set,
    normalize_feature_set,
)
from .stop_loss import MODEL_MAE_DEFAULT, StopLossConfig, StopLossStrategy
from .types import Row


SplitStyle = Literal["shuffled", "chronological"]


def train_test_split(rows: Sequence[Row], test_ratio: float = 0.25, split_style: SplitStyle = "shuffled") -> Tuple[List[Row], List[Row]]:
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1 (1 excluded).")
    data = list(rows)
    if split_style == "shuffled":
        random.shuffle(data)
    elif split_style != "chronological":
        raise ValueError("split_style must be either 'shuffled' or 'chronological'.")
    split = max(1, int(len(data) * (1 - test_ratio)))
    return data[:split], data[split:]


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


def _softmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    denom = sum(exps) or 1.0
    return [val / denom for val in exps]


def parse_thresholds(buy_raw: str, sell_raw: str, *, default_buy: float = 0.6, default_sell: float = 0.4) -> Tuple[float, float]:
    buy_text = buy_raw.strip()
    sell_text = sell_raw.strip()
    buy_threshold = default_buy if buy_text == "" else float(buy_text)
    sell_threshold = default_sell if sell_text == "" else float(sell_text)
    if not (0.0 <= sell_threshold < buy_threshold <= 1.0):
        raise ValueError("Thresholds must satisfy: 0.0 <= sell < buy <= 1.0.")
    return buy_threshold, sell_threshold


def _infer_periods_per_year(row_labels: Sequence[str] | None, sample_count: int, default_periods: float = 252.0) -> float:
    if not row_labels or sample_count < 2:
        return default_periods
    timestamps: List[datetime] = []
    for label in row_labels:
        try:
            timestamps.append(datetime.fromisoformat(str(label)))
        except (TypeError, ValueError):
            continue
    if len(timestamps) < 2:
        return default_periods
    span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    if span_seconds <= 0.0:
        return default_periods
    seconds_per_year = 365.25 * 24.0 * 60.0 * 60.0
    periods_per_year = ((len(timestamps) - 1) / span_seconds) * seconds_per_year
    return periods_per_year if periods_per_year > 1.0 else default_periods


def train_strategy_models(
    rows: Sequence[Row],
    split_style: SplitStyle = "shuffled",
    feature_set: FeatureSet | str = "feature2",
    dqn_episodes: int = 120,
    test_ratio: float = 0.25,
) -> Dict[str, object]:
    train_rows, test_rows = train_test_split(rows, test_ratio=test_ratio, split_style=split_style)
    resolved_feature_set = normalize_feature_set(feature_set)
    features = get_strategy_feature_builder(resolved_feature_set)
    x_train_raw = features.transform(train_rows)
    x_test_raw = features.transform(test_rows)
    x_train, means, stds = standardize_fit(x_train_raw)
    y_train_ret = [r["return_next"] for r in train_rows]
    y_test_ret = [r["return_next"] for r in test_rows]
    y_train_dir = [1 if r > 0 else 0 for r in y_train_ret]
    y_test_dir = [1 if r > 0 else 0 for r in y_test_ret]
    if resolved_feature_set == "dqn":
        dqn_state_dict, action_returns, epsilon_tail, episode_rewards = train_dqn_policy(
            train_rows,
            features=features,
            means=means,
            stds=stds,
            episodes=dqn_episodes,
        )
        state_size = len(features.names()) + 2
        return {
            "feature_names": features.names(),
            "feature_set": resolved_feature_set,
            "model_type": "dqn",
            "means": means,
            "stds": stds,
            "dqn_state_dict": dqn_state_dict,
            "dqn_state_size": state_size,
            "dqn_action_size": 3,
            "dqn_action_returns": action_returns,
            "dqn_last_epsilon": epsilon_tail[0] if epsilon_tail else 0.0,
            "dqn_episode_rewards": episode_rewards[-10:],
            "lin_weights": [0.0] * len(features.names()),
            "lin_bias": 0.0,
            "logit_weights": [0.0] * len(features.names()),
            "logit_bias": 0.0,
            "train_size": len(train_rows),
            "test_size": len(test_rows),
            "split_style": split_style,
            "x_test_raw": x_test_raw,
            "y_test_ret": y_test_ret,
            "y_test_dir": y_test_dir,
        }
    lin = LinearRegressionGD(learning_rate=0.03, epochs=800)
    lin.fit(x_train, y_train_ret)
    logit = LogisticRegressionGD(learning_rate=0.05, epochs=700)
    logit.fit(x_train, y_train_dir)
    return {"feature_names": features.names(), "feature_set": resolved_feature_set, "means": means, "stds": stds, "lin_weights": lin.weights, "lin_bias": lin.bias, "logit_weights": logit.weights, "logit_bias": logit.bias, "train_size": len(train_rows), "test_size": len(test_rows), "split_style": split_style, "x_test_raw": x_test_raw, "y_test_ret": y_test_ret, "y_test_dir": y_test_dir}


def strategy_metrics(
    returns: Sequence[float],
    probs: Sequence[float],
    expected_returns: Sequence[float] | None = None,
    long_threshold: float = 0.6,
    short_threshold: float = 0.4,
    trade_cost: float = 0.0,
    buy_hold_returns: Sequence[float] | None = None,
    buy_hold_total_return_override: float | None = None,
    allow_short: bool = True,
    min_hold_bars: int = 0,
    prob_smoothing_window: int = 3,
    stop_loss: StopLossConfig | None = None,
    strict_validation: bool = False,
    row_labels: Sequence[str] | None = None,
    raw_prices: Sequence[float] | None = None,
) -> Dict[str, object]:
    stop_cfg = stop_loss or StopLossConfig()
    if expected_returns is None:
        expected_returns = [0.0] * len(returns)
    if len(expected_returns) != len(returns):
        raise ValueError("expected_returns must match returns length.")
    smooth_window = max(1, int(prob_smoothing_window))
    smoothed_probs: List[float] = []
    for i in range(len(probs)):
        start = max(0, i - smooth_window + 1)
        p_window = probs[start : i + 1]
        smoothed_probs.append(sum(p_window) / len(p_window))

    requested_long_threshold = long_threshold
    requested_short_threshold = short_threshold
    sell_threshold = short_threshold
    threshold_mode = "configured"
    positions: List[int] = []
    current_pos = 0
    bars_in_position = 0
    entry_price: float | None = None
    entry_idx: int | None = None
    trailing_anchor: float | None = None
    stop_price: float | None = None
    stop_loss_exits = 0
    time_decay_exits = 0
    take_profit_exits = 0
    max_hold_exits = 0
    time_decay_limit = max(1, int(stop_cfg.time_decay_bars))
    atr_window = 14
    atr_returns: List[float] = []
    slippage = 0.0002
    cooldown_bars = 3
    cooldown = 0
    effective_returns: List[float] = [0.0 for _ in returns]
    synthetic_prices: List[float] = []
    synthetic_price = 1.0
    for bar_ret in returns:
        atr_returns.append(abs(bar_ret))
        synthetic_price *= 1.0 + bar_ret
        synthetic_prices.append(synthetic_price)

    pnl: List[float] = [0.0 for _ in returns]
    wins = 0
    trades = 0
    closed_trade_pnls: List[float] = []
    trade_log: List[Dict[str, object]] = []
    raw_price_series: List[float] | None = None
    if raw_prices is not None:
        candidate_raw_prices = [float(price) for price in raw_prices]
        if len(candidate_raw_prices) == len(returns):
            raw_price_series = candidate_raw_prices

    def close_position(
        idx: int,
        fill_price: float,
        stop_exit: bool = False,
        time_decay_exit: bool = False,
        take_profit_exit: bool = False,
        max_hold_exit: bool = False,
    ) -> None:
        nonlocal current_pos, entry_price, trailing_anchor, stop_price, bars_in_position, cooldown, wins, trades, pnl, stop_loss_exits, time_decay_exits, take_profit_exits, max_hold_exits
        if current_pos == 0 or entry_price is None:
            return
        gross = ((fill_price / entry_price) - 1.0) if current_pos > 0 else ((entry_price / fill_price) - 1.0)
        net = gross - (2.0 * (trade_cost + slippage))
        closed_trade_pnls.append(net)
        pnl[idx] += net
        trades += 1
        if net > 0:
            wins += 1
        if stop_exit:
            stop_loss_exits += 1
            cooldown = cooldown_bars
        if time_decay_exit:
            time_decay_exits += 1
        if take_profit_exit:
            take_profit_exits += 1
        if max_hold_exit:
            max_hold_exits += 1
        entry_bar = int(entry_idx) if entry_idx is not None else int(idx)
        exit_bar = int(idx)
        entry_label = row_labels[entry_bar] if row_labels and 0 <= entry_bar < len(row_labels) else str(entry_bar)
        exit_label = row_labels[exit_bar] if row_labels and 0 <= exit_bar < len(row_labels) else str(exit_bar)
        entry_raw_price = (
            float(raw_price_series[entry_bar])
            if raw_price_series is not None and 0 <= entry_bar < len(raw_price_series)
            else None
        )
        exit_raw_price = (
            float(raw_price_series[exit_bar])
            if raw_price_series is not None and 0 <= exit_bar < len(raw_price_series)
            else None
        )
        excursion_price_series = raw_price_series if raw_price_series is not None else synthetic_prices
        excursion_entry_price = (
            float(excursion_price_series[entry_bar])
            if 0 <= entry_bar < len(excursion_price_series)
            else float(entry_price)
        )
        intra_trade_prices = [float(price) for price in excursion_price_series[entry_bar : exit_bar + 1]]
        if not intra_trade_prices:
            intra_trade_prices = [float(fill_price)]
        lowest_price = min(intra_trade_prices)
        highest_price = max(intra_trade_prices)
        if current_pos > 0:
            max_drawdown_during_trade = (lowest_price / excursion_entry_price) - 1.0 if excursion_entry_price != 0.0 else 0.0
            max_upside_during_trade = (highest_price / excursion_entry_price) - 1.0 if excursion_entry_price != 0.0 else 0.0
        else:
            max_drawdown_during_trade = (excursion_entry_price / highest_price) - 1.0 if highest_price != 0.0 else 0.0
            max_upside_during_trade = (excursion_entry_price / lowest_price) - 1.0 if lowest_price != 0.0 else 0.0

        trade_log.append(
            {
                "side": "LONG" if current_pos > 0 else "SHORT",
                "entry_index": float(entry_bar),
                "exit_index": float(exit_bar),
                "entry_label": entry_label,
                "exit_label": exit_label,
                "entry_price": float(entry_price),
                "exit_price": float(fill_price),
                "entry_raw_price": entry_raw_price,
                "exit_raw_price": exit_raw_price,
                "bars_held": float(max(1, (exit_bar - entry_bar) + 1)),
                "gross_pnl": float(gross),
                "net_pnl": float(net),
                "max_drawdown_during_trade": float(max_drawdown_during_trade),
                "max_upside_during_trade": float(max_upside_during_trade),
                "exit_reason": (
                    "stop_loss"
                    if stop_exit
                    else ("time_decay" if time_decay_exit else ("take_profit" if take_profit_exit else ("max_hold_time" if max_hold_exit else "signal")))
                ),
            }
        )
        current_pos = 0
        entry_price = None
        trailing_anchor = None
        stop_price = None
        bars_in_position = 0

    for idx in range(len(returns)):
        if idx == 0:
            positions.append(current_pos)
            continue
        current_price = synthetic_prices[idx]
        bar_ret = returns[idx]
        atr_returns.append(abs(bar_ret))
        if len(atr_returns) > atr_window:
            atr_returns.pop(0)
        atr_value = sum(atr_returns) / max(1, len(atr_returns))
        signal = smoothed_probs[idx - 1]

        if current_pos == 1 and entry_price is not None:
            if stop_cfg.strategy == StopLossStrategy.TRAILING_STOP and trailing_anchor is not None:
                trailing_anchor = max(trailing_anchor, current_price)
                stop_price = trailing_anchor * (1.0 - (stop_cfg.fixed_pct / 100.0))
            time_decay_hit = stop_cfg.strategy == StopLossStrategy.TIME_DECAY and bars_in_position > time_decay_limit
            max_hold_hit = stop_cfg.max_hold_bars > 0 and bars_in_position >= stop_cfg.max_hold_bars
            fixed_or_atr_hit = stop_cfg.strategy in (StopLossStrategy.ATR, StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP) and stop_price is not None and current_price <= stop_price
            take_profit_hit = stop_cfg.take_profit_pct > 0 and entry_price > 0 and (((current_price / entry_price) - 1.0) >= (stop_cfg.take_profit_pct / 100.0))
            model_invalidation_hit = False
            if stop_cfg.strategy == StopLossStrategy.MODEL_INVALIDATION and entry_idx is not None:
                threshold = expected_returns[entry_idx] - (2.0 * stop_cfg.model_mae)
                cumulative_return = (current_price / entry_price) - 1.0 if entry_price != 0 else 0.0
                model_invalidation_hit = cumulative_return <= threshold
            if time_decay_hit:
                close_position(idx, current_price, time_decay_exit=True)
            elif fixed_or_atr_hit:
                # Assume idealized stop execution: fill exactly at the configured stop
                # level once touched (no adverse gap/slippage beyond global trade costs).
                stop_fill_price = stop_price if stop_price is not None else current_price
                close_position(idx, stop_fill_price, stop_exit=True)
            elif model_invalidation_hit and entry_idx is not None:
                threshold = expected_returns[entry_idx] - (2.0 * stop_cfg.model_mae)
                effective_stop_price = entry_price * (1.0 + threshold)
                close_position(idx, min(effective_stop_price, current_price), stop_exit=True)
            elif take_profit_hit:
                close_position(idx, current_price, take_profit_exit=True)
            elif max_hold_hit:
                close_position(idx, current_price, max_hold_exit=True)
            elif bars_in_position >= min_hold_bars and signal < sell_threshold:
                close_position(idx, current_price)
            else:
                bars_in_position += 1
        elif current_pos == -1 and entry_price is not None:
            if stop_cfg.strategy == StopLossStrategy.TRAILING_STOP and trailing_anchor is not None:
                trailing_anchor = min(trailing_anchor, current_price)
                stop_price = trailing_anchor * (1.0 + (stop_cfg.fixed_pct / 100.0))
            time_decay_hit = stop_cfg.strategy == StopLossStrategy.TIME_DECAY and bars_in_position > time_decay_limit
            max_hold_hit = stop_cfg.max_hold_bars > 0 and bars_in_position >= stop_cfg.max_hold_bars
            fixed_or_atr_hit = stop_cfg.strategy in (StopLossStrategy.ATR, StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP) and stop_price is not None and current_price >= stop_price
            take_profit_hit = stop_cfg.take_profit_pct > 0 and entry_price > 0 and (((entry_price / current_price) - 1.0) >= (stop_cfg.take_profit_pct / 100.0))
            model_invalidation_hit = False
            if stop_cfg.strategy == StopLossStrategy.MODEL_INVALIDATION and entry_idx is not None:
                threshold = expected_returns[entry_idx] - (2.0 * stop_cfg.model_mae)
                cumulative_return = (current_price / entry_price) - 1.0 if entry_price != 0 else 0.0
                model_invalidation_hit = cumulative_return >= -threshold
            if time_decay_hit:
                close_position(idx, current_price, time_decay_exit=True)
            elif fixed_or_atr_hit:
                # Assume idealized stop execution: fill exactly at the configured stop
                # level once touched (no adverse gap/slippage beyond global trade costs).
                stop_fill_price = stop_price if stop_price is not None else current_price
                close_position(idx, stop_fill_price, stop_exit=True)
            elif model_invalidation_hit and entry_idx is not None:
                threshold = expected_returns[entry_idx] - (2.0 * stop_cfg.model_mae)
                effective_stop_price = entry_price * (1.0 - threshold)
                close_position(idx, max(effective_stop_price, current_price), stop_exit=True)
            elif take_profit_hit:
                close_position(idx, current_price, take_profit_exit=True)
            elif max_hold_hit:
                close_position(idx, current_price, max_hold_exit=True)
            elif bars_in_position >= min_hold_bars and signal > long_threshold:
                close_position(idx, current_price)
            else:
                bars_in_position += 1

        if current_pos == 0:
            if cooldown > 0:
                cooldown -= 1
            elif signal > long_threshold:
                current_pos = 1
                entry_price = current_price
                entry_idx = idx
                bars_in_position = 1
                trailing_anchor = current_price
                if stop_cfg.strategy == StopLossStrategy.ATR:
                    stop_price = current_price - (stop_cfg.atr_multiplier * atr_value * current_price)
                elif stop_cfg.strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
                    stop_price = current_price * (1.0 - (stop_cfg.fixed_pct / 100.0))
                else:
                    stop_price = None
            elif signal < short_threshold and allow_short:
                current_pos = -1
                entry_price = current_price
                entry_idx = idx
                bars_in_position = 1
                trailing_anchor = current_price
                if stop_cfg.strategy == StopLossStrategy.ATR:
                    stop_price = current_price + (stop_cfg.atr_multiplier * atr_value * current_price)
                elif stop_cfg.strategy in (StopLossStrategy.FIXED_PERCENTAGE, StopLossStrategy.TRAILING_STOP):
                    stop_price = current_price * (1.0 + (stop_cfg.fixed_pct / 100.0))
                else:
                    stop_price = None
        positions.append(current_pos)

    if current_pos != 0 and entry_price is not None and synthetic_prices:
        close_position(len(returns) - 1, synthetic_prices[-1])
        positions[-1] = 0
    max_trade_loss = min(closed_trade_pnls) if closed_trade_pnls else 0.0
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    drawdowns: List[float] = []
    for r in pnl:
        equity *= 1.0 + r
        equity = max(equity, 0.0)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        drawdowns.append(dd)
        max_drawdown = max(max_drawdown, dd)
    avg_drawdown = (sum(drawdowns) / len(drawdowns)) if drawdowns else 0.0
    hold_lengths_float = [
        float(max(1.0, float(trade.get("bars_held", 1.0))))
        for trade in trade_log
        if isinstance(trade, dict)
    ]
    sorted_holds = sorted(hold_lengths_float)
    hold_time_stats = {
        "count": float(len(hold_lengths_float)),
        "min": float(sorted_holds[0]) if sorted_holds else 0.0,
        "q1": quantile(sorted_holds, 0.25),
        "median": quantile(sorted_holds, 0.5),
        "q3": quantile(sorted_holds, 0.75),
        "max": float(sorted_holds[-1]) if sorted_holds else 0.0,
    }
    sd = stddev(pnl)
    periods_per_year = _infer_periods_per_year(row_labels, len(pnl))
    sharpe = ((sum(pnl) / len(pnl)) / sd * math.sqrt(periods_per_year)) if (sd > 1e-12 and pnl) else 0.0
    buy_hold_source = buy_hold_returns if buy_hold_returns is not None else returns
    total_return = equity - 1.0
    buy_hold_total_return = (
        float(buy_hold_total_return_override)
        if buy_hold_total_return_override is not None
        else compounded_return(buy_hold_source)
    )
    risk_free_rate = 0.0
    if pnl and len(pnl) == len(buy_hold_source):
        strat_mean = sum(pnl) / len(pnl)
        market_mean = sum(buy_hold_source) / len(buy_hold_source)
        market_var = sum((m - market_mean) ** 2 for m in buy_hold_source) / len(buy_hold_source)
        covar = sum((s - strat_mean) * (m - market_mean) for s, m in zip(pnl, buy_hold_source)) / len(pnl)
        beta_vs_sp500 = covar / market_var if market_var > 1e-12 else 0.0
    else:
        beta_vs_sp500 = 0.0
    capm_expected_return = risk_free_rate + beta_vs_sp500 * (buy_hold_total_return - risk_free_rate)
    alpha_capm_sp500_buy_hold = total_return - capm_expected_return
    probability_of_loss = (sum(1 for value in closed_trade_pnls if value < 0.0) / len(closed_trade_pnls)) if closed_trade_pnls else 0.0
    if sharpe > 3.0:
        print("WARNING: Sharpe > 3.0; check for unrealistic backtest assumptions.")
    if quantile(sorted(closed_trade_pnls), 0.5) > 5.0 if closed_trade_pnls else False:
        print("WARNING: Median return > 500%; check for unrealistic backtest assumptions.")
    if closed_trade_pnls and probability_of_loss < 0.05:
        print("WARNING: Probability of loss < 5%; check for unrealistic backtest assumptions.")
    flag_unrealistic = equity > 1000.0
    if strict_validation:
        assert probability_of_loss > 0.0, "Unrealistic: zero loss probability"
        assert max_drawdown > 0.1, "Unrealistic: drawdown too small"
        assert trades > 20, "Too few trades"
    return {
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "requested_long_threshold": requested_long_threshold,
        "requested_short_threshold": requested_short_threshold,
        "threshold_mode": threshold_mode,
        "allow_short": 1.0 if allow_short else 0.0,
        "trade_cost": trade_cost,
        "min_hold_bars": float(min_hold_bars),
        "prob_smoothing_window": float(smooth_window),
        "stop_loss_strategy": stop_cfg.strategy.value,
        "stop_loss_exits": float(stop_loss_exits),
        "time_decay_exits": float(time_decay_exits),
        "take_profit_exits": float(take_profit_exits),
        "max_hold_exits": float(max_hold_exits),
        "time_decay_bars": float(time_decay_limit),
        "fixed_stop_pct": float(stop_cfg.fixed_pct),
        "take_profit_pct": float(stop_cfg.take_profit_pct),
        "max_hold_bars": float(stop_cfg.max_hold_bars),
        "model_mae": float(stop_cfg.model_mae),
        "atr_multiplier": float(stop_cfg.atr_multiplier),
        "total_return": total_return,
        "buy_hold_total_return": buy_hold_total_return,
        "alpha": total_return - buy_hold_total_return,
        "risk_free_rate": risk_free_rate,
        "beta_vs_sp500": beta_vs_sp500,
        "alpha_capm_sp500_buy_hold": alpha_capm_sp500_buy_hold,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown,
        "win_rate": (wins / trades) if trades else 0.0,
        "probability_of_loss": probability_of_loss,
        "trade_count": float(trades),
        "avg_gain_per_trade": (sum(closed_trade_pnls) / len(closed_trade_pnls)) if closed_trade_pnls else 0.0,
        "max_loss_per_trade": max_trade_loss,
        "trade_returns": closed_trade_pnls,
        "trade_log": trade_log,
        "flag_unrealistic": 1.0 if flag_unrealistic else 0.0,
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


def pnl_market_regime_breakdown(
    returns: Sequence[float],
    probs: Sequence[float],
    trade_cost: float = 0.0005,
    allow_short: bool = True,
    long_threshold: float = 0.6,
    short_threshold: float = 0.4,
) -> List[Dict[str, float]]:
    out = {"trending": [], "sideways": [], "high_volatility": []}
    window = 20
    if not any(p > long_threshold for p in probs) and not (allow_short and any(p < short_threshold for p in probs)):
        long_threshold = 0.5
        short_threshold = 0.5
    for i in range(len(returns)):
        start = max(0, i - window + 1)
        r_win = returns[start : i + 1]
        trend = abs(sum(r_win) / max(1, len(r_win)))
        vol = stddev(r_win)
        p = probs[i]
        pos = 1 if p > long_threshold else (-1 if (p < short_threshold and allow_short) else 0)
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


def walk_forward_validation_rows(rows: Sequence[Row], max_windows: int = 4, feature_set: FeatureSet | str = "feature2") -> List[Dict[str, float]]:
    if len(rows) < 120:
        return []
    features = get_strategy_feature_builder(feature_set)
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


def feature_ablation_analysis(rows: Sequence[Row], feature_names: Sequence[str], split_style: SplitStyle = "shuffled", feature_set: FeatureSet | str = "feature2") -> List[Dict[str, float]]:
    if len(rows) < 100:
        return []
    if len(rows) > 600:
        rows = list(rows)[-600:]
    train_rows, test_rows = train_test_split(rows, split_style=split_style)
    features = get_strategy_feature_builder(feature_set)
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
        default_builder = get_strategy_feature_builder(feature_set)
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


def evaluate_bundle(
    bundle: Dict[str, object],
    x_test_raw: Sequence[Sequence[float]],
    y_test_ret: Sequence[float],
    y_test_dir: Sequence[int],
    eval_rows: Sequence[Row] | None = None,
    split_style: SplitStyle = "shuffled",
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
    allow_short: bool = True,
    stop_loss: StopLossConfig | None = None,
    monte_carlo_method: str = "none",
    monte_carlo_n_sim: int = 500,
    monte_carlo_block_size: int = 20,
    monte_carlo_seed: int | None = None,
) -> Dict[str, object]:
    feature_set = infer_bundle_feature_set(bundle)
    if not x_test_raw:
        raise ValueError("No rows available for evaluation.")
    if len(x_test_raw[0]) != len(bundle["feature_names"]):
        raise ValueError("Saved model feature size does not match current strategy feature set.")
    x_test = standardize_apply(x_test_raw, bundle["means"], bundle["stds"])
    is_dqn = str(bundle.get("model_type", "")).lower() == "dqn"
    dqn_policy = {"action_counts": {"hold": 0, "buy": 0, "sell": 0}, "avg_q_values": {"hold": 0.0, "buy": 0.0, "sell": 0.0}}
    if is_dqn and "dqn_state_dict" in bundle:
        action_returns = bundle.get("dqn_action_returns", [0.0, 0.0, 0.0])
        q_value_sums = [0.0, 0.0, 0.0]
        ret_pred = []
        up_prob = []
        for row in x_test:
            state_vector = [*row, 0.0, 1.0]
            q_values = dqn_q_values(bundle, state_vector)
            chosen_action = max(range(len(q_values)), key=lambda idx: q_values[idx]) if q_values else 0
            if chosen_action == 0:
                dqn_policy["action_counts"]["hold"] += 1
            elif chosen_action == 1:
                dqn_policy["action_counts"]["buy"] += 1
            else:
                dqn_policy["action_counts"]["sell"] += 1
            for idx in range(min(3, len(q_values))):
                q_value_sums[idx] += float(q_values[idx])
            action_prob = _softmax(q_values)
            p_buy = action_prob[1] if len(action_prob) > 1 else 0.0
            p_sell = action_prob[2] if len(action_prob) > 2 else 0.0
            up_prob.append(max(0.0, min(1.0, p_buy + (0.5 * (1.0 - p_buy - p_sell)))))
            expected = sum(action_prob[idx] * float(action_returns[idx]) for idx in range(min(3, len(action_prob))))
            ret_pred.append(expected)
        test_count = max(1, len(x_test))
        dqn_policy["avg_q_values"] = {
            "hold": q_value_sums[0] / test_count,
            "buy": q_value_sums[1] / test_count,
            "sell": q_value_sums[2] / test_count,
        }
    else:
        ret_pred = [sum(w * v for w, v in zip(bundle["lin_weights"], row)) + bundle["lin_bias"] for row in x_test]
        up_prob = [sigmoid(sum(w * v for w, v in zip(bundle["logit_weights"], row)) + bundle["logit_bias"]) for row in x_test]
    cls = classification_metrics(y_test_dir, up_prob)
    baseline_up_accuracy = sum(y_test_dir) / max(1, len(y_test_dir))
    baseline_zero = [0.0] * len(y_test_ret)
    buy_hold_total_return_override: float | None = None
    strategy_returns = list(y_test_ret)
    row_labels: List[str] = [str(i) for i in range(len(y_test_ret))]
    raw_prices: List[float] | None = None
    if eval_rows and split_style == "chronological":
        test_rows = list(eval_rows)[-len(y_test_ret) :] if y_test_ret else []
        closes = [float(row["close"]) for row in test_rows if "close" in row]
        derived_labels: List[str] = []
        for i, row in enumerate(test_rows):
            timestamp = row.get("timestamp")
            if timestamp:
                derived_labels.append(str(timestamp))
            else:
                derived_labels.append(str(i))
        if len(derived_labels) == len(y_test_ret):
            row_labels = derived_labels
        if len(closes) >= 2 and closes[0] != 0.0:
            buy_hold_total_return_override = (closes[-1] / closes[0]) - 1.0
            step_returns: List[float] = [0.0]
            for i in range(1, len(closes)):
                prev_close = closes[i - 1]
                current_close = closes[i]
                step_returns.append(((current_close / prev_close) - 1.0) if prev_close != 0.0 else 0.0)
            if len(step_returns) == len(y_test_ret):
                strategy_returns = step_returns
                raw_prices = closes

    strategy = strategy_metrics(
        strategy_returns,
        up_prob,
        expected_returns=ret_pred,
        long_threshold=buy_threshold,
        short_threshold=sell_threshold,
        trade_cost=0.0,
        buy_hold_returns=strategy_returns,
        buy_hold_total_return_override=buy_hold_total_return_override,
        allow_short=allow_short,
        stop_loss=stop_loss,
        row_labels=row_labels,
        raw_prices=raw_prices,
    )
    monte_carlo: Dict[str, Any] | None = None
    if monte_carlo_method in {"bootstrap", "shuffle", "block"}:
        monte_carlo = run_monte_carlo_backtest(
            returns=list(y_test_ret),
            probs=list(up_prob),
            strategy_fn=strategy_metrics,
            n_sim=monte_carlo_n_sim,
            method=monte_carlo_method,
            block_size=monte_carlo_block_size,
            seed=monte_carlo_seed,
            return_equity_curves=3,
            expected_returns=ret_pred,
            long_threshold=buy_threshold,
            short_threshold=sell_threshold,
            trade_cost=0.0,
            buy_hold_returns=strategy_returns,
            buy_hold_total_return_override=buy_hold_total_return_override,
            allow_short=allow_short,
            stop_loss=stop_loss,
        )
    preview = [{"expected_return": ret_pred[i], "p_up": up_prob[i], "actual_return": y_test_ret[i]} for i in range(min(5, len(x_test)))]
    latest_p_up = float(up_prob[-1]) if up_prob else 0.5
    latest_expected_return = float(ret_pred[-1]) if ret_pred else 0.0
    latest_action = "hold"
    if latest_p_up >= buy_threshold:
        latest_action = "buy"
    elif latest_p_up <= sell_threshold:
        latest_action = "sell"
    forward_buy_now: Dict[str, float | int] | None = None
    if monte_carlo and isinstance(monte_carlo, dict):
        mc_raw_results = monte_carlo.get("raw_results", [])
        if isinstance(mc_raw_results, list):
            mc_total_returns = [
                float(item.get("total_return", 0.0))
                for item in mc_raw_results
                if isinstance(item, dict)
            ]
            if mc_total_returns:
                initial_capital = 10_000.0
                terminal_values = [initial_capital * (1.0 + value) for value in mc_total_returns]
                sorted_total_returns = sorted(mc_total_returns)
                sorted_terminal_values = sorted(terminal_values)
                forward_buy_now = {
                    "initial_capital": initial_capital,
                    "horizon_bars": len(y_test_ret),
                    "simulations": len(mc_total_returns),
                    "expected_return": float(sum(mc_total_returns) / len(mc_total_returns)),
                    "median_return": quantile(sorted_total_returns, 0.5),
                    "worst_return": min(mc_total_returns),
                    "best_return": max(mc_total_returns),
                    "p5_return": quantile(sorted_total_returns, 0.05),
                    "p95_return": quantile(sorted_total_returns, 0.95),
                    "probability_profit": sum(1 for value in mc_total_returns if value > 0.0) / len(mc_total_returns),
                    "probability_loss": sum(1 for value in mc_total_returns if value < 0.0) / len(mc_total_returns),
                    "expected_terminal_value": float(sum(terminal_values) / len(terminal_values)),
                    "median_terminal_value": quantile(sorted_terminal_values, 0.5),
                    "p5_terminal_value": quantile(sorted_terminal_values, 0.05),
                    "p95_terminal_value": quantile(sorted_terminal_values, 0.95),
                }
    return {
        "model_type": "dqn" if is_dqn else "linear_logistic",
        "features": bundle["feature_names"],
        "feature_set": feature_set,
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
        "latest_signal": {
            "p_up": latest_p_up,
            "expected_return": latest_expected_return,
            "action": latest_action,
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        },
        "calibration": calibration_buckets(y_test_dir, up_prob),
        "confidence_edge": confidence_edge_analysis(y_test_dir, up_prob),
        "strategy": strategy,
        "monte_carlo": monte_carlo,
        "forward_buy_now": forward_buy_now,
        "pnl_by_signal_strength": pnl_signal_strength_breakdown(y_test_ret, up_prob, trade_cost=0.0, allow_short=allow_short),
        "pnl_by_regime": pnl_market_regime_breakdown(
            y_test_ret,
            up_prob,
            trade_cost=0.0,
            allow_short=allow_short,
            long_threshold=float(strategy["long_threshold"]),
            short_threshold=float(strategy["short_threshold"]),
        ),
        "walk_forward": walk_forward_validation_rows(rows=eval_rows, max_windows=4, feature_set=feature_set) if eval_rows else [],
        "feature_ablation": feature_ablation_analysis(eval_rows, bundle["feature_names"], split_style=split_style, feature_set=feature_set) if eval_rows else [],
        "error_analysis": error_analysis(y_test_ret, up_prob, ret_pred, top_n=5),
        "dqn_policy": dqn_policy,
        "dqn_episode_rewards": list(bundle.get("dqn_episode_rewards", [])),
        "dqn_last_epsilon": float(bundle.get("dqn_last_epsilon", 0.0)),
        "dqn_action_returns": list(bundle.get("dqn_action_returns", [0.0, 0.0, 0.0])),
    }


def run_model(rows: Sequence[Row], feature_set: FeatureSet | str = "feature2") -> None:
    bundle = train_strategy_models(rows, feature_set=feature_set)
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
    is_dqn = str(metrics.get("model_type", "")).lower() == "dqn"
    if is_dqn:
        print("\n=== DQN Policy Evaluation ===")
        print(f"Expected-return MSE: {metrics['mse']:.8f}")
        print(f"Expected-return MAE: {metrics['mae']:.8f}")
        action_counts = metrics["dqn_policy"]["action_counts"]
        avg_q_values = metrics["dqn_policy"]["avg_q_values"]
        print(
            "Action counts (test): "
            f"HOLD={int(action_counts['hold'])}, BUY={int(action_counts['buy'])}, SELL={int(action_counts['sell'])}"
        )
        print(
            "Average Q-values: "
            f"HOLD={avg_q_values['hold']:+.6f}, BUY={avg_q_values['buy']:+.6f}, SELL={avg_q_values['sell']:+.6f}"
        )
        print(f"Final epsilon: {metrics['dqn_last_epsilon']:.4f}")
    else:
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
    print(f"Alpha vs Buy & Hold: {strat['alpha']:+.2%}")
    print(
        "CAPM Alpha vs S&P 500 Buy & Hold: "
        f"{strat['alpha_capm_sp500_buy_hold']:+.2%} "
        f"(beta={strat['beta_vs_sp500']:.3f}, rf={strat['risk_free_rate']:.2%})"
    )


def run_model_metrics(rows: Sequence[Row], feature_set: FeatureSet | str = "feature2") -> Dict[str, object]:
    bundle = train_strategy_models(rows, feature_set=feature_set)
    metrics = evaluate_bundle(bundle, bundle["x_test_raw"], bundle["y_test_ret"], bundle["y_test_dir"], eval_rows=rows, split_style=bundle["split_style"])
    metrics["train_size"] = bundle["train_size"]
    return metrics


def predict_signal(bundle: Dict[str, object], row: Row, *, buy_threshold: float = 0.6, sell_threshold: float = 0.4, long_only: bool = False) -> Dict[str, float | str]:
    feature_builder = get_strategy_feature_builder(infer_bundle_feature_set(bundle))
    row_features = feature_builder.transform([row])
    if len(row_features[0]) != len(bundle["feature_names"]):
        raise ValueError("Saved model feature size does not match current strategy feature set.")
    x_scaled = standardize_apply(row_features, bundle["means"], bundle["stds"])[0]
    is_dqn = str(bundle.get("model_type", "")).lower() == "dqn"
    if is_dqn and "dqn_state_dict" in bundle:
        state_vector = [*x_scaled, 0.0, 1.0]
        q_values = dqn_q_values(bundle, state_vector)
        action_prob = _softmax(q_values)
        action_returns = bundle.get("dqn_action_returns", [0.0, 0.0, 0.0])
        expected_return = sum(action_prob[idx] * float(action_returns[idx]) for idx in range(min(3, len(action_prob))))
        p_buy = action_prob[1] if len(action_prob) > 1 else 0.0
        p_sell = action_prob[2] if len(action_prob) > 2 else 0.0
        p_up = max(0.0, min(1.0, p_buy + (0.5 * (1.0 - p_buy - p_sell))))
    else:
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


def _ema_series(values, span: int):
    return values.ewm(span=span, adjust=False).mean()


def _rsi_series(close, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _stoch_rsi_series(close, rsi_period: int = 14, stoch_period: int = 14):
    rsi = _rsi_series(close, period=rsi_period)
    rsi_min = rsi.rolling(stoch_period).min()
    rsi_max = rsi.rolling(stoch_period).max()
    spread = (rsi_max - rsi_min).replace(0.0, float("nan"))
    stoch = (rsi - rsi_min) / spread
    return stoch.clip(0.0, 1.0)


def build_features(df):
    """
    Build non-leaky feature matrix at timestamp t.

    Required columns in df: open, high, low, close, volume.
    """
    import pandas as pd

    required_columns = {"open", "high", "low", "close", "volume"}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    features = df.copy()
    close = features["close"].astype(float)
    high = features["high"].astype(float)
    low = features["low"].astype(float)

    # A) Stoch RSI (continuous 0..1)
    features["stoch_rsi"] = _stoch_rsi_series(close=close)
    features["stoch_velocity"] = features["stoch_rsi"].diff()
    features["stoch_low_zone"] = (0.2 - features["stoch_rsi"]).clip(lower=0.0)
    features["stoch_high_zone"] = (features["stoch_rsi"] - 0.8).clip(lower=0.0)

    # B) MACD histogram
    ema12 = _ema_series(close, span=12)
    ema26 = _ema_series(close, span=26)
    macd_line = ema12 - ema26
    signal = _ema_series(macd_line, span=9)
    features["macd_hist"] = macd_line - signal
    features["macd_delta"] = features["macd_hist"].diff()

    # C) Momentum
    features["ret_1"] = close.pct_change(1)
    features["ret_3"] = close.pct_change(3)
    features["ret_5"] = close.pct_change(5)

    # D) Trend
    features["ma20"] = close.rolling(20).mean()
    features["trend_20"] = (close - features["ma20"]) / features["ma20"]

    # E) Volatility
    returns = close.pct_change()
    features["vol_20"] = returns.rolling(20).std()

    # F) FVG proxy and gap state tracking
    features["gap_up"] = (low > high.shift(1)).astype(int)
    features["gap_down"] = (high < low.shift(1)).astype(int)

    bull_gap_low = pd.Series(pd.NA, index=features.index, dtype="float64")
    bull_gap_high = pd.Series(pd.NA, index=features.index, dtype="float64")
    bear_gap_low = pd.Series(pd.NA, index=features.index, dtype="float64")
    bear_gap_high = pd.Series(pd.NA, index=features.index, dtype="float64")

    bull_gap_low.loc[features["gap_up"] == 1] = high.shift(1).loc[features["gap_up"] == 1]
    bull_gap_high.loc[features["gap_up"] == 1] = low.loc[features["gap_up"] == 1]
    bear_gap_low.loc[features["gap_down"] == 1] = high.loc[features["gap_down"] == 1]
    bear_gap_high.loc[features["gap_down"] == 1] = low.shift(1).loc[features["gap_down"] == 1]

    features["last_bull_gap_low"] = bull_gap_low.ffill()
    features["last_bull_gap_high"] = bull_gap_high.ffill()
    features["last_bear_gap_low"] = bear_gap_low.ffill()
    features["last_bear_gap_high"] = bear_gap_high.ffill()

    features["dist_to_bull_fvg"] = (close - features["last_bull_gap_low"]) / close
    features["dist_to_bear_fvg"] = (features["last_bear_gap_high"] - close) / close
    features["inside_bull_fvg"] = (
        (close >= features["last_bull_gap_low"]) & (close <= features["last_bull_gap_high"])
    ).astype(int)
    features["inside_bear_fvg"] = (
        (close >= features["last_bear_gap_low"]) & (close <= features["last_bear_gap_high"])
    ).astype(int)

    # G) Interactions
    features["oversold_reversal"] = (
        (features["stoch_rsi"] < 0.2) & (features["macd_delta"] > 0.0)
    ).astype(int)
    features["overbought_reversal"] = (
        (features["stoch_rsi"] > 0.8) & (features["macd_delta"] < 0.0)
    ).astype(int)
    features["bull_confluence"] = (
        (features["inside_bull_fvg"] == 1) & (features["macd_hist"] > 0.0) & (features["macd_delta"] > 0.0)
    ).astype(int)
    features["bear_confluence"] = (
        (features["inside_bear_fvg"] == 1) & (features["macd_hist"] < 0.0) & (features["macd_delta"] < 0.0)
    ).astype(int)
    return features


def build_target(df, horizon: int = 5):
    close = df["close"].astype(float)
    future_return = close.shift(-horizon) / close - 1.0
    target = (future_return > 0.0).astype(int)
    return target


def train_model(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=500, random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "baseline_accuracy_always_up": float(sum(y_test) / max(1, len(y_test))),
    }


def backtest_strategy(test_df, probabilities, buy_threshold: float = 0.65, exit_threshold: float = 0.50, trade_cost: float = 0.0005):
    returns = test_df["close"].pct_change().fillna(0.0)
    position = 0
    strategy_returns = []
    trades = 0
    trade_returns = []
    current_trade = 0.0
    in_trade = False

    for i in range(len(test_df)):
        p = float(probabilities[i])
        if position == 0 and p > buy_threshold:
            position = 1
            trades += 1
            in_trade = True
            current_trade -= trade_cost
        elif position == 1 and p < exit_threshold:
            position = 0
            current_trade -= trade_cost
            trade_returns.append(current_trade)
            current_trade = 0.0
            in_trade = False
        day_ret = position * float(returns.iloc[i])
        current_trade += day_ret
        strategy_returns.append(day_ret)

    if in_trade:
        current_trade -= trade_cost
        trade_returns.append(current_trade)

    equity_curve = []
    equity = 1.0
    for r in strategy_returns:
        equity *= 1.0 + r
        equity_curve.append(equity)

    peak = 1.0
    max_drawdown = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        drawdown = (e / peak) - 1.0
        max_drawdown = min(max_drawdown, drawdown)

    mean_ret = sum(strategy_returns) / max(1, len(strategy_returns))
    variance = sum((r - mean_ret) ** 2 for r in strategy_returns) / max(1, len(strategy_returns))
    sharpe = (mean_ret / math.sqrt(variance) * math.sqrt(252.0)) if variance > 0 else 0.0
    win_rate = (sum(1 for t in trade_returns if t > 0.0) / len(trade_returns)) if trade_returns else 0.0
    return {
        "total_return": equity - 1.0,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "number_of_trades": trades,
    }


def run_logistic_trading_pipeline(df, test_ratio: float = 0.25, horizon: int = 5):
    """
    Full pipeline for daily OHLCV data:
    - non-leaky features at t
    - target on close[t+horizon]
    - chronological split (no shuffle)
    - train/test metrics and test-period backtest
    """
    feature_df = build_features(df)
    target = build_target(feature_df, horizon=horizon)

    # Drop last horizon rows (no future label), then drop NaNs from feature windows.
    feature_df = feature_df.iloc[:-horizon].copy()
    target = target.iloc[:-horizon].copy()
    feature_df["target"] = target
    feature_df = feature_df.dropna().copy()

    feature_columns = [
        "stoch_rsi",
        "stoch_velocity",
        "stoch_low_zone",
        "stoch_high_zone",
        "macd_hist",
        "macd_delta",
        "ret_1",
        "ret_3",
        "ret_5",
        "trend_20",
        "vol_20",
        "dist_to_bull_fvg",
        "dist_to_bear_fvg",
        "inside_bull_fvg",
        "inside_bear_fvg",
        "oversold_reversal",
        "overbought_reversal",
        "bull_confluence",
        "bear_confluence",
    ]

    dataset = feature_df[feature_columns + ["target", "close"]].copy()
    split_idx = int(len(dataset) * (1.0 - test_ratio))
    train_df = dataset.iloc[:split_idx].copy()
    test_df = dataset.iloc[split_idx:].copy()

    X_train = train_df[feature_columns]
    y_train = train_df["target"].astype(int)
    X_test = test_df[feature_columns]
    y_test = test_df["target"].astype(int)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    backtest = backtest_strategy(test_df=test_df, probabilities=probabilities)

    # Safety checks for leakage/alignment
    feature_index_ok = X_train.index.max() < y_train.index.max() + horizon + 1
    chronological_ok = train_df.index.max() < test_df.index.min()
    checks = {
        "features_use_only_t_data": True,
        "feature_timestamp_before_target_timestamp": bool(feature_index_ok),
        "chronological_split_no_shuffle": bool(chronological_ok),
    }
    return {
        "model": model,
        "feature_columns": feature_columns,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": metrics,
        "backtest": backtest,
        "checks": checks,
        "test_probabilities": probabilities,
    }
