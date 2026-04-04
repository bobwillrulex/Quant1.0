from quant.ml import evaluate_bundle


def _minimal_bundle(feature_count: int = 1) -> dict[str, object]:
    return {
        "feature_names": [f"f{i}" for i in range(feature_count)],
        "means": [0.0] * feature_count,
        "stds": [1.0] * feature_count,
        "lin_weights": [0.0] * feature_count,
        "lin_bias": 0.0,
        "logit_weights": [0.0] * feature_count,
        "logit_bias": 0.0,
    }


def test_strategy_uses_step_returns_for_chronological_eval_rows() -> None:
    bundle = _minimal_bundle()
    bundle["logit_bias"] = 10.0  # always long
    x_test_raw = [[0.0], [0.0], [0.0]]
    y_test_ret = [1.0, 1.0, 1.0]  # overlapping horizon returns (misleading for per-bar compounding)
    y_test_dir = [1, 1, 1]
    eval_rows = [
        {"close": 100.0, "return_next": 1.0},
        {"close": 50.0, "return_next": 1.0},
        {"close": 25.0, "return_next": 1.0},
    ]

    metrics = evaluate_bundle(
        bundle=bundle,
        x_test_raw=x_test_raw,
        y_test_ret=y_test_ret,
        y_test_dir=y_test_dir,
        eval_rows=eval_rows,
        split_style="chronological",
        allow_short=False,
    )

    total_return = float(metrics["strategy"]["total_return"])
    assert total_return < 0.0


def test_buy_hold_uses_close_to_close_for_chronological_eval_rows() -> None:
    bundle = _minimal_bundle()
    x_test_raw = [[0.0], [0.0], [0.0]]
    y_test_ret = [0.10, 0.10, 0.10]  # would compound to +33.1%
    y_test_dir = [1, 1, 1]
    eval_rows = [
        {"close": 260.0, "return_next": 0.10},
        {"close": 240.0, "return_next": 0.10},
        {"close": 220.0, "return_next": 0.10},
    ]

    metrics = evaluate_bundle(
        bundle=bundle,
        x_test_raw=x_test_raw,
        y_test_ret=y_test_ret,
        y_test_dir=y_test_dir,
        eval_rows=eval_rows,
        split_style="chronological",
        allow_short=False,
    )

    buy_hold = float(metrics["strategy"]["buy_hold_total_return"])
    expected = (220.0 / 260.0) - 1.0
    assert abs(buy_hold - expected) < 1e-12


def test_buy_hold_falls_back_to_returns_when_not_chronological() -> None:
    bundle = _minimal_bundle()
    x_test_raw = [[0.0], [0.0]]
    y_test_ret = [0.1, -0.05]
    y_test_dir = [1, 0]
    eval_rows = [{"close": 100.0}, {"close": 50.0}]

    metrics = evaluate_bundle(
        bundle=bundle,
        x_test_raw=x_test_raw,
        y_test_ret=y_test_ret,
        y_test_dir=y_test_dir,
        eval_rows=eval_rows,
        split_style="shuffled",
        allow_short=False,
    )

    buy_hold = float(metrics["strategy"]["buy_hold_total_return"])
    expected = (1.1 * 0.95) - 1.0
    assert abs(buy_hold - expected) < 1e-12


def test_sharpe_annualization_respects_non_daily_interval() -> None:
    bundle = _minimal_bundle()
    bundle["logit_bias"] = 8.0  # keep strategy in long mode
    x_test_raw = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    y_test_ret = [0.02, -0.01, 0.015, -0.005, 0.01, -0.002]
    y_test_dir = [1 if value > 0 else 0 for value in y_test_ret]

    daily_metrics = evaluate_bundle(
        bundle=bundle,
        x_test_raw=x_test_raw,
        y_test_ret=y_test_ret,
        y_test_dir=y_test_dir,
        split_style="shuffled",
        allow_short=False,
        interval="1d",
    )
    hourly_metrics = evaluate_bundle(
        bundle=bundle,
        x_test_raw=x_test_raw,
        y_test_ret=y_test_ret,
        y_test_dir=y_test_dir,
        split_style="shuffled",
        allow_short=False,
        interval="1h",
    )

    daily_sharpe = float(daily_metrics["strategy"]["sharpe"])
    hourly_sharpe = float(hourly_metrics["strategy"]["sharpe"])
    expected_ratio = (6.5) ** 0.5
    assert daily_sharpe != 0.0
    assert abs((hourly_sharpe / daily_sharpe) - expected_ratio) < 1e-6
