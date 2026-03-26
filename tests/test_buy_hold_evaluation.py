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
