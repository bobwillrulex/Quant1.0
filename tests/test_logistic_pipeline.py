import math
import random

import pytest

from quant.ml import build_features, build_target, run_logistic_trading_pipeline

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")


def _sample_ohlcv(n: int = 300) -> pd.DataFrame:
    random.seed(7)
    close = []
    price = 100.0
    for _ in range(n):
        price *= math.exp(0.0005 + random.gauss(0.0, 0.01))
        close.append(price)
    open_ = [c * (1.0 + random.gauss(0.0, 0.001)) for c in close]
    high = [max(o, c) * (1.0 + abs(random.gauss(0.0, 0.002))) for o, c in zip(open_, close)]
    low = [min(o, c) * (1.0 - abs(random.gauss(0.0, 0.002))) for o, c in zip(open_, close)]
    volume = [float(random.randint(1_000, 5_000)) for _ in range(n)]
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_build_target_drops_last_horizon_labels() -> None:
    df = _sample_ohlcv(40)
    y = build_target(df, horizon=5)
    assert len(y) == len(df)
    assert y.iloc[-5:].eq(0).all()


def test_run_logistic_trading_pipeline_outputs_metrics_and_checks() -> None:
    df = _sample_ohlcv(320)
    result = run_logistic_trading_pipeline(df, test_ratio=0.25, horizon=5)

    assert result["train_rows"] > 0
    assert result["test_rows"] > 0
    assert "accuracy" in result["metrics"]
    assert "confusion_matrix" in result["metrics"]
    assert "total_return" in result["backtest"]
    assert result["checks"]["features_use_only_t_data"] is True
    assert result["checks"]["chronological_split_no_shuffle"] is True


def test_build_features_uses_required_columns() -> None:
    df = _sample_ohlcv(100)
    features = build_features(df)
    for name in ["stoch_rsi", "macd_hist", "vol_20", "bull_confluence", "bear_confluence"]:
        assert name in features.columns
