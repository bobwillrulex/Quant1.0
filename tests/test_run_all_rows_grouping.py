from unittest.mock import patch

try:
    from main import build_run_all_rows
except ModuleNotFoundError:
    build_run_all_rows = None


def test_build_run_all_rows_groups_by_interval() -> None:
    if build_run_all_rows is None:
        return
    mock_rows = [
        {
            "model_name": "m_daily",
            "ticker": "AAPL",
            "interval": "1d",
            "row_count": 250,
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
            "stop_strategy": "none",
            "expected_return": 0.01,
            "p_up": 0.55,
            "stop_price": None,
            "action": "BUY",
            "provider_notice": "",
            "error": "",
        },
        {
            "model_name": "m_5m",
            "ticker": "MSFT",
            "interval": "5m",
            "row_count": 300,
            "buy_threshold": 0.6,
            "sell_threshold": 0.4,
            "stop_strategy": "none",
            "expected_return": 0.01,
            "p_up": 0.55,
            "stop_price": None,
            "action": "BUY",
            "provider_notice": "",
            "error": "",
        },
    ]

    with patch("main.evaluate_run_all_models", return_value=mock_rows):
        html = build_run_all_rows({}, {}, mode="stocks", long_only=False)

    assert "5 min presets" in html
    assert "1d presets" in html
    assert html.index("5 min presets") < html.index("1d presets")
