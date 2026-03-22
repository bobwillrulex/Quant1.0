import unittest
from unittest.mock import patch

try:
    from main import create_app
except ModuleNotFoundError:
    create_app = None


class StopLossUITests(unittest.TestCase):
    def setUp(self):
        if create_app is None:
            self.skipTest("Flask is not installed in this environment.")
        try:
            self.app = create_app().test_client()
        except ModuleNotFoundError:
            self.skipTest("Flask is not installed in this environment.")

    def test_stop_loss_dropdown_and_fixed_input_present(self):
        response = self.app.get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Stop Loss Strategy", html)
        self.assertIn('option value="none"', html)
        self.assertIn('option value="atr"', html)
        self.assertIn('option value="model_invalidation"', html)
        self.assertIn('option value="time_decay"', html)
        self.assertIn('option value="fixed_percentage"', html)
        self.assertIn('id="fixedStopLossWrap"', html)
        self.assertIn("toggleFixedStopField", html)
        self.assertIn('step="any" name="fixed_stop_pct"', html)
        self.assertIn('name="present_stop_loss_strategy"', html)
        self.assertIn('id="presentFixedStopLossWrap"', html)
        self.assertIn("togglePresentFixedStopField", html)
        self.assertIn("Stop Strategy", html)
        self.assertIn("Stop Price", html)

    def test_manage_models_disables_fixed_stop_input_when_not_fixed_strategy(self):
        response = self.app.get("/manage-models")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="cfgFixedStopPct"', html)
        self.assertIn("fixedStopInput.disabled = !isFixed;", html)

    @patch("main.fetch_yahoo_rows")
    @patch("main.evaluate_bundle")
    @patch("main.train_strategy_models")
    def test_train_submit_works_when_fixed_pct_blank_and_strategy_none(self, train_mock, eval_mock, fetch_mock):
        fetch_mock.return_value = [
            {"return_next": 0.01},
            {"return_next": -0.01},
            {"return_next": 0.02},
            {"return_next": 0.01},
        ]
        train_mock.return_value = {
            "x_test_raw": [[0.1], [0.2]],
            "y_test_ret": [0.01, -0.01],
            "y_test_dir": [1, 0],
            "train_size": 2,
            "test_size": 2,
            "split_style": "shuffled",
        }
        eval_mock.return_value = {
            "accuracy": 0.5,
            "mse": 0.01,
            "strategy": {
                "total_return": 0.01,
                "trade_count": 1,
                "win_rate": 1.0,
                "max_drawdown": 0.0,
                "sharpe_like": 1.0,
                "profit_factor": 1.0,
                "avg_return_per_trade": 0.01,
                "stop_exit_count": 0,
                "model_exit_count": 0,
                "sell_exit_count": 0,
                "hold_streak_stats": {"count": 0, "min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0},
                "equity_curve": [1.0, 1.01],
            },
            "preview": [],
            "lin_weights": [],
            "logit_weights": [],
            "calibration": [],
            "pnl_by_signal_strength": [],
            "pnl_by_regime": [],
            "confidence_edge": {"p_gt_0.6": {"count": 0, "accuracy": 0.0}, "p_gt_0.7": {"count": 0, "accuracy": 0.0}},
            "return_vs_pred_corr": 0.0,
        }
        response = self.app.post(
            "/",
            data={
                "mode": "train",
                "train_action": "train",
                "ticker": "AAPL",
                "interval": "1d",
                "rows": "50",
                "split_style": "shuffled",
                "buy_threshold": "0.6",
                "sell_threshold": "0.4",
                "selected_model": "__new__",
                "model_name": "",
                "stop_loss_strategy": "none",
                "fixed_stop_pct": "",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Model Evaluation Summary", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
