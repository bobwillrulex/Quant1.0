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
        self.assertIn('option value="trailing_stop"', html)
        self.assertIn('option value="vwap_intraday_reversion"', html)
        self.assertIn('option value="vwap_intraday_momentum"', html)
        self.assertIn('option value="vwap_intraday_5m_session"', html)
        self.assertIn('option value="vwap_breakout_reversion_regime"', html)
        self.assertIn('option value="open15_trend_momentum_daytrade"', html)
        self.assertIn('option value="open15_dual_breakout_daytrade"', html)
        self.assertIn('option value="open15_dual_breakout_daytrade_plus"', html)
        self.assertIn('option value="open15_dual_breakout_daytrade_scalp"', html)
        self.assertIn('id="featurePipelineHint"', html)
        self.assertIn("featurePipelineDescriptions", html)
        self.assertIn("vwap_intraday_5m_session: \"5m session reset VWAP", html)
        self.assertIn("open15_trend_momentum_daytrade: \"Built for 5m day trading after first 15m", html)
        self.assertIn("open15_dual_breakout_daytrade: \"Built for 5m day trading after first 15m", html)
        self.assertIn("open15_dual_breakout_daytrade_plus: \"Built for 5m day trading after first 15m", html)
        self.assertIn("open15_dual_breakout_daytrade_scalp: \"Built for 5m day trading after first 15m", html)
        self.assertIn('id="fixedStopLossWrap"', html)
        self.assertIn("toggleFixedStopField", html)
        self.assertIn('step="any" name="fixed_stop_pct"', html)
        self.assertIn('name="present_stop_loss_strategy"', html)
        self.assertIn('id="presentFixedStopLossWrap"', html)
        self.assertIn("togglePresentFixedStopField", html)
        self.assertIn("Stop Strategy", html)
        self.assertIn("Stop Price", html)
        self.assertIn("split(/[,\\\\n]/)", html)
        self.assertIn('value="evaluate_historical"', html)

    def test_manage_models_disables_fixed_stop_input_when_not_fixed_strategy(self):
        response = self.app.get("/manage-models")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="cfgFixedStopPct"', html)
        self.assertIn('id="cfgTakeProfitPct"', html)
        self.assertIn('id="cfgMaxHoldBars"', html)
        self.assertIn("fixedStopInput.disabled = !isFixed;", html)
        self.assertIn("Edit all model presets", html)
        self.assertIn('value="save_all_configs"', html)
        self.assertIn("openAllSettings", html)
        self.assertIn("Leave fields blank to keep existing values unchanged.", html)
        self.assertIn('id="allTicker"', html)
        self.assertIn('id="allRows"', html)
        self.assertIn('id="allTakeProfitPct"', html)
        self.assertIn('id="allMaxHoldBars"', html)
        self.assertIn('option value="">No change</option>', html)
        self.assertIn('document.getElementById("allTicker").value = "";', html)

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

    @patch("main.list_saved_models")
    @patch("main.save_model_configs")
    @patch("main.load_model_configs")
    @patch("main.save_model_bundle")
    @patch("main.fetch_market_rows")
    @patch("main.evaluate_bundle")
    @patch("main.train_strategy_models")
    def test_blank_model_name_defaults_to_ticker_rows_and_prediction_horizon(
        self,
        train_mock,
        eval_mock,
        fetch_market_rows_mock,
        save_bundle_mock,
        load_configs_mock,
        save_configs_mock,
        list_models_mock,
    ):
        fetch_market_rows_mock.return_value = (
            [
                {"return_next": 0.01},
                {"return_next": -0.01},
                {"return_next": 0.02},
                {"return_next": 0.01},
            ],
            None,
        )
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
        load_configs_mock.return_value = {}
        list_models_mock.return_value = []

        response = self.app.post(
            "/",
            data={
                "mode": "train",
                "train_action": "train",
                "ticker": "AAPL",
                "interval": "1d",
                "rows": "4000",
                "split_style": "shuffled",
                "feature_set": "fvg2",
                "buy_threshold": "0.6",
                "sell_threshold": "0.4",
                "selected_model": "__new__",
                "model_name": "",
                "stop_loss_strategy": "none",
                "fixed_stop_pct": "",
            },
        )

        self.assertEqual(response.status_code, 200)
        save_bundle_mock.assert_called_once()
        self.assertEqual(save_bundle_mock.call_args[0][1], "aapl_1d_4000_5")
        saved_payload = save_configs_mock.call_args[0][1]
        self.assertIn("aapl_1d_4000_5", saved_payload)

    @patch("main.save_model_configs")
    @patch("main.list_saved_models")
    @patch("main.load_model_configs")
    def test_save_all_configs_updates_only_filled_values(self, load_configs_mock, list_models_mock, save_configs_mock):
        list_models_mock.return_value = ["model_a", "model_b"]
        load_configs_mock.return_value = {
            "model_a": {
                "ticker": "AAPL",
                "interval": "1d",
                "rows": 250,
                "include_in_run_all": True,
                "buy_threshold": 0.6,
                "sell_threshold": 0.4,
                "stop_loss_strategy": "none",
                "fixed_stop_pct": 2.0,
            },
            "model_b": {
                "ticker": "MSFT",
                "interval": "1h",
                "rows": 300,
                "include_in_run_all": False,
                "buy_threshold": 0.65,
                "sell_threshold": 0.35,
                "stop_loss_strategy": "none",
                "fixed_stop_pct": 2.5,
            },
        }
        response = self.app.post(
            "/manage-models",
            data={
                "action": "save_all_configs",
                "ticker": "",
                "interval": "",
                "rows": "",
                "buy_threshold": "",
                "sell_threshold": "",
                "stop_loss_strategy": "atr",
                "fixed_stop_pct": "",
                "include_in_run_all": "",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(save_configs_mock.called)
        saved_payload = save_configs_mock.call_args[0][1]
        self.assertEqual(saved_payload["model_a"]["ticker"], "AAPL")
        self.assertEqual(saved_payload["model_b"]["ticker"], "MSFT")
        self.assertEqual(saved_payload["model_a"]["rows"], 250)
        self.assertEqual(saved_payload["model_b"]["rows"], 300)
        self.assertEqual(saved_payload["model_a"]["stop_loss_strategy"], "atr")
        self.assertEqual(saved_payload["model_b"]["stop_loss_strategy"], "atr")

    @patch("main.fetch_market_rows")
    @patch("main.evaluate_bundle")
    @patch("main.train_strategy_models")
    def test_evaluate_real_history_forces_chronological_split(self, train_mock, eval_mock, fetch_market_rows_mock):
        fetch_market_rows_mock.return_value = (
            [
                {"return_next": 0.01},
                {"return_next": -0.01},
                {"return_next": 0.02},
                {"return_next": 0.01},
            ],
            None,
        )
        train_mock.return_value = {
            "x_test_raw": [[0.1], [0.2]],
            "y_test_ret": [0.01, -0.01],
            "y_test_dir": [1, 0],
            "train_size": 2,
            "test_size": 2,
            "split_style": "chronological",
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
                "train_action": "evaluate_historical",
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
        self.assertEqual(train_mock.call_args.kwargs["split_style"], "chronological")
        self.assertIn("Historical evaluation mode", response.get_data(as_text=True))

    @patch("main.list_saved_models")
    @patch("main.save_model_configs")
    @patch("main.load_model_configs")
    @patch("main.save_model_bundle")
    @patch("main.fetch_market_rows")
    @patch("main.evaluate_bundle")
    @patch("main.train_strategy_models")
    def test_multi_ticker_training_saves_model_configs_without_explicit_model_names(
        self,
        train_mock,
        eval_mock,
        fetch_market_rows_mock,
        save_bundle_mock,
        load_configs_mock,
        save_configs_mock,
        list_models_mock,
    ):
        fetch_market_rows_mock.return_value = (
            [
                {"return_next": 0.01},
                {"return_next": -0.01},
                {"return_next": 0.02},
                {"return_next": 0.01},
            ],
            None,
        )
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
        load_configs_mock.return_value = {}
        list_models_mock.return_value = []

        response = self.app.post(
            "/",
            data={
                "mode": "train",
                "train_action": "train",
                "ticker": "NVDA,AMD",
                "interval": "1d",
                "rows": "4000",
                "split_style": "shuffled",
                "feature_set": "hybrid_sharpe_core_no_stack",
                "buy_threshold": "0.6",
                "sell_threshold": "0.4",
                "selected_model": "__new__",
                "model_name": "",
                "stop_loss_strategy": "none",
                "fixed_stop_pct": "",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(save_bundle_mock.call_count, 2)
        self.assertTrue(save_configs_mock.called)
        saved_payload = save_configs_mock.call_args[0][1]
        self.assertEqual(saved_payload["nvda_1d_4000_5"]["ticker"], "NVDA")
        self.assertEqual(saved_payload["amd_1d_4000_5"]["ticker"], "AMD")


if __name__ == "__main__":
    unittest.main()
