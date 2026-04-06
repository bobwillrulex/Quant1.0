import unittest

from quant.ml import strategy_metrics
from quant.stop_loss import StopLossConfig, StopLossStrategy, stop_loss_price


class StopLossStrategyTests(unittest.TestCase):
    def test_stop_loss_price_fixed_long(self):
        price = stop_loss_price(
            strategy=StopLossStrategy.FIXED_PERCENTAGE,
            action="BUY",
            reference_price=100.0,
            fixed_pct=2.0,
        )
        self.assertAlmostEqual(price or 0.0, 98.0, places=6)

    def test_stop_loss_price_model_invalidation_short(self):
        price = stop_loss_price(
            strategy=StopLossStrategy.MODEL_INVALIDATION,
            action="SELL",
            reference_price=100.0,
            expected_return=0.01,
            model_mae=0.02,
        )
        self.assertAlmostEqual(price or 0.0, 103.0, places=6)

    def test_stop_loss_price_trailing_stop_matches_fixed_distance(self):
        price = stop_loss_price(
            strategy=StopLossStrategy.TRAILING_STOP,
            action="BUY",
            reference_price=100.0,
            fixed_pct=2.0,
        )
        self.assertAlmostEqual(price or 0.0, 98.0, places=6)

    def test_fixed_percentage_generates_stop_exits(self):
        returns = [0.0, 0.0, -0.03, 0.0, 0.0]
        probs = [0.8] * len(returns)
        expected = [0.04] * len(returns)
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.FIXED_PERCENTAGE, fixed_pct=2.0),
        )
        self.assertGreaterEqual(metrics["stop_loss_exits"], 1.0)
        self.assertEqual(metrics["trade_count"], 1.0)

    def test_trailing_stop_generates_stop_exits(self):
        returns = [0.01, 0.01, -0.04, 0.002]
        probs = [0.8] * len(returns)
        expected = [0.04] * len(returns)
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.TRAILING_STOP, fixed_pct=2.0),
        )
        self.assertGreaterEqual(metrics["stop_loss_exits"], 1.0)

    def test_time_decay_exits_on_bar_limit(self):
        returns = [0.001] * 20
        probs = [0.9] * 20
        expected = [0.01] * 20
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.TIME_DECAY, time_decay_bars=3),
        )
        self.assertGreaterEqual(metrics["time_decay_exits"], 1.0)

    def test_none_strategy_applies_no_stop_logic(self):
        returns = [0.001] * 10
        probs = [0.9] * 10
        expected = [0.01] * 10
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE),
        )
        self.assertEqual(metrics["stop_loss_exits"], 0.0)
        self.assertEqual(metrics["time_decay_exits"], 0.0)

    def test_model_invalidation_triggers_exit(self):
        returns = [-0.03, -0.03, -0.02, 0.0]
        probs = [0.9, 0.9, 0.9, 0.9]
        expected = [0.01, 0.01, 0.01, 0.01]
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.MODEL_INVALIDATION, model_mae=0.01),
        )
        self.assertGreaterEqual(metrics["stop_loss_exits"], 1.0)

    def test_fixed_stop_exit_uses_stop_price_return(self):
        returns = [0.0, 0.0, -0.05, 0.0]
        probs = [0.9, 0.9, 0.9, 0.2]
        expected = [0.02, 0.02, 0.02, 0.02]
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.FIXED_PERCENTAGE, fixed_pct=2.0),
        )
        self.assertAlmostEqual(float(metrics["total_return"]), -0.0204, places=6)

    def test_model_invalidation_exit_uses_stop_price_return(self):
        returns = [0.0, 0.0, -0.05, 0.0]
        probs = [0.9, 0.9, 0.9, 0.2]
        expected = [0.01, 0.01, 0.01, 0.01]
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.MODEL_INVALIDATION, model_mae=0.01),
        )
        self.assertAlmostEqual(float(metrics["total_return"]), -0.0504, places=6)

    def test_total_return_uses_equity_curve(self):
        returns = [0.0, 0.0, 0.10, 0.0]
        probs = [0.9, 0.9, 0.9, 0.2]
        expected = [0.01, 0.01, 0.01, 0.01]
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE),
        )
        self.assertAlmostEqual(float(metrics["total_return"]), 0.0996, places=6)

    def test_trade_log_includes_normalized_and_raw_prices(self):
        returns = [0.0, 0.01, -0.02, 0.0]
        probs = [0.9, 0.9, 0.9, 0.2]
        expected = [0.01, 0.01, 0.01, 0.01]
        raw_prices = [100.0, 101.0, 99.0, 98.0]
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE),
            raw_prices=raw_prices,
        )
        self.assertEqual(len(metrics["trade_log"]), 1)
        trade = metrics["trade_log"][0]
        self.assertAlmostEqual(float(trade["entry_price"]), 1.01, places=6)
        self.assertAlmostEqual(float(trade["exit_price"]), 0.9898, places=6)
        self.assertAlmostEqual(float(trade["entry_raw_price"]), 101.0, places=6)
        self.assertAlmostEqual(float(trade["exit_raw_price"]), 98.0, places=6)
        self.assertAlmostEqual(float(trade["max_drawdown_during_trade"]), -0.02970297029702973, places=6)
        self.assertAlmostEqual(float(trade["max_upside_during_trade"]), 0.0, places=6)

    def test_no_signals_keeps_configured_thresholds_and_zero_trades(self):
        returns = [0.01, -0.005, 0.004, -0.002]
        probs = [0.525] * len(returns)
        expected = [0.002] * len(returns)
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.4,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE),
        )
        self.assertEqual(metrics["threshold_mode"], "configured")
        self.assertEqual(metrics["trade_count"], 0.0)

    def test_fixed_stop_caps_max_trade_loss_near_stop_plus_costs(self):
        returns = [0.0, 0.0, -0.08, 0.0]
        probs = [0.9, 0.9, 0.9, 0.2]
        expected = [0.02, 0.02, 0.02, 0.02]
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.FIXED_PERCENTAGE, fixed_pct=2.0),
        )
        self.assertAlmostEqual(float(metrics["max_loss_per_trade"]), -0.0204, places=6)

    def test_sharpe_annualizes_from_timestamped_rows(self):
        returns = [0.0, 0.002, -0.001, 0.003, -0.001]
        probs = [0.9, 0.9, 0.9, 0.9, 0.9]
        expected = [0.01] * len(returns)
        row_labels = [
            "2026-03-06T14:40:00+00:00",
            "2026-03-06T14:45:00+00:00",
            "2026-03-06T14:50:00+00:00",
            "2026-03-06T14:55:00+00:00",
            "2026-03-06T15:00:00+00:00",
        ]
        metrics_with_labels = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE),
            row_labels=row_labels,
        )
        metrics_without_labels = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE),
        )
        self.assertGreater(float(metrics_with_labels["sharpe"]), float(metrics_without_labels["sharpe"]))

    def test_fixed_stop_caps_reported_max_drawdown_to_trade_stop_bound(self):
        returns = [0.10, 0.10, -0.06, 0.0]
        probs = [0.9] * len(returns)
        expected = [0.02] * len(returns)
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.FIXED_PERCENTAGE, fixed_pct=2.0),
        )
        self.assertGreater(float(metrics["max_drawdown"]), 0.0203)

    def test_max_drawdown_is_bounded_to_100_percent_without_stop(self):
        returns = [-0.70, -0.70, 0.0]
        probs = [0.9, 0.9, 0.2]
        expected = [0.0, 0.0, 0.0]
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE),
        )
        self.assertLessEqual(float(metrics["max_drawdown"]), 1.0)

    def test_stop_loss_cooldown_blocks_immediate_reentry(self):
        returns = [0.0, 0.0, -0.03, 0.0, 0.0, 0.0]
        probs = [0.9] * len(returns)
        expected = [0.02] * len(returns)
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.FIXED_PERCENTAGE, fixed_pct=2.0),
        )
        self.assertEqual(metrics["stop_loss_exits"], 1.0)
        self.assertEqual(metrics["time_decay_exits"], 0.0)

    def test_signal_execution_is_shifted_by_one_bar(self):
        returns = [-0.20, 0.0, 0.0]
        probs = [0.9, 0.2, 0.2]
        expected = [0.0, 0.0, 0.0]
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE),
        )
        self.assertAlmostEqual(float(metrics["total_return"]), -0.0004, places=6)

    def test_take_profit_exit_triggers_when_target_is_hit(self):
        returns = [0.0, 0.0, 0.03, 0.0]
        probs = [0.9, 0.9, 0.9, 0.9]
        expected = [0.01] * len(returns)
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE, take_profit_pct=2.0),
        )
        self.assertGreaterEqual(metrics["take_profit_exits"], 1.0)

    def test_max_hold_exit_triggers_when_bar_limit_reached(self):
        returns = [0.0, 0.001, 0.001, 0.001, 0.001]
        probs = [0.9] * len(returns)
        expected = [0.01] * len(returns)
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE, max_hold_bars=2),
        )
        self.assertGreaterEqual(metrics["max_hold_exits"], 1.0)

    def test_hold_time_stats_use_closed_trades_not_position_streaks(self):
        returns = [0.0] + [0.001] * 12
        probs = [0.9] * len(returns)
        expected = [0.01] * len(returns)
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            prob_smoothing_window=1,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.NONE, max_hold_bars=2),
        )
        hold_stats = metrics["hold_time_stats"]
        self.assertGreaterEqual(float(hold_stats["count"]), 2.0)
        self.assertLessEqual(float(hold_stats["max"]), 3.0)


if __name__ == "__main__":
    unittest.main()
