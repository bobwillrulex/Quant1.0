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

    def test_fixed_percentage_generates_stop_exits(self):
        returns = [-0.01, -0.015, -0.01, 0.005, 0.004]
        probs = [0.8] * len(returns)
        expected = [0.04] * len(returns)
        metrics = strategy_metrics(
            returns,
            probs,
            expected_returns=expected,
            long_threshold=0.6,
            short_threshold=0.2,
            allow_short=False,
            stop_loss=StopLossConfig(strategy=StopLossStrategy.FIXED_PERCENTAGE, fixed_pct=2.0),
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
            stop_loss=StopLossConfig(strategy=StopLossStrategy.MODEL_INVALIDATION, model_mae=0.01),
        )
        self.assertGreaterEqual(metrics["stop_loss_exits"], 1.0)


if __name__ == "__main__":
    unittest.main()
