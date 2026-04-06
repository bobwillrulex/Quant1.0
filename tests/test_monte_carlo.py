import math
import unittest
from unittest.mock import patch

from quant.ml import evaluate_bundle
from quant.ml_monte_carlo import cvar, run_monte_carlo_backtest, summarize_distribution


class MonteCarloBacktestTests(unittest.TestCase):
    def test_summarize_distribution_basic(self):
        summary = summarize_distribution([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(summary["mean"], 3.0)
        self.assertAlmostEqual(summary["median"], 3.0)
        self.assertAlmostEqual(summary["min"], 1.0)
        self.assertAlmostEqual(summary["max"], 5.0)
        self.assertAlmostEqual(summary["p5"], 1.2)
        self.assertAlmostEqual(summary["p95"], 4.8)

    def test_cvar_tail_average(self):
        values = [-0.8, -0.4, -0.1, 0.05, 0.1, 0.2]
        self.assertAlmostEqual(cvar(values, 0.5), (-0.8 - 0.4 - 0.1) / 3.0)
        self.assertAlmostEqual(cvar([], 0.05), 0.0)

    def test_run_monte_carlo_backtest_methods(self):
        returns = [0.01, -0.02, 0.015, -0.005, 0.012, 0.0, -0.01, 0.004]
        probs = [0.65, 0.4, 0.7, 0.45, 0.66, 0.52, 0.35, 0.6]
        strategy_fn = lambda mc_returns, _mc_probs, **_kwargs: {"bar_returns": list(mc_returns), "trade_returns": []}
        for method in ("bootstrap", "shuffle", "block"):
            result = run_monte_carlo_backtest(
                returns=returns,
                probs=probs,
                strategy_fn=strategy_fn,
                n_sim=20,
                method=method,
                block_size=3,
                seed=7,
            )
            self.assertIn("raw_results", result)
            self.assertIn("summary", result)
            self.assertEqual(len(result["raw_results"]), 20)
            self.assertIn("mean_return", result["summary"])
            self.assertIn("median_return", result["summary"])
            self.assertIn("cvar_5_return", result["summary"])
            self.assertIn("log_mean_return", result["summary"])
            self.assertIn("log_median_return", result["summary"])
            self.assertIn("skewness", result["summary"])
            self.assertIn("kurtosis", result["summary"])
            self.assertIn("probability_of_loss", result["summary"])
            self.assertIn("probability_of_large_loss", result["summary"])
            self.assertIn("probability_of_ruin", result["summary"])
            self.assertTrue(math.isfinite(float(result["summary"]["log_mean_return"])))

    def test_evaluate_bundle_monte_carlo_uses_strategy_step_returns(self):
        bundle = {
            "feature_names": ["f1"],
            "means": [0.0],
            "stds": [1.0],
            "lin_weights": [0.0],
            "lin_bias": 0.0,
            "logit_weights": [0.0],
            "logit_bias": 0.0,
        }
        y_test_ret = [0.10, -0.05, 0.08]
        eval_rows = [
            {"close": 100.0, "timestamp": "2026-01-01T09:30:00"},
            {"close": 100.1, "timestamp": "2026-01-01T09:35:00"},
            {"close": 100.05, "timestamp": "2026-01-01T09:40:00"},
        ]
        captured: dict[str, list[float]] = {}

        def fake_mc(*args, **kwargs):
            captured["returns"] = list(kwargs.get("returns", []))
            return {"raw_results": [], "summary": {"mean_return": 0.0}}

        with patch("quant.ml.run_monte_carlo_backtest", side_effect=fake_mc):
            evaluate_bundle(
                bundle=bundle,
                x_test_raw=[[1.0], [2.0], [3.0]],
                y_test_ret=y_test_ret,
                y_test_dir=[1, 0, 1],
                eval_rows=eval_rows,
                split_style="chronological",
                monte_carlo_method="bootstrap",
                monte_carlo_n_sim=5,
            )

        self.assertIn("returns", captured)
        expected_step_returns = [0.0, (100.1 / 100.0) - 1.0, (100.05 / 100.1) - 1.0]
        for actual, expected in zip(captured["returns"], expected_step_returns):
            self.assertAlmostEqual(actual, expected)



if __name__ == "__main__":
    unittest.main()
