import math
import unittest

from quant.ml import cvar, run_monte_carlo_backtest, summarize_distribution


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
        for method in ("bootstrap", "shuffle", "block"):
            result = run_monte_carlo_backtest(
                returns=returns,
                probs=probs,
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


if __name__ == "__main__":
    unittest.main()
