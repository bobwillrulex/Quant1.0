# ML Features and Outputs Reference

This project’s ML pipeline is centered around:

- **Feature-set builders** in `quant/strategy_features.py`.
- **Model training/evaluation** in `quant/ml.py`.
- **Monte Carlo simulation utilities** in `quant/ml_monte_carlo.py`.

## 1) Feature Sets (what each model input uses)

A feature set is selected by `feature_set` and normalized via `normalize_feature_set(...)`.

Supported canonical sets:

- `feature2` (default)
- `new`
- `legacy`
- `fvg2`
- `fvg3`
- `derivative`
- `derivative2`
- `dqn`
- `ema`
- `bollinger_bands`
- `vwap_anchor`
- `vwap_intraday_reversion`
- `vwap_intraday_momentum`
- `vwap_intraday_5m_session`
- `vwap_breakout_reversion_regime`
- `hybrid_sharpe_core`
- `hybrid_sharpe_core_no_stack`
- `hybrid_sharpe_momentum`
- `hybrid_sharpe_selective`
- `hybrid_sharpe_regime`
- `hybrid_sharpe_volume_flow`
- `hybrid_sharpe_volume_regime`
- `rsi_thresholds`
- `stoch_rsi_thresholds`

### Typical feature categories used across sets

Depending on the selected set, transformed inputs include combinations of:

- **Oscillator/state features** (e.g., `stoch_rsi`, threshold flags, velocity).
- **Momentum/returns** (e.g., `ret_1`, `ret_3`, `ret_5`).
- **Trend/volatility features** (e.g., `trend_20`, `vol_20`).
- **MACD-derived features** (histogram level, delta, directional states).
- **FVG structure/context** (distance, in-zone flags, confluence signals).
- **EMA/VWAP/Bollinger/regime features** for richer multi-regime sets.

For the exact, ordered feature names used in a run, use:

- `bundle["feature_names"]` after training.
- `metrics["features"]` from `evaluate_bundle(...)`.

## 2) Training outputs (what `train_strategy_models(...)` returns)

### Common output fields

- `feature_names`, `feature_set`
- `means`, `stds` (standardization parameters)
- `train_size`, `test_size`, `split_style`
- `x_test_raw`, `y_test_ret`, `y_test_dir`

### Linear/Logistic mode outputs

- `lin_weights`, `lin_bias` (return regression)
- `logit_weights`, `logit_bias` (direction classification)

### DQN mode outputs (`feature_set == "dqn"`)

- `model_type = "dqn"`
- `dqn_state_dict`, `dqn_state_size`, `dqn_action_size`
- `dqn_action_returns`, `dqn_last_epsilon`, `dqn_episode_rewards`
- linear/logistic weights are present as zero placeholders for compatibility.

## 3) Evaluation outputs (what `evaluate_bundle(...)` returns)

### Predictive metrics

- Regression: `mse`, `mae`
- Classification: `accuracy`, `precision`, `recall`, `f1`, `tp`, `tn`, `fp`, `fn`
- Baselines and deltas:
  - `baseline_always_up_accuracy`, `accuracy_vs_baseline`
  - `baseline_zero_mse`, `baseline_zero_mae`
  - `mse_vs_zero_baseline`, `mae_vs_zero_baseline`

### Prediction artifacts

- `preview`: sample rows of `{expected_return, p_up, actual_return}`
- `lin_weights` / `logit_weights` zipped with feature names
- `calibration`, `confidence_edge`, `error_analysis`

### Strategy/backtest outputs (`strategy`)

`strategy_metrics(...)` produces a detailed dict including:

- Signal/threshold config used
- Stop-loss behavior and exit counts
- Returns and attribution (`total_return`, `buy_hold_total_return`, `alpha`)
- Risk metrics (`sharpe`, `max_drawdown`, `avg_drawdown`, `probability_of_loss`)
- Trade diagnostics (`trade_count`, `win_rate`, `avg_gain_per_trade`, `max_loss_per_trade`)
- `trade_returns` list (per-trade return series)
- `hold_time_stats`

### Monte Carlo outputs (`monte_carlo`, optional)

Enabled when `monte_carlo_method` is one of `bootstrap`, `shuffle`, or `block`.

`run_monte_carlo_backtest(...)` returns:

- `raw_results`: simulation-level dicts:
  - `total_return`, `sharpe`, `max_drawdown`, `win_rate`, `log_total_return`
- `summary`: aggregate distribution metrics:
  - central tendency/spread (`mean_return`, `median_return`, `std_return`)
  - tails (`p5_return`, `p95_return`, `cvar_5_return`)
  - shape (`skewness`, `kurtosis`)
  - risk probabilities (`probability_of_loss`, `probability_of_large_loss`, `probability_of_ruin`)
  - drawdown and Sharpe summaries (`mean_sharpe`, `mean_drawdown`, `worst_drawdown`)
- optional `equity_curves` sample paths.

### Additional diagnostics

- `pnl_by_signal_strength`
- `pnl_by_regime`
- `walk_forward`
- `feature_ablation`
- DQN diagnostics (`dqn_policy`, `dqn_episode_rewards`, `dqn_last_epsilon`, `dqn_action_returns`)

## 4) Monte Carlo refactor notes

Monte Carlo/statistical helpers were separated into `quant/ml_monte_carlo.py` so the core model lifecycle in `quant/ml.py` stays focused on:

- data split/scaling,
- model training,
- strategy simulation,
- bundle evaluation orchestration.

This keeps simulation logic reusable and easier to test independently.
