from quant.ml import get_strategy_feature_builder, normalize_feature_set


def test_feature2_is_supported_and_default_builder_can_be_selected() -> None:
    assert normalize_feature_set("feature2") == "feature2"
    builder = get_strategy_feature_builder("feature2")
    names = builder.names()
    assert "macd_delta" in names
    assert "inside_bull_fvg" in names
    assert "oversold_reversal" not in names


def test_derivative_feature_set_is_supported() -> None:
    assert normalize_feature_set("derivative") == "derivative"
    builder = get_strategy_feature_builder("derivative")
    names = builder.names()
    expected = {
        "macd_hist",
        "macd_hist_delta",
        "macd_green_increasing",
        "macd_red_recovering",
        "macd_green_fading",
        "macd_red_deepening",
        "ema9",
        "ema26",
        "ema9_derivative_1",
        "ema9_derivative_2",
        "ema9_derivative_3",
        "ema26_derivative_1",
        "ema26_derivative_2",
        "ema26_derivative_3",
    }
    assert expected.issubset(set(names))


def test_fvg2_feature_set_splits_stoch_extremes() -> None:
    assert normalize_feature_set("fvg_2") == "fvg2"
    builder = get_strategy_feature_builder("fvg2")
    names = builder.names()
    assert "stoch_extreme_80" in names
    assert "stoch_extreme_20" in names
    assert "stoch_extreme" not in names
    assert "fvg_conflict_penalty" in names


def test_fvg3_feature_set_is_supported() -> None:
    assert normalize_feature_set("fvg_3") == "fvg3"
    builder = get_strategy_feature_builder("fvg3")
    names = builder.names()
    assert "stoch_extreme" in names
    assert "stoch_extreme_neg" in names
    assert "macd_hist_delta_absolute" in names
    assert "fvg_conflict_penalty" in names


def test_derivative2_feature_set_is_supported() -> None:
    assert normalize_feature_set("derivate2") == "derivative2"
    builder = get_strategy_feature_builder("derivative2")
    names = builder.names()
    expected = {
        "macd_hist",
        "macd_hist_delta",
        "macd_green_increasing",
        "macd_red_recovering",
        "macd_green_fading",
        "macd_red_deepening",
        "ema9",
        "ema26",
        "ema9_derivative_1",
        "ema9_derivative_2",
        "ema9_derivative_3",
        "ema26_derivative_1",
        "ema26_derivative_2",
        "ema26_derivative_3",
        "ema_derivative_1_diff",
        "ema_derivative_2_diff",
        "ema_derivative_3_diff",
        "ema_derivative_1_cross",
        "ema_derivative_1_cross_positive",
        "ema_derivative_1_cross_negative",
        "ema_derivative_2_cross",
        "ema_derivative_2_cross_positive",
        "ema_derivative_2_cross_negative",
        "ema_derivative_3_cross",
        "ema_derivative_3_cross_positive",
        "ema_derivative_3_cross_negative",
    }
    assert expected.issubset(set(names))


def test_dqn_feature_set_is_supported() -> None:
    assert normalize_feature_set("dqn") == "dqn"
    dqn_builder = get_strategy_feature_builder("dqn")
    base_builder = get_strategy_feature_builder("feature2")
    assert dqn_builder.names() == base_builder.names()


def test_ema_feature_set_is_supported() -> None:
    assert normalize_feature_set("many-ema") == "ema"
    builder = get_strategy_feature_builder("ema")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema_stack_bullish",
        "ema_stack_bearish",
        "ema3_slope",
        "ema9_slope",
        "ema21_slope",
    }
    assert expected.issubset(set(names))


def test_bollinger_bands_feature_set_is_supported() -> None:
    assert normalize_feature_set("bollinger") == "bollinger_bands"
    builder = get_strategy_feature_builder("bollinger_bands")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_percent_b",
        "price_to_bb_mid",
    }
    assert expected.issubset(set(names))


def test_vwap_anchor_feature_set_is_supported() -> None:
    assert normalize_feature_set("anchored-vwap") == "vwap_anchor"
    builder = get_strategy_feature_builder("vwap_anchor")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "vwap_anchor_high",
        "vwap_anchor_low",
        "vwap_anchor_spread",
        "price_vs_vwap_high",
        "price_vs_vwap_low",
        "vwap_anchor_mid_bias",
    }
    assert expected.issubset(set(names))


def test_vwap_intraday_reversion_feature_set_is_supported() -> None:
    assert normalize_feature_set("intraday-vwap-reversion") == "vwap_intraday_reversion"
    builder = get_strategy_feature_builder("vwap_intraday_reversion")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "vwap_anchor_high",
        "vwap_anchor_low",
        "vwap_anchor_spread",
        "price_vs_vwap_mid",
        "distance_to_vwap_low",
        "distance_to_vwap_high",
        "zscore_vwap_mid",
        "mean_revert_long_bias",
        "mean_revert_short_bias",
    }
    assert expected.issubset(set(names))


def test_vwap_intraday_momentum_feature_set_is_supported() -> None:
    assert normalize_feature_set("intraday-vwap-momentum") == "vwap_intraday_momentum"
    builder = get_strategy_feature_builder("vwap_intraday_momentum")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema3_slope",
        "ema9_slope",
        "macd_hist",
        "macd_hist_delta",
        "vwap_anchor_high",
        "vwap_anchor_low",
        "price_vs_vwap_mid",
        "vwap_breakout_strength",
        "vwap_breakdown_strength",
        "session_trend_pressure",
    }
    assert expected.issubset(set(names))


def test_vwap_intraday_5m_session_feature_set_is_supported() -> None:
    assert normalize_feature_set("session_vwap_5m") == "vwap_intraday_5m_session"
    builder = get_strategy_feature_builder("vwap_intraday_5m_session")
    names = builder.names()
    expected = {
        "session_vwap_5m",
        "session_vwap_delta_5m",
        "price_vs_session_vwap_5m",
        "price_vs_session_vwap_pct_5m",
        "abs_price_vs_session_vwap_5m",
        "session_vwap_reversion_signal_5m",
    }
    assert expected.issubset(set(names))


def test_vwap_breakout_reversion_regime_feature_set_is_supported() -> None:
    assert normalize_feature_set("vwap-breakout-vs-reversion") == "vwap_breakout_reversion_regime"
    builder = get_strategy_feature_builder("vwap_breakout_reversion_regime")
    names = builder.names()
    expected = {
        "vwap_anchor_high",
        "vwap_anchor_low",
        "price_vs_vwap_mid",
        "mean_revert_long_bias",
        "mean_revert_short_bias",
        "vwap_breakout_strength",
        "vwap_breakdown_strength",
        "breakout_pressure",
        "mean_reversion_pressure",
        "vwap_regime_signal",
        "is_breakout_regime",
        "is_reversion_regime",
    }
    assert expected.issubset(set(names))


def test_open15_orb_intraday_feature_set_is_supported() -> None:
    assert normalize_feature_set("opening_range_breakout") == "open15_orb_intraday"
    builder = get_strategy_feature_builder("open15_orb_intraday")
    names = builder.names()
    expected = {
        "post_opening_range_window_15m",
        "opening_range_breakout_up_15m",
        "opening_range_breakdown_15m",
        "price_vs_opening_range_high_15m",
        "price_vs_opening_range_low_15m",
        "opening_range_width_pct_15m",
        "price_vs_session_vwap_5m",
        "intraday_trade_window_open",
        "near_session_close_5m",
        "bars_remaining_in_session_5m",
    }
    assert expected.issubset(set(names))


def test_open15_vwap_reclaim_intraday_feature_set_is_supported() -> None:
    assert normalize_feature_set("vwap_reclaim_open15") == "open15_vwap_reclaim_intraday"
    builder = get_strategy_feature_builder("open15_vwap_reclaim_intraday")
    names = builder.names()
    expected = {
        "post_opening_range_window_15m",
        "opening_range_position_pct_15m",
        "price_vs_session_vwap_5m",
        "session_vwap_reversion_signal_5m",
        "vwap_reclaim_long_signal_5m",
        "vwap_reclaim_short_signal_5m",
        "stoch_rsi_norm",
        "intraday_trade_window_open",
        "near_session_close_5m",
    }
    assert expected.issubset(set(names))


def test_open15_trend_momentum_daytrade_feature_set_is_supported() -> None:
    assert normalize_feature_set("open15_momentum_daytrade") == "open15_trend_momentum_daytrade"
    builder = get_strategy_feature_builder("open15_trend_momentum_daytrade")
    names = builder.names()
    expected = {
        "post_opening_range_window_15m",
        "opening_range_breakout_up_15m",
        "opening_range_breakdown_15m",
        "ema3_9_spread",
        "ema9_21_spread",
        "macd_hist",
        "macd_hist_delta",
        "session_trend_pressure",
        "intraday_trade_window_open",
        "near_session_close_5m",
        "bars_remaining_in_session_5m",
        "trade_count_today",
        "trades_remaining_cap_2",
        "avoid_overnight_bias",
        "open15_breakout_strength_atr",
        "open15_breakdown_strength_atr",
    }
    assert expected.issubset(set(names))


def test_open15_trend_momentum_daytrade_active_feature_set_is_supported() -> None:
    assert normalize_feature_set("open15_active") == "open15_trend_momentum_daytrade_active"
    builder = get_strategy_feature_builder("open15_trend_momentum_daytrade_active")
    names = builder.names()
    expected = {
        "post_opening_range_window_15m",
        "opening_range_breakout_up_15m",
        "opening_range_breakdown_15m",
        "price_vs_session_vwap_5m",
        "session_vwap_delta_5m",
        "ema_slope_alignment",
        "trade_count_today",
        "trades_remaining_cap_4",
        "third_plus_trade_requires_trend_health",
        "avoid_overnight_bias",
        "open15_breakout_strength_atr",
    }
    assert expected.issubset(set(names))


def test_open15_dual_breakout_daytrade_feature_set_is_supported() -> None:
    assert normalize_feature_set("open15_breakout_followthrough") == "open15_dual_breakout_daytrade"
    builder = get_strategy_feature_builder("open15_dual_breakout_daytrade")
    names = builder.names()
    expected = {
        "post_opening_range_window_15m",
        "opening_range_breakout_up_15m",
        "opening_range_breakdown_15m",
        "opening_range_position_pct_15m",
        "price_vs_session_vwap_5m",
        "vwap_breakout_strength",
        "vwap_breakdown_strength",
        "ema3_9_spread",
        "ema9_21_spread",
        "macd_hist",
        "macd_hist_delta",
        "momentum_alignment",
        "intraday_trade_window_open",
        "near_session_close_5m",
        "bars_remaining_in_session_5m",
        "trade_count_today",
        "second_trade_only_if_trend_intact",
        "avoid_overnight_bias",
    }
    assert expected.issubset(set(names))


def test_adaptive_opening_range_momentum_daytrade_feature_set_is_supported() -> None:
    assert normalize_feature_set("open5_15_20_adaptive") == "adaptive_opening_range_momentum_daytrade"
    builder = get_strategy_feature_builder("adaptive_opening_range_momentum_daytrade")
    names = builder.names()
    expected = {
        "session_bar_index_5m",
        "open5_ready_flag",
        "open15_ready_flag",
        "open20_ready_flag",
        "opening_range_breakout_up_15m",
        "opening_range_breakdown_15m",
        "session_vwap_reversion_signal_5m",
        "vwap_reclaim_long_signal_5m",
        "trades_remaining_cap_3",
        "avoid_overnight_bias",
        "breakout_strength_atr",
    }
    assert expected.issubset(set(names))


def test_open15_dual_breakout_daytrade_plus_feature_set_is_supported() -> None:
    assert normalize_feature_set("open15_breakout_followthrough_plus") == "open15_dual_breakout_daytrade_plus"
    builder = get_strategy_feature_builder("open15_dual_breakout_daytrade_plus")
    names = builder.names()
    expected = {
        "post_opening_range_window_15m",
        "opening_range_breakout_up_15m",
        "opening_range_breakdown_15m",
        "opening_range_position_pct_15m",
        "price_vs_session_vwap_5m",
        "session_vwap_delta_to_mean_5m",
        "vwap_reclaim_long_signal_5m",
        "vwap_breakout_strength_atr",
        "ema3_9_spread",
        "macd_hist",
        "trend_pullback_quality",
        "intraday_trade_window_open",
        "near_session_close_5m",
        "trade_count_today",
        "trades_remaining_cap_3",
        "third_trade_only_if_reclaim_valid",
        "avoid_overnight_bias",
    }
    assert expected.issubset(set(names))


def test_open15_dual_breakout_daytrade_scalp_feature_set_is_supported() -> None:
    assert normalize_feature_set("open15_breakout_followthrough_scalp") == "open15_dual_breakout_daytrade_scalp"
    builder = get_strategy_feature_builder("open15_dual_breakout_daytrade_scalp")
    names = builder.names()
    expected = {
        "post_opening_range_window_15m",
        "opening_range_breakout_up_15m",
        "opening_range_breakdown_15m",
        "price_vs_session_vwap_5m",
        "session_vwap_reversion_signal_5m",
        "vwap_reclaim_long_signal_5m",
        "vwap_breakout_strength_atr",
        "ema3_9_spread",
        "ema3_slope",
        "macd_hist",
        "ret_1",
        "ret_3",
        "intraday_trade_window_open",
        "trade_count_today",
        "trades_remaining_cap_5",
        "fourth_plus_trade_requires_momentum",
        "avoid_overnight_bias",
    }
    assert expected.issubset(set(names))


def test_war_shock_reversion_feature_set_is_supported() -> None:
    assert normalize_feature_set("war-shock-reversion") == "war_shock_reversion"
    builder = get_strategy_feature_builder("war_shock_reversion")
    names = builder.names()
    expected = {
        "shock_intensity",
        "panic_down_move",
        "oversold_snapback_bias",
        "distance_to_vwap_low_atr",
        "bb_reversion_distance",
        "mean_revert_long_bias",
        "session_vwap_reversion_signal_5m",
    }
    assert expected.issubset(set(names))


def test_vwap_volume_first5_trend_momentum_5m_feature_set_is_supported() -> None:
    assert normalize_feature_set("first5_vwap_volume_momentum") == "vwap_volume_first5_trend_momentum_5m"
    builder = get_strategy_feature_builder("vwap_volume_first5_trend_momentum_5m")
    names = builder.names()
    expected = {
        "session_bar_index_5m",
        "first5_bar_window_ready",
        "opening_range_breakout_up_15m",
        "opening_range_breakdown_15m",
        "price_vs_session_vwap_5m",
        "vwap_breakout_strength_atr",
        "vwap_breakdown_strength_atr",
        "volume_spike_ratio",
        "trade_count_today",
        "trades_remaining_cap_2",
        "avoid_overnight_bias",
    }
    assert expected.issubset(set(names))


def test_vwap_volume_profile_first5_trend_momentum_5m_feature_set_is_supported() -> None:
    assert normalize_feature_set("first5_vwap_volume_profile_momentum") == "vwap_volume_profile_first5_trend_momentum_5m"
    builder = get_strategy_feature_builder("vwap_volume_profile_first5_trend_momentum_5m")
    names = builder.names()
    expected = {
        "first5_bar_window_ready",
        "trades_remaining_cap_2",
        "session_vwap_std_1_5m",
        "session_vwap_std_2_5m",
        "session_vwap_std_1_range_5m",
        "session_vwap_std_2_range_5m",
        "volume_profile_acceptance_1sigma",
        "volume_profile_expansion_2sigma",
        "volume_profile_momentum_bias",
    }
    assert expected.issubset(set(names))


def test_war_shock_momentum_feature_set_is_supported() -> None:
    assert normalize_feature_set("war-shock-momentum") == "war_shock_momentum"
    builder = get_strategy_feature_builder("war_shock_momentum")
    names = builder.names()
    expected = {
        "shock_intensity",
        "ema3_9_spread",
        "ema9_21_spread",
        "macd_hist_delta",
        "breakout_strength_atr",
        "breakdown_strength_atr",
        "trend_follow_thrust",
    }
    assert expected.issubset(set(names))


def test_hybrid_sharpe_core_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-core") == "hybrid_sharpe_core"
    builder = get_strategy_feature_builder("hybrid_sharpe_core")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema_stack_bullish",
        "ema_stack_bearish",
        "ema3_slope",
        "ema9_slope",
        "ema21_slope",
        "macd_hist",
        "macd_hist_delta",
        "macd_green_increasing",
        "macd_red_recovering",
        "macd_green_fading",
        "macd_red_deepening",
        "ema_derivative_1_diff",
        "ema_derivative_1_cross_positive",
        "ema_derivative_1_cross_negative",
    }
    assert expected.issubset(set(names))


def test_hybrid_sharpe_core_no_stack_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-core-no-stack") == "hybrid_sharpe_core_no_stack"
    builder = get_strategy_feature_builder("hybrid_sharpe_core_no_stack")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema3_slope",
        "ema9_slope",
        "ema21_slope",
        "macd_hist",
        "macd_hist_delta",
        "macd_green_increasing",
        "macd_red_recovering",
        "macd_green_fading",
        "macd_red_deepening",
        "ema_derivative_1_diff",
        "ema_derivative_1_cross_positive",
        "ema_derivative_1_cross_negative",
    }
    assert expected.issubset(set(names))
    assert "ema_stack_bullish" not in names
    assert "ema_stack_bearish" not in names


def test_hybrid_sharpe_momentum_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid_momentum") == "hybrid_sharpe_momentum"
    builder = get_strategy_feature_builder("hybrid_sharpe_momentum")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "macd_hist",
        "macd_hist_delta",
        "ema_derivative_2_diff",
        "ema_derivative_3_diff",
        "ema_derivative_2_cross_positive",
        "ema_derivative_2_cross_negative",
        "ema_derivative_3_cross_positive",
        "ema_derivative_3_cross_negative",
        "ema_slope_alignment",
        "ema_spread_balance",
    }
    assert expected.issubset(set(names))


def test_hybrid_sharpe_selective_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-selective") == "hybrid_sharpe_selective"
    builder = get_strategy_feature_builder("hybrid_sharpe_selective")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema3_slope",
        "ema9_slope",
        "macd_hist",
        "macd_green_increasing",
        "macd_red_recovering",
        "ema_derivative_1_diff",
        "ema_derivative_1_cross_positive",
        "ema_derivative_1_cross_negative",
    }
    assert expected.issubset(set(names))
    assert "ema21_slope" not in names
    assert "macd_hist_delta" not in names


def test_hybrid_sharpe_regime_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-regime") == "hybrid_sharpe_regime"
    builder = get_strategy_feature_builder("hybrid_sharpe_regime")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "trend_20",
        "vol_20",
        "ret_1",
        "ret_3",
        "stoch_rsi_norm",
        "stoch_velocity",
        "ema_slope_alignment",
    }
    assert expected.issubset(set(names))


def test_hybrid_sharpe_volume_flow_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-volume-flow") == "hybrid_sharpe_volume_flow"
    builder = get_strategy_feature_builder("hybrid_sharpe_volume_flow")
    names = builder.names()
    expected = {
        "ema3",
        "ema9",
        "ema21",
        "ema3_9_spread",
        "ema9_21_spread",
        "macd_hist",
        "macd_hist_delta",
        "volume",
        "volume_ma20",
        "volume_spike_ratio",
        "signed_volume_pressure",
        "volume_volatility_coupling",
        "volume_trend_coupling",
    }
    assert expected.issubset(set(names))
    assert "ema_stack_bullish" not in names
    assert "ema_stack_bearish" not in names


def test_hybrid_sharpe_volume_regime_feature_set_is_supported() -> None:
    assert normalize_feature_set("hybrid-volume-regime") == "hybrid_sharpe_volume_regime"
    builder = get_strategy_feature_builder("hybrid_sharpe_volume_regime")
    names = builder.names()
    expected = {
        "volume",
        "volume_ma20",
        "volume_spike_ratio",
        "high_volume_regime",
        "low_volume_regime",
        "trend_in_high_volume",
        "pullback_in_low_volume",
    }
    assert expected.issubset(set(names))


def test_rsi_thresholds_feature_set_is_supported() -> None:
    assert normalize_feature_set("rsi_thresholds") == "rsi_thresholds"
    builder = get_strategy_feature_builder("rsi_thresholds")
    assert builder.names() == ["rsi_over_70", "rsi_below_30"]


def test_stoch_rsi_thresholds_feature_set_is_supported() -> None:
    assert normalize_feature_set("stoch_rsi_thresholds") == "stoch_rsi_thresholds"
    builder = get_strategy_feature_builder("stoch_rsi_thresholds")
    assert builder.names() == ["stoch_rsi_over_80", "stoch_rsi_under_20"]


def test_close_hold_reversion_feature_set_is_supported() -> None:
    assert normalize_feature_set("eod_reversion") == "close_hold_reversion"
    builder = get_strategy_feature_builder("close_hold_reversion")
    names = builder.names()
    expected = {
        "ret_1",
        "ret_3",
        "stoch_rsi_norm",
        "stoch_low_zone",
        "stoch_high_zone",
        "bb_percent_b",
        "bb_reversion_distance",
        "vwap_mid_dislocation",
        "close_to_vwap_low_atr",
        "close_to_vwap_high_atr",
        "bearish_exhaustion",
        "bullish_exhaustion",
    }
    assert expected.issubset(set(names))


def test_close_hold_momentum_feature_set_is_supported() -> None:
    assert normalize_feature_set("overnight_momentum") == "close_hold_momentum"
    builder = get_strategy_feature_builder("close_hold_momentum")
    names = builder.names()
    expected = {
        "ret_1",
        "ret_3",
        "ret_5",
        "ema3_9_spread",
        "ema9_21_spread",
        "ema3_slope",
        "ema9_slope",
        "ema21_slope",
        "macd_hist",
        "macd_hist_delta",
        "macd_acceleration",
        "vwap_breakout_strength",
        "vwap_breakdown_strength",
        "momentum_alignment",
    }
    assert expected.issubset(set(names))
