from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StopLossStrategy(str, Enum):
    NONE = "none"
    ATR = "atr"
    MODEL_INVALIDATION = "model_invalidation"
    TIME_DECAY = "time_decay"
    FIXED_PERCENTAGE = "fixed_percentage"


MODEL_MAE_DEFAULT = 0.0547


@dataclass
class StopLossConfig:
    strategy: StopLossStrategy = StopLossStrategy.NONE
    fixed_pct: float = 2.0
    atr_multiplier: float = 1.5
    model_mae: float = MODEL_MAE_DEFAULT
    time_decay_bars: int = 25


def parse_stop_loss_strategy(raw: str | None) -> StopLossStrategy:
    value = (raw or StopLossStrategy.NONE.value).strip().lower()
    try:
        return StopLossStrategy(value)
    except ValueError:
        raise ValueError("Unknown stop-loss strategy.")


def validate_fixed_stop_pct(value: float) -> float:
    pct = float(value)
    if pct <= 0:
        raise ValueError("Fixed stop-loss percent must be > 0.")
    return pct


def stop_loss_price(
    *,
    strategy: StopLossStrategy,
    action: str,
    reference_price: float,
    expected_return: float = 0.0,
    fixed_pct: float = 2.0,
    atr_fraction: float = 0.0,
    atr_multiplier: float = 1.5,
    model_mae: float = MODEL_MAE_DEFAULT,
) -> float | None:
    if reference_price <= 0:
        return None
    normalized_action = action.strip().upper()
    if normalized_action not in ("BUY", "SELL"):
        return None
    if strategy in (StopLossStrategy.NONE, StopLossStrategy.TIME_DECAY):
        return None

    is_long = normalized_action == "BUY"
    if strategy == StopLossStrategy.FIXED_PERCENTAGE:
        distance = abs(fixed_pct) / 100.0
    elif strategy == StopLossStrategy.ATR:
        distance = abs(atr_multiplier * atr_fraction)
    else:
        threshold = expected_return - (2.0 * model_mae)
        return reference_price * (1.0 + threshold if is_long else 1.0 - threshold)

    if is_long:
        return reference_price * (1.0 - distance)
    return reference_price * (1.0 + distance)
