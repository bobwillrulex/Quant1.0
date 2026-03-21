from __future__ import annotations

from typing import Callable, Dict

Row = Dict[str, float]
FeatureFn = Callable[[Row], float]
