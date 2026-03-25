import pytest

from main import parse_manual_feature_weights


def test_parse_manual_feature_weights_allows_negative_values_when_sum_is_one() -> None:
    weights = parse_manual_feature_weights("[1.2, -0.2]", expected_feature_count=2)
    assert weights == [1.2, -0.2]


def test_parse_manual_feature_weights_normalizes_when_total_is_not_one() -> None:
    weights = parse_manual_feature_weights("[2, 1]", expected_feature_count=2)
    assert weights == [pytest.approx(2 / 3), pytest.approx(1 / 3)]


def test_parse_manual_feature_weights_rejects_zero_total() -> None:
    with pytest.raises(ValueError, match="total cannot be zero"):
        parse_manual_feature_weights("[0.2, -0.2]", expected_feature_count=2)
