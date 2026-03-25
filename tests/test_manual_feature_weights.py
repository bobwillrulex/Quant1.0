import pytest

from main import parse_manual_feature_weights


def test_parse_manual_feature_weights_allows_negative_values_when_sum_is_one() -> None:
    weights = parse_manual_feature_weights("[1.2, -0.2]", expected_feature_count=2)
    assert weights == [1.2, -0.2]


def test_parse_manual_feature_weights_rejects_total_not_equal_to_one() -> None:
    with pytest.raises(ValueError, match="add up to 1.0"):
        parse_manual_feature_weights("[0.2, -0.1]", expected_feature_count=2)
