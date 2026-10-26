import pytest

from sentiment.metrics import classification_metrics


def test_metrics_match_a_hand_computed_confusion_matrix():
    # 2 true positives, 1 false positive, 1 false negative, 1 true negative.
    m = classification_metrics([1, 1, 1, 0, 0], [1, 1, 0, 1, 0])
    assert m.accuracy == pytest.approx(3 / 5)
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)


def test_metrics_are_zero_not_nan_when_nothing_is_predicted_positive():
    m = classification_metrics([1, 0], [0, 0])
    assert (m.accuracy, m.precision, m.recall, m.f1) == (0.5, 0.0, 0.0, 0.0)


def test_metrics_reject_mismatched_inputs():
    with pytest.raises(ValueError):
        classification_metrics([1, 0], [1])
    with pytest.raises(ValueError):
        classification_metrics([], [])
