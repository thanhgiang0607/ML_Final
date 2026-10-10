"""Test cho meanshift/bandwidth.py — người phụ trách: D.

Đây là vài test MẪU để bắt đầu; D bổ sung thêm test của mình.
"""
import pytest
from sklearn.cluster import estimate_bandwidth as sk_estimate_bandwidth
from sklearn.datasets import make_blobs

from meanshift.bandwidth import estimate_bandwidth


@pytest.mark.parametrize("quantile", [0.1, 0.3, 0.5])
def test_matches_sklearn(quantile):
    X, _ = make_blobs(n_samples=500, centers=4, random_state=5)
    assert estimate_bandwidth(X, quantile=quantile) == pytest.approx(
        sk_estimate_bandwidth(X, quantile=quantile)
    )


def test_matches_sklearn_with_subsample():
    X, _ = make_blobs(n_samples=500, centers=4, random_state=5)
    assert estimate_bandwidth(X, n_samples=200, random_state=1) == pytest.approx(
        sk_estimate_bandwidth(X, n_samples=200, random_state=1)
    )


def test_invalid_quantile():
    with pytest.raises(ValueError):
        estimate_bandwidth([[0.0, 0.0], [1.0, 1.0]], quantile=1.5)
