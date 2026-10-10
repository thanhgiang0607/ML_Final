"""Test cho meanshift/seeding.py — người phụ trách: D.

Đây là vài test MẪU để bắt đầu; D bổ sung thêm test của mình.
"""
import numpy as np
import pytest
from sklearn.cluster import get_bin_seeds as sk_get_bin_seeds
from sklearn.datasets import make_blobs

from meanshift.seeding import get_bin_seeds


def sort_rows(a):
    return a[np.lexsort(a.T[::-1])]


@pytest.mark.parametrize("bin_size, min_bin_freq", [(1.0, 1), (0.5, 3)])
def test_matches_sklearn(bin_size, min_bin_freq):
    X, _ = make_blobs(n_samples=500, centers=4, random_state=6)
    ours = get_bin_seeds(X, bin_size, min_bin_freq)
    theirs = sk_get_bin_seeds(X, bin_size, min_bin_freq)
    np.testing.assert_allclose(sort_rows(ours), sort_rows(theirs))


def test_bin_size_zero_returns_X():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_array_equal(get_bin_seeds(X, 0), X)
