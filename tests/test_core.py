"""Test cho meanshift/core.py — người phụ trách: A.

Đây là vài test MẪU để bắt đầu; A bổ sung thêm test của mình.
"""
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from meanshift import MeanShiftCustom


@pytest.fixture
def two_blobs():
    return make_blobs(n_samples=300, centers=[[0, 0], [10, 10]], cluster_std=0.5, random_state=0)


@pytest.mark.parametrize("kernel", ["flat", "gaussian"])
@pytest.mark.parametrize("search", ["brute", "kdtree"])
def test_two_far_blobs_give_two_clusters(two_blobs, kernel, search):
    X, y = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0, kernel=kernel, neighbor_search=search).fit(X)
    assert len(ms.cluster_centers_) == 2
    assert adjusted_rand_score(y, ms.labels_) == 1.0


def test_single_point():
    ms = MeanShiftCustom(bandwidth=1.0).fit([[3.0, 4.0]])
    np.testing.assert_allclose(ms.cluster_centers_, [[3.0, 4.0]])
    assert ms.labels_.tolist() == [0]


def test_predict_matches_labels(two_blobs):
    X, _ = two_blobs
    ms = MeanShiftCustom(bandwidth=2.0).fit(X)
    np.testing.assert_array_equal(ms.predict(X), ms.labels_)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="chưa được fit"):  # lỗi từ _check_is_fitted()
        MeanShiftCustom().predict([[0.0, 0.0]])
