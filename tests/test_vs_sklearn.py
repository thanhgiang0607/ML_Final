"""Đối chiếu MeanShiftCustom với sklearn.cluster.MeanShift — người phụ trách: E.

Đây là test MẪU; E bổ sung thêm (nhiều bộ dữ liệu, bin_seeding, cluster_all=False...).
"""
import numpy as np
import pytest
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

from meanshift import MeanShiftCustom


@pytest.mark.parametrize("bin_seeding", [False, True])
@pytest.mark.parametrize("search", ["brute", "kdtree"])
def test_flat_kernel_matches_sklearn(bin_seeding, search):
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=0.8, random_state=7)
    h = 1.2
    ours = MeanShiftCustom(bandwidth=h, neighbor_search=search, bin_seeding=bin_seeding).fit(X)
    ref = MeanShift(bandwidth=h, bin_seeding=bin_seeding).fit(X)
    assert len(ours.cluster_centers_) == len(ref.cluster_centers_)
    np.testing.assert_allclose(ours.cluster_centers_, ref.cluster_centers_, atol=1e-10)
    np.testing.assert_array_equal(ours.labels_, ref.labels_)
