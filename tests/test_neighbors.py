"""Test cho meanshift/neighbors.py — người phụ trách: C.

Đây là vài test MẪU để bắt đầu; C bổ sung thêm test của mình.
"""
import numpy as np
import pytest

from meanshift.neighbors import BruteForceSearch, KDTreeSearch, make_neighbor_search


def as_sets(result):
    return [set(idx.tolist()) for idx, _ in result]


def test_brute_and_kdtree_find_same_neighbors():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 2)
    Q = rng.randn(20, 2)
    a = BruteForceSearch().fit(X).query_radius(Q, 0.7)
    b = KDTreeSearch().fit(X).query_radius(Q, 0.7)
    assert as_sets(a) == as_sets(b)


def test_sq_dist_is_squared_distance():
    X = np.array([[0.0, 0.0], [3.0, 4.0]])
    for search in (BruteForceSearch(), KDTreeSearch()):
        (idx, sq), = search.fit(X).query_radius(np.array([[0.0, 0.0]]), 10.0)
        order = np.argsort(idx)
        np.testing.assert_allclose(sq[order], [0.0, 25.0], atol=1e-9)


def test_point_on_boundary_is_included():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    for search in (BruteForceSearch(), KDTreeSearch()):
        (idx, _), = search.fit(X).query_radius(np.array([[0.0, 0.0]]), 1.0)
        assert set(idx.tolist()) == {0, 1}


def test_make_neighbor_search():
    assert isinstance(make_neighbor_search("brute"), BruteForceSearch)
    assert isinstance(make_neighbor_search("kdtree"), KDTreeSearch)
    with pytest.raises(ValueError):
        make_neighbor_search("abc")
