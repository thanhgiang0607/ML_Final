"""Test cho meanshift/neighbors.py — người phụ trách: C.

Đây là vài test MẪU để bắt đầu; C bổ sung thêm test của mình.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

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
# ---------------------------------------------------------------------------
# Test bổ sung
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("search_cls", [BruteForceSearch, KDTreeSearch])
def test_diem_tren_bien_voi_r_bang_dung_khoang_cach(search_cls):
    """r = đúng khoảng cách tới một điểm -> điểm đó phải nằm trong kết quả."""
    # Dùng tam giác 3-4-5 để khoảng cách tính ra chính xác (không có sai số làm tròn)
    X = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [10.0, 0.0]])
    q = np.array([[0.0, 0.0]])

    r = float(np.linalg.norm(X[1] - q[0]))  # đúng 5.0
    assert r == 5.0

    (idx, sq), = search_cls().fit(X).query_radius(q, r)
    assert 1 in idx.tolist()                 # điểm nằm đúng trên biên được tính vào
    assert set(idx.tolist()) == {0, 1}       # điểm cách 10 thì không
    assert_allclose(sq[np.argsort(idx)], [0.0, 25.0])

    # Giảm r đi một chút thì điểm biên phải bị loại
    (idx2, _), = search_cls().fit(X).query_radius(q, r - 1e-6)
    assert set(idx2.tolist()) == {0}


def test_kha_nang_phan_chia_lo_tren_cung_idx_va_sq_dist():
    # Tạo dữ liệu ngẫu nhiên có thể tái lập
    rng = np.random.default_rng(42)
    X = rng.random((100, 2))
    points = rng.random((20, 2))

    # Thử nhiều bán kính và chunk_size nhỏ để kiểm tra chia lô
    for r in [0.05, 0.2, 0.5, 1.0]:
        for chunk_size in [1, 3, 7]:
            brute = BruteForceSearch(chunk_size=chunk_size).fit(X)
            kdtree = KDTreeSearch().fit(X)

            brute_results = brute.query_radius(points, r)
            kdtree_results = kdtree.query_radius(points, r)

            # Mỗi điểm truy vấn phải có cùng kết quả
            assert len(brute_results) == len(kdtree_results)

            for (idx_b, sq_b), (idx_k, sq_k) in zip(
                brute_results, kdtree_results
            ):
                # So sánh chỉ số láng giềng
                np.testing.assert_array_equal(idx_b, idx_k)

                # So sánh khoảng cách bình phương
                assert_allclose(sq_b, sq_k, rtol=1e-7, atol=1e-12)

@pytest.mark.parametrize("ten_la", ["abc", "", "KDTREE", None])
def test_ten_la_bao_value_error(ten_la):
    with pytest.raises(ValueError):
        make_neighbor_search(ten_la)

def test_diem_trung_toa_do():
    """Các điểm trùng tọa độ phải được trả về đầy đủ."""
    X = np.array([
        [0.0, 0.0],  # index 0
        [1.0, 1.0],  # index 1
        [0.0, 0.0],  # index 2: trùng index 0
        [2.0, 2.0],
        [1.0, 1.0],  # index 4: trùng index 1
    ])
    points = np.array([[0.0, 0.0], [1.0, 1.0]])

    expected = [
        (np.array([0, 2]), np.array([0.0, 0.0])),
        (np.array([1, 4]), np.array([0.0, 0.0])),
    ]

    for search in [
        BruteForceSearch(chunk_size=1),
        KDTreeSearch(),
    ]:
        search.fit(X)
        results = search.query_radius(points, r=0.0)

        assert len(results) == len(expected)

        for (idx, sq_dist), (expected_idx, expected_sq) in zip(
            results, expected
        ):
            np.testing.assert_array_equal(idx, expected_idx)
            assert_allclose(sq_dist, expected_sq, atol=1e-12)


def test_ban_kinh_bang_0():
    """Bán kính bằng 0 chỉ nhận các điểm trùng tọa độ với điểm truy vấn."""
    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],  # trùng với điểm truy vấn thứ nhất
        [2.0, 2.0],
    ])
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [5.0, 5.0],  # không có điểm trùng
    ])

    expected_indices = [
        np.array([0, 2]),
        np.array([1]),
        np.array([], dtype=int),
    ]
    expected_sq_distances = [
        np.array([0.0, 0.0]),
        np.array([0.0]),
        np.array([]),
    ]

    for search in [
        BruteForceSearch(chunk_size=1),
        KDTreeSearch(),
    ]:
        search.fit(X)
        results = search.query_radius(points, r=0.0)

        assert len(results) == len(points)

        for (idx, sq_dist), expected_idx, expected_sq in zip(
            results, expected_indices, expected_sq_distances
        ):
            np.testing.assert_array_equal(idx, expected_idx)
            assert_allclose(sq_dist, expected_sq, atol=1e-12)