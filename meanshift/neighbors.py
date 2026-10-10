"""Tìm láng giềng trong bán kính r: bản brute-force và bản KD-Tree.

Người phụ trách: C  ·  Test: tests/test_neighbors.py  ·  Branch: oop-neighbors

Cả hai lớp có CÙNG giao diện để core.py đổi qua lại mà không sửa code.
"""
from abc import ABC, abstractmethod


class NeighborSearch(ABC):
    """Giao diện chung. KHÔNG đổi tên hoặc tham số các hàm dưới đây."""

    name = "base"

    def fit(self, X):
        """Nạp dữ liệu X: ndarray (n, d). Trả về self."""
        self.X_ = X
        return self

    @abstractmethod
    def query_radius(self, points, r):
        """Tìm láng giềng cho từng điểm truy vấn.

        points : ndarray (m, d)
        r      : float — lấy các điểm có khoảng cách <= r (tính cả điểm nằm đúng trên biên)
        return : list gồm m phần tử, phần tử thứ i là tuple (idx, sq_dist):
                   idx     : ndarray (k_i,) chỉ số các điểm trong X_ cách points[i] <= r
                   sq_dist : ndarray (k_i,) khoảng cách BÌNH PHƯƠNG tương ứng
        """


class BruteForceSearch(NeighborSearch):
    """Tính khoảng cách tới toàn bộ n điểm (vector hoá bằng NumPy).

    Gợi ý: xử lý theo lô (chunk_size điểm truy vấn mỗi lần) để không tràn bộ nhớ.
    """

    name = "brute"

    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def query_radius(self, points, r):
        # TODO(C)
        raise NotImplementedError


class KDTreeSearch(NeighborSearch):
    """Dùng scipy.spatial.cKDTree (dựng cây trong fit, truy vấn bằng query_ball_point)."""

    name = "kdtree"

    def fit(self, X):
        # TODO(C): gọi super().fit(X) rồi dựng self.tree_
        raise NotImplementedError

    def query_radius(self, points, r):
        # TODO(C)
        raise NotImplementedError


def make_neighbor_search(name):
    """'brute' -> BruteForceSearch(), 'kdtree' -> KDTreeSearch(), đối tượng NeighborSearch -> giữ nguyên.

    Tên khác -> ValueError.
    """
    # TODO(C)
    raise NotImplementedError
