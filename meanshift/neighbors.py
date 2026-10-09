"""Tìm láng giềng trong bán kính r: bản brute-force và bản KD-Tree.

Cả hai trả về, với mỗi điểm truy vấn, (chỉ số láng giềng, khoảng cách bình phương)
của các điểm dữ liệu có khoảng cách <= r.
"""
import numpy as np
from scipy.spatial import cKDTree


class NeighborSearch:
    def fit(self, X):
        self.X_ = X
        return self

    def query_radius(self, points, r):
        raise NotImplementedError


class BruteForceSearch(NeighborSearch):
    """Tính khoảng cách tới toàn bộ n điểm: O(n·d) mỗi truy vấn.

    Truy vấn theo lô (chunk) để giới hạn bộ nhớ ma trận khoảng cách.
    """

    name = "brute"

    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def query_radius(self, points, r):
        X = self.X_
        x_sq = np.einsum("ij,ij->i", X, X)
        r_sq = r * r
        out = []
        for start in range(0, len(points), self.chunk_size):
            P = points[start:start + self.chunk_size]
            p_sq = np.einsum("ij,ij->i", P, P)
            sq = p_sq[:, None] + x_sq[None, :] - 2.0 * P @ X.T
            np.maximum(sq, 0.0, out=sq)  # sai số làm tròn có thể cho giá trị âm nhỏ
            for row in sq:
                idx = np.flatnonzero(row <= r_sq)
                out.append((idx, row[idx]))
        return out


class KDTreeSearch(NeighborSearch):
    """Dùng scipy.spatial.cKDTree: ~O(log n + k) mỗi truy vấn khi h nhỏ."""

    name = "kdtree"

    def fit(self, X):
        super().fit(X)
        self.tree_ = cKDTree(X)
        return self

    def query_radius(self, points, r):
        X = self.X_
        out = []
        for p, idx in zip(points, self.tree_.query_ball_point(points, r)):
            idx = np.asarray(idx, dtype=np.intp)
            diff = X[idx] - p
            out.append((idx, np.einsum("ij,ij->i", diff, diff)))
        return out


def make_neighbor_search(name):
    if isinstance(name, NeighborSearch):
        return name
    if name == "brute":
        return BruteForceSearch()
    if name == "kdtree":
        return KDTreeSearch()
    raise ValueError(f"neighbor_search phải là 'brute' hoặc 'kdtree', nhận được {name!r}.")
