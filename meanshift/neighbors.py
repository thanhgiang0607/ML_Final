"""Tìm láng giềng trong bán kính r: bản brute-force và bản KD-Tree.
"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial import cKDTree

class NeighborSearch(ABC):
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
        X = np.asarray(self.X_, dtype=float)
        P = np.atleast_2d(np.asarray(points, dtype=float))
        r2 = float(r) ** 2
        x_sq = np.einsum("ij,ij->i", X, X)  
        step = max(1, int(self.chunk_size))
        result = []
        for start in range(0, len(P), step):
            chunk = P[start:start + step]
            p_sq = np.einsum("ij,ij->i", chunk, chunk)
            d2 = p_sq[:, None] - 2.0 * (chunk @ X.T) + x_sq[None, :]
            np.maximum(d2, 0.0, out=d2)  # loại sai số âm do làm tròn
            for i in range(len(chunk)):
                idx = np.nonzero(d2[i] <= r2 * (1 + 1e-9) + 1e-12)[0]
                diff = X[idx] - chunk[i]
                sq = np.einsum("ij,ij->i", diff, diff)
                keep = sq <= r2
                result.append((idx[keep], sq[keep]))
        return result

class KDTreeSearch(NeighborSearch):
    """Dùng scipy.spatial.cKDTree (dựng cây trong fit, truy vấn bằng query_ball_point)."""

    name = "kdtree"

    def fit(self, X):
        super().fit(X)
        self.tree_ = cKDTree(np.asarray(X, dtype=float))
        return self

    def query_radius(self, points, r):
        X = np.asarray(self.X_, dtype=float)
        P = np.atleast_2d(np.asarray(points, dtype=float))
        r2 = float(r) ** 2
        neighbors = self.tree_.query_ball_point(P, r * (1 + 1e-9))
        result = []
        for i, ids in enumerate(neighbors):
            idx = np.sort(np.asarray(ids, dtype=np.intp))
            diff = X[idx] - P[i]
            sq = np.einsum("ij,ij->i", diff, diff)
            keep = sq <= r2
            result.append((idx[keep], sq[keep]))
        return result


def make_neighbor_search(name):
    """'brute' -> BruteForceSearch(), 'kdtree' -> KDTreeSearch(), đối tượng NeighborSearch -> giữ nguyên.

    Tên khác -> ValueError.
    """
    if isinstance(name, NeighborSearch):
        return name
    if name == "brute":
        return BruteForceSearch()
    if name == "kdtree":
        return KDTreeSearch()
    raise ValueError(f"Tên bộ tìm láng giềng không hợp lệ: {name!r} (chỉ chấp nhận 'brute' hoặc 'kdtree')")