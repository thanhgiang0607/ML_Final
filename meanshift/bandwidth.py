"""Ước lượng băng thông h khi người dùng không truyền vào.

Người phụ trách: D  ·  Test: tests/test_bandwidth.py  ·  Branch: oop-bandwidth-seeding

Phải cho CÙNG kết quả với sklearn.cluster.estimate_bandwidth.
"""
import numpy as np
from scipy.spatial import cKDTree


def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0):
    """h = trung bình (trên mọi điểm) của khoảng cách tới láng giềng thứ k,
    với k = max(1, int(quantile · n)) và tính cả chính điểm đó.

    X            : array-like (n, d)
    quantile     : float trong [0, 1] (sai -> ValueError)
    n_samples    : int hoặc None — nếu có, chỉ lấy ngẫu nhiên n_samples điểm
                   (np.random.RandomState(random_state).permutation)
    return       : float

    Gợi ý: scipy.spatial.cKDTree(X).query(X, k=k)
    """
    # TODO(D)
    if not 0 <= quantile <= 1:
        raise ValueError(f"quantile phải nằm trong [0, 1], nhận được {quantile!r}")

    X = np.asarray(X, dtype=float)

    if n_samples is not None:
        if n_samples < 1:
            raise ValueError(f"n_samples phải >= 1 hoặc None, nhận được {n_samples!r}")
        rng = np.random.RandomState(random_state)
        X = X[rng.permutation(X.shape[0])[:n_samples]]

    k = max(1, int(X.shape[0] * quantile))

    # Khoảng cách tới láng giềng thứ k (tính cả chính điểm đó, khoảng cách 0).
    dist, _ = cKDTree(X).query(X, k=k)
    kth = dist if k == 1 else dist[:, -1]  # k=1 -> mảng 1-D; k>1 -> cột cuối là xa nhất
    return float(np.mean(kth))