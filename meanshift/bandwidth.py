"""Ước lượng băng thông h (cùng công thức với sklearn.cluster.estimate_bandwidth)."""
import numpy as np
from scipy.spatial import cKDTree

from .base import check_array


def estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0):
    """h = trung bình (trên mọi điểm) của khoảng cách tới láng giềng thứ k,
    với k = quantile · n (tính cả chính điểm đó).
    """
    X = check_array(X)
    if not 0 <= quantile <= 1:
        raise ValueError("quantile phải nằm trong [0, 1].")
    if n_samples is not None:
        idx = np.random.RandomState(random_state).permutation(X.shape[0])[:n_samples]
        X = X[idx]
    k = max(1, int(X.shape[0] * quantile))
    dist, _ = cKDTree(X).query(X, k=k)
    dist = dist.reshape(len(X), -1)
    return float(dist.max(axis=1).mean())
