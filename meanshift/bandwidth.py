"""Ước lượng băng thông h khi người dùng không truyền vào.

Người phụ trách: D  ·  Test: tests/test_bandwidth.py  ·  Branch: oop-bandwidth-seeding

Phải cho CÙNG kết quả với sklearn.cluster.estimate_bandwidth.
"""


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
    raise NotImplementedError
