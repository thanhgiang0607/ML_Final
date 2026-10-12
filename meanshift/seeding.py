"""Bin seeding: chọn điểm xuất phát trên lưới cỡ h thay vì dùng toàn bộ dữ liệu.

Người phụ trách: D  ·  Test: tests/test_seeding.py  ·  Branch: oop-bandwidth-seeding

Phải cho CÙNG kết quả với sklearn.cluster.get_bin_seeds (thứ tự các seed có thể khác).
"""

import warnings

import numpy as np


def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """Làm tròn mỗi điểm về ô lưới gần nhất: round(x / bin_size).
    Giữ các ô có >= min_bin_freq điểm, trả về toạ độ ô * bin_size.

    X            : ndarray (n, d)
    bin_size     : float — bằng 0 thì trả về X
    min_bin_freq : int
    return       : ndarray (m, d)

    Lưu ý để khớp sklearn:
      - toạ độ ô được ép kiểu float32 trước khi nhân bin_size
      - nếu số ô == n (không giảm được) -> warnings.warn(...) và trả về X
    """

    X = np.asarray(X)
    if bin_size == 0:
        return X

    # Ô lưới của từng điểm, rồi đếm số điểm trong mỗi ô.
    cells, counts = np.unique(np.round(X / bin_size), axis=0, return_counts=True)

    # Chỉ giữ các ô đủ điểm; ép float32 trước khi nhân bin_size để khớp sklearn.
    seeds = cells[counts >= min_bin_freq].astype(np.float32)

    if len(seeds) == len(X):
        warnings.warn(
            "Binning data failed with provided bin_size=%f, using data points as seeds."
            % bin_size
        )
        return X
    return seeds * bin_size

