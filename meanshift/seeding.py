"""Bin seeding: chọn điểm khởi đầu trên lưới cỡ h thay vì dùng toàn bộ dữ liệu."""
import warnings
from collections import Counter

import numpy as np


def get_bin_seeds(X, bin_size, min_bin_freq=1):
    """Làm tròn mỗi điểm về ô lưới gần nhất; giữ các ô có >= min_bin_freq điểm.

    Giữ đúng hành vi của sklearn (kể cả ép float32) để so sánh được.
    """
    if bin_size == 0:
        return X
    counts = Counter(map(tuple, np.round(X / bin_size)))
    seeds = np.array(
        [cell for cell, freq in counts.items() if freq >= min_bin_freq],
        dtype=np.float32,
    )
    if len(seeds) == len(X):
        warnings.warn(
            f"Bin seeding không giảm được số seed với bin_size={bin_size:f}, "
            "dùng toàn bộ điểm dữ liệu làm seed."
        )
        return X
    return seeds * bin_size
