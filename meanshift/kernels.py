"""Hàm nhân (kernel) dùng trong bước dịch chuyển Mean-Shift.

Mỗi kernel trả về trọng số g(||x - x_i||^2 / h^2) cho các láng giềng và cho biết
bán kính cần tìm láng giềng (support).
"""
from abc import ABC, abstractmethod

import numpy as np


class Kernel(ABC):
    name = "base"

    @abstractmethod
    def support(self, bandwidth):
        """Bán kính tìm láng giềng.

        bandwidth : float  (h > 0)
        return    : float
        """

    @abstractmethod
    def weights(self, sq_dist, bandwidth):
        """Trọng số cho các láng giềng.

        sq_dist   : ndarray (k,) khoảng cách BÌNH PHƯƠNG từ tâm cửa sổ tới k láng giềng
        bandwidth : float
        return    : ndarray (k,) trọng số >= 0
        """


class FlatKernel(Kernel):
    """g = 1 với mọi điểm trong bán kính h (tương ứng nhân Epanechnikov của hàm mật độ).

    Bước dịch chuyển = trung bình cộng các điểm trong cửa sổ (giống sklearn).
    """

    name = "flat"

    def support(self, bandwidth):
        return bandwidth

    def weights(self, sq_dist, bandwidth):
        return np.ones_like(np.asarray(sq_dist, dtype=np.float64))


class GaussianKernel(Kernel):
    """g(u) = exp(-u / 2) với u = sq_dist / h^2, cắt ở bán kính cutoff * h."""

    name = "gaussian"

    def __init__(self, cutoff=3.0):
        if not cutoff > 0:
            raise ValueError(f"cutoff phải > 0, nhận được {cutoff!r}.")
        self.cutoff = cutoff

    def support(self, bandwidth):
        return self.cutoff * bandwidth

    def weights(self, sq_dist, bandwidth):
        sq_dist = np.asarray(sq_dist, dtype=np.float64)
        return np.exp(-0.5 * sq_dist / bandwidth**2)


def make_kernel(kernel, gaussian_cutoff=3.0):
    if isinstance(kernel, Kernel):
        return kernel
    if kernel == "flat":
        return FlatKernel()
    if kernel == "gaussian":
        return GaussianKernel(cutoff=gaussian_cutoff)
    raise ValueError(f"Kernel không hợp lệ: {kernel!r}. Chọn 'flat' hoặc 'gaussian'.")
