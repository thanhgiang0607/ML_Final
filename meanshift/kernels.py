"""Hàm nhân (kernel) dùng trong bước dịch chuyển Mean-Shift.

Người phụ trách: B  ·  Test: tests/test_kernels.py  ·  Branch: oop-kernels

Mỗi kernel trả về trọng số g(||x - x_i||^2 / h^2) cho các láng giềng và cho biết
bán kính cần tìm láng giềng (support).
"""
from abc import ABC, abstractmethod

import numpy as np


class Kernel(ABC):
    """Giao diện chung cho mọi kernel. KHÔNG đổi tên hoặc tham số các hàm dưới đây."""

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
        # TODO(B): trả về h
        raise NotImplementedError

    def weights(self, sq_dist, bandwidth):
        # TODO(B): mảng toàn số 1, cùng kích thước sq_dist
        raise NotImplementedError


class GaussianKernel(Kernel):
    """g(u) = exp(-u / 2) với u = sq_dist / h^2, cắt ở bán kính cutoff * h."""

    name = "gaussian"

    def __init__(self, cutoff=3.0):
        # TODO(B): kiểm tra cutoff > 0 (ValueError nếu sai), lưu vào self.cutoff
        raise NotImplementedError

    def support(self, bandwidth):
        # TODO(B): trả về cutoff * h
        raise NotImplementedError

    def weights(self, sq_dist, bandwidth):
        # TODO(B): exp(-0.5 * sq_dist / h^2)
        raise NotImplementedError


def make_kernel(kernel, gaussian_cutoff=3.0):
    """'flat' -> FlatKernel(), 'gaussian' -> GaussianKernel(cutoff), đối tượng Kernel -> giữ nguyên.

    Tên khác -> ValueError.
    """
    # TODO(B)
    raise NotImplementedError
