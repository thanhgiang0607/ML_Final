"""Hàm nhân (kernel) dùng trong bước dịch chuyển Mean-Shift.

Mỗi kernel trả về trọng số g(||x - x_i||^2 / h^2) cho các láng giềng và cho biết
bán kính hỗ trợ (support) — chỉ các điểm trong bán kính này mới được xét.
"""
import numpy as np


class Kernel:
    name = "base"

    def support(self, bandwidth):
        """Bán kính tìm láng giềng."""
        raise NotImplementedError

    def weights(self, sq_dist, bandwidth):
        """Trọng số cho các láng giềng, sq_dist = khoảng cách bình phương."""
        raise NotImplementedError


class FlatKernel(Kernel):
    """g = 1 trong bán kính h (tương ứng nhân Epanechnikov của hàm mật độ).

    Bước dịch chuyển = trung bình cộng các điểm trong cửa sổ — giống sklearn.
    """

    name = "flat"

    def support(self, bandwidth):
        return bandwidth

    def weights(self, sq_dist, bandwidth):
        return np.ones_like(sq_dist)


class GaussianKernel(Kernel):
    """g(u) = exp(-u / 2), cắt ở bán kính cutoff * h để tìm láng giềng nhanh.

    Phần bị cắt có trọng số < exp(-cutoff^2 / 2) (≈ 0.011 với cutoff = 3).
    """

    name = "gaussian"

    def __init__(self, cutoff=3.0):
        if cutoff <= 0:
            raise ValueError("cutoff phải > 0.")
        self.cutoff = cutoff

    def support(self, bandwidth):
        return self.cutoff * bandwidth

    def weights(self, sq_dist, bandwidth):
        return np.exp(-0.5 * sq_dist / bandwidth**2)


def make_kernel(kernel, gaussian_cutoff=3.0):
    if isinstance(kernel, Kernel):
        return kernel
    if kernel == "flat":
        return FlatKernel()
    if kernel == "gaussian":
        return GaussianKernel(gaussian_cutoff)
    raise ValueError(f"kernel phải là 'flat' hoặc 'gaussian', nhận được {kernel!r}.")
