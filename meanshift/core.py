"""MeanShiftCustom: cài đặt Mean-Shift từ đầu (NumPy/SciPy), API giống scikit-learn.

Người phụ trách: A  ·  Test: tests/test_core.py  ·  Branch: oop-core

Luồng xử lý trong fit():
    1. Chọn băng thông h (gọi estimate_bandwidth nếu bandwidth=None).        <- D
    2. Chọn seed: toàn bộ X / get_bin_seeds / seeds người dùng truyền.       <- D
    3. Dịch chuyển từng seed:                                                 <- C, B
         láng giềng = search.query_radius(seed, kernel.support(h))
         w          = kernel.weights(sq_dist, h)
         seed mới   = sum(w * X[idx]) / sum(w)
       dừng khi ||Δ|| <= tol·h hoặc số vòng lặp == max_iter.
    4. Gộp mode: sắp theo (intensity, toạ độ) giảm dần, bỏ mode nằm trong
       merge_radius (mặc định h) của một mode mạnh hơn đã giữ.
       intensity = số điểm trong bán kính h quanh mode ở bước cuối.
    5. Gán nhãn: mỗi điểm thuộc tâm gần nhất (cluster_all=False -> xa hơn h thì nhãn -1).

Với kernel='flat', tol=1e-3, merge_radius=None, kết quả phải GIỐNG
sklearn.cluster.MeanShift để E1 so sánh được. Nên đọc mã nguồn sklearn:
    python3 -c "import sklearn.cluster._mean_shift as m; print(m.__file__)"
"""
from .base import BaseClusterer, check_array  # noqa: F401
from .bandwidth import estimate_bandwidth  # noqa: F401
from .kernels import make_kernel  # noqa: F401
from .neighbors import make_neighbor_search  # noqa: F401
from .seeding import get_bin_seeds  # noqa: F401


class MeanShiftCustom(BaseClusterer):
    """Phân cụm Mean-Shift.

    Tham số (KHÔNG đổi tên / thứ tự / giá trị mặc định)
    ----------------------------------------------------
    bandwidth : float hoặc None     Băng thông h. None -> estimate_bandwidth(X).
    kernel : 'flat' | 'gaussian'
    neighbor_search : 'brute' | 'kdtree'
    gaussian_cutoff : float         Bán kính cắt (bội số h) cho kernel Gaussian.
    seeds : ndarray (m, d) hoặc None
    bin_seeding : bool
    min_bin_freq : int
    cluster_all : bool
    max_iter : int
    tol : float                     Dừng khi độ dời <= tol·h.
    merge_radius : float hoặc None  Ngưỡng gộp mode. None -> h.
    record_trajectories : bool      Lưu quỹ đạo từng seed vào trajectories_.

    Thuộc tính sau fit
    ------------------
    cluster_centers_ : ndarray (k, d)
    labels_          : ndarray (n,) int
    intensities_     : ndarray (k,) int
    bandwidth_       : float
    n_iter_          : int  (số vòng lặp lớn nhất trong các seed)
    seed_n_iter_     : ndarray (số seed,)
    trajectories_    : list[ndarray]  (chỉ khi record_trajectories=True)
    """

    def __init__(
        self,
        bandwidth=None,
        kernel="flat",
        neighbor_search="kdtree",
        gaussian_cutoff=3.0,
        seeds=None,
        bin_seeding=False,
        min_bin_freq=1,
        cluster_all=True,
        max_iter=300,
        tol=1e-3,
        merge_radius=None,
        record_trajectories=False,
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.neighbor_search = neighbor_search
        self.gaussian_cutoff = gaussian_cutoff
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.max_iter = max_iter
        self.tol = tol
        self.merge_radius = merge_radius
        self.record_trajectories = record_trajectories

    def fit(self, X, y=None):
        """Chạy thuật toán (bước 1–5 ở đầu file). Trả về self."""
        # TODO(A)
        raise NotImplementedError

    def predict(self, X):
        """Gán mỗi điểm của X vào tâm cụm gần nhất. Trả về ndarray (n,) int."""
        self._check_is_fitted()
        # TODO(A)
        raise NotImplementedError

    # fit_predict(X) đã có sẵn trong BaseClusterer (= fit(X).labels_)

    # ------------------------------------------------- hàm nội bộ (gợi ý chia nhỏ)
    def _init_seeds(self, X):
        """Bước 2. Trả về ndarray (m, d)."""
        raise NotImplementedError

    def _shift_seeds(self, X, seeds):
        """Bước 3. Trả về (modes (m, d), intensities (m,), n_iter (m,))."""
        raise NotImplementedError

    def _merge_modes(self, modes, intensities):
        """Bước 4. Trả về (centers (k, d), center_intensities (k,))."""
        raise NotImplementedError

    def _assign_labels(self, X, centers, cluster_all):
        """Bước 5. Trả về labels (n,)."""
        raise NotImplementedError
