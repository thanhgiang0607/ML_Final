"""MeanShiftCustom: cài đặt Mean-Shift từ đầu (NumPy/SciPy), API giống scikit-learn.

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
import numpy as np
from scipy.spatial import cKDTree

from .bandwidth import estimate_bandwidth
from .base import BaseClusterer, check_array
from .kernels import make_kernel
from .neighbors import make_neighbor_search
from .seeding import get_bin_seeds


class MeanShiftCustom(BaseClusterer):
    """Phân cụm Mean-Shift.

    Tham số
    -------
    bandwidth : float hoặc None
        Băng thông h. None -> estimate_bandwidth(X, quantile=0.3).
    kernel : {'flat', 'gaussian'}
    neighbor_search : {'brute', 'kdtree'}
    gaussian_cutoff : float
        Bán kính cắt (bội số của h) khi kernel='gaussian'.
    seeds : array (m, d) hoặc None
    bin_seeding : bool
    min_bin_freq : int
    cluster_all : bool
        False -> điểm cách tâm gần nhất > h được gán nhãn -1.
    max_iter : int
    tol : float
        Dừng khi độ dời <= tol·h.
    merge_radius : float hoặc None
        Ngưỡng gộp mode. None -> h.
    record_trajectories : bool
        Lưu quỹ đạo của từng seed vào trajectories_.

    Thuộc tính sau fit
    ------------------
    cluster_centers_, labels_, intensities_, bandwidth_, n_iter_,
    seed_n_iter_ (số vòng lặp từng seed), trajectories_ (nếu bật).
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

    # ------------------------------------------------------------------ fit
    def fit(self, X, y=None):
        X = check_array(X)
        h = self.bandwidth if self.bandwidth is not None else estimate_bandwidth(X)
        if h <= 0:
            raise ValueError("bandwidth phải > 0.")
        self.bandwidth_ = float(h)
        self._kernel = make_kernel(self.kernel, self.gaussian_cutoff)
        self._search = make_neighbor_search(self.neighbor_search).fit(X)

        seeds = self._init_seeds(X)
        modes, intensities, n_iter = self._shift_seeds(X, seeds)

        found = intensities > 0
        if not found.any():
            raise ValueError(
                f"Không có điểm nào trong bán kính h={h:f} quanh các seed. "
                "Hãy tăng bandwidth hoặc đổi cách chọn seed."
            )
        self.seed_n_iter_ = n_iter
        self.n_iter_ = int(n_iter.max())

        centers, center_intensity = self._merge_modes(modes[found], intensities[found])
        self.cluster_centers_ = centers
        self.intensities_ = center_intensity
        self.labels_ = self._assign_labels(X, centers, self.cluster_all)
        return self

    def predict(self, X):
        self._check_is_fitted()
        return self._assign_labels(check_array(X), self.cluster_centers_, cluster_all=True)

    # -------------------------------------------------------------- các bước
    def _init_seeds(self, X):
        if self.seeds is not None:
            return check_array(self.seeds)
        if self.bin_seeding:
            return np.asarray(get_bin_seeds(X, self.bandwidth_, self.min_bin_freq), dtype=np.float64)
        return X

    def _shift_seeds(self, X, seeds):
        """Dịch chuyển mọi seed song song (theo lô các seed chưa hội tụ).

        Trả về: vị trí mode, intensity (số điểm trong cửa sổ h ở bước cuối),
        số vòng lặp của từng seed — cùng quy ước với sklearn.
        """
        h = self.bandwidth_
        stop_thresh = self.tol * h
        radius = self._kernel.support(h)
        h_sq = h * h

        means = seeds.astype(np.float64, copy=True)
        intensities = np.zeros(len(means), dtype=np.int64)
        n_iter = np.zeros(len(means), dtype=np.int64)
        active = np.arange(len(means))
        trajectories = [[m.copy()] for m in means] if self.record_trajectories else None

        while active.size:
            neighbors = self._search.query_radius(means[active], radius)
            still_active = []
            for i, (idx, sq_dist) in zip(active, neighbors):
                if idx.size == 0:  # seed không có láng giềng nào -> bỏ
                    intensities[i] = 0
                    continue
                w = self._kernel.weights(sq_dist, h)
                old = means[i].copy()
                means[i] = w @ X[idx] / w.sum()
                intensities[i] = np.count_nonzero(sq_dist <= h_sq)
                if trajectories is not None:
                    trajectories[i].append(means[i].copy())
                if np.linalg.norm(means[i] - old) <= stop_thresh or n_iter[i] == self.max_iter:
                    continue
                n_iter[i] += 1
                still_active.append(i)
            active = np.asarray(still_active, dtype=np.intp)

        if trajectories is not None:
            self.trajectories_ = [np.array(t) for t in trajectories]
        return means, intensities, n_iter

    def _merge_modes(self, modes, intensities):
        """Gộp các mode gần nhau, giữ mode có intensity lớn hơn."""
        radius = self.merge_radius if self.merge_radius is not None else self.bandwidth_
        unique_modes = {}
        for m, c in zip(map(tuple, modes), intensities):
            unique_modes[m] = int(c)
        # Sắp theo (intensity, toạ độ) giảm dần — cùng quy tắc phá hoà với sklearn
        ordered = sorted(unique_modes.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        sorted_modes = np.array([m for m, _ in ordered])
        sorted_counts = np.array([c for _, c in ordered])

        tree = cKDTree(sorted_modes)
        keep = np.ones(len(sorted_modes), dtype=bool)
        for i, m in enumerate(sorted_modes):
            if keep[i]:
                keep[tree.query_ball_point(m, radius)] = False
                keep[i] = True
        return sorted_modes[keep], sorted_counts[keep]

    def _assign_labels(self, X, centers, cluster_all):
        dist, idx = cKDTree(centers).query(X, k=1)
        labels = np.asarray(idx, dtype=np.int64)
        if not cluster_all:
            labels[dist > self.bandwidth_] = -1
        return labels
