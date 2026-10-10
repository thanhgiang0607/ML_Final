"""E1 – Chạy đối chiếu MeanShiftCustom với sklearn.cluster.MeanShift.

Người phụ trách: E  ·  Branch: e1-compare

Yêu cầu
-------
Dữ liệu : data/synthetic/<bộ>_seed<k>.csv  (5 bộ × 10 seed; sinh bằng data/make_synthetic.py)
Cấu hình: h = alpha · estimate_bandwidth(X), alpha ∈ {0.25, 0.5, 1.0}
          bin_seeding ∈ {False, True};  neighbor_search ∈ {'brute', 'kdtree'}
          sklearn chạy với n_jobs=1, cùng h, cùng bin_seeding
Mỗi lần chạy ghi lại:
    - số cụm của hai bên
    - ARI giữa nhãn hai bên (sklearn.metrics.adjusted_rand_score)
    - độ lệch tâm lớn nhất sau ghép cặp Hungarian (scipy.optimize.linear_sum_assignment), chia h
    - nhãn có giống hệt không
    - n_iter_ hai bên, thời gian chạy hai bên
    - ARI của mỗi bên so với nhãn đúng
Đầu ra (results/e1/):
    e1_runs.csv     – mỗi dòng một lần chạy
    e1_summary.csv  – gộp theo (bộ, alpha, bin_seeding, search)
    e1_examples.png – hình 5 hàng × 2 cột: custom | sklearn (seed 0, alpha 0.5)
In ra màn hình: % cùng số cụm, % nhãn giống hệt, ARI nhỏ nhất, lệch tâm lớn nhất,
và liệt kê các lần chạy KHÁC sklearn (nếu có) để phân tích nguyên nhân.

Chạy: python3 experiments/e1_compare_sklearn.py [--seeds 2]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "synthetic"
OUT = ROOT / "results" / "e1"
DATASETS = ["blobs", "blobs_varied", "blobs_aniso", "moons", "circles"]
ALPHAS = [0.25, 0.5, 1.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    if not DATA.exists():
        sys.exit("Chưa có dữ liệu giả. Chạy trước: python3 data/make_synthetic.py")
    # TODO(E): chạy lưới cấu hình ở trên, lưu csv + hình vào OUT
    raise NotImplementedError


if __name__ == "__main__":
    main()
