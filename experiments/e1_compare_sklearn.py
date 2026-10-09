"""E1 – Chạy đối chiếu MeanShiftCustom với sklearn.cluster.MeanShift.

Cùng dữ liệu, cùng bandwidth, cùng cách chọn seed -> so sánh:
    - số cụm tìm được
    - ARI giữa nhãn của hai bên (1 = giống hệt)
    - độ lệch tâm cụm sau khi ghép cặp (Hungarian), chia cho h
    - số vòng lặp, thời gian chạy
    - ARI của mỗi bên so với nhãn đúng (để thấy chất lượng phân cụm)

Chạy:  python3 experiments/e1_compare_sklearn.py            (10 seed, ~vài phút)
       python3 experiments/e1_compare_sklearn.py --seeds 2  (chạy nhanh)
Kết quả: results/e1/
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import MeanShift
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meanshift import MeanShiftCustom, estimate_bandwidth  # noqa: E402

DATA = ROOT / "data" / "synthetic"
OUT = ROOT / "results" / "e1"
DATASETS = ["blobs", "blobs_varied", "blobs_aniso", "moons", "circles"]
ALPHAS = [0.25, 0.5, 1.0]  # h = alpha * h_est (h_est: quantile = 0.3)


def load(name, seed):
    d = np.loadtxt(DATA / f"{name}_seed{seed}.csv", delimiter=",", skiprows=1)
    return d[:, :2], d[:, 2].astype(int)


def center_deviation(a, b):
    """Độ lệch lớn nhất giữa các tâm sau khi ghép cặp tối ưu (Hungarian)."""
    cost = cdist(a, b)
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols].max()


def timed_fit(model, X):
    t = time.perf_counter()
    model.fit(X)
    return time.perf_counter() - t


def run(n_seeds):
    rows = []
    total = len(DATASETS) * n_seeds * len(ALPHAS) * 2
    done = 0
    for name in DATASETS:
        for seed in range(n_seeds):
            X, y = load(name, seed)
            h_est = estimate_bandwidth(X)
            for alpha in ALPHAS:
                h = alpha * h_est
                for bin_seeding in (False, True):
                    ref = MeanShift(bandwidth=h, bin_seeding=bin_seeding, n_jobs=1)
                    t_ref = timed_fit(ref, X)
                    for search in ("brute", "kdtree"):
                        ours = MeanShiftCustom(bandwidth=h, neighbor_search=search, bin_seeding=bin_seeding)
                        t_ours = timed_fit(ours, X)
                        rows.append({
                            "dataset": name, "seed": seed, "alpha": alpha, "h": h,
                            "bin_seeding": bin_seeding, "search": search,
                            "k_custom": len(ours.cluster_centers_),
                            "k_sklearn": len(ref.cluster_centers_),
                            "k_true": len(np.unique(y)),
                            "ari_custom_vs_sklearn": adjusted_rand_score(ref.labels_, ours.labels_),
                            "center_dev_over_h": center_deviation(ours.cluster_centers_, ref.cluster_centers_) / h,
                            "labels_identical": bool(np.array_equal(ours.labels_, ref.labels_)),
                            "n_iter_custom": ours.n_iter_, "n_iter_sklearn": ref.n_iter_,
                            "ari_custom_vs_truth": adjusted_rand_score(y, ours.labels_),
                            "ari_sklearn_vs_truth": adjusted_rand_score(y, ref.labels_),
                            "time_custom_s": t_ours, "time_sklearn_s": t_ref,
                        })
                    done += 1
                    print(f"\r{done}/{total} cấu hình", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def summarize(df):
    df = df.assign(same_k=df.k_custom == df.k_sklearn)
    g = df.groupby(["dataset", "alpha", "bin_seeding", "search"])
    return g.agg(
        runs=("seed", "size"),
        same_k_pct=("same_k", lambda s: 100 * s.mean()),
        identical_labels_pct=("labels_identical", lambda s: 100 * s.mean()),
        ari_min=("ari_custom_vs_sklearn", "min"),
        center_dev_max=("center_dev_over_h", "max"),
        k_mean=("k_custom", "mean"),
        k_true=("k_true", "first"),
        ari_truth_custom=("ari_custom_vs_truth", "mean"),
        ari_truth_sklearn=("ari_sklearn_vs_truth", "mean"),
        time_custom_s=("time_custom_s", "median"),
        time_sklearn_s=("time_sklearn_s", "median"),
    ).reset_index()


def plot_examples(path, alpha=0.5, seed=0):
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(8, 3.2 * len(DATASETS)))
    for r, name in enumerate(DATASETS):
        X, _ = load(name, seed)
        h = alpha * estimate_bandwidth(X)
        ours = MeanShiftCustom(bandwidth=h).fit(X)
        ref = MeanShift(bandwidth=h).fit(X)
        for c, (model, title) in enumerate([(ours, "MeanShiftCustom"), (ref, "sklearn MeanShift")]):
            ax = axes[r, c]
            ax.scatter(X[:, 0], X[:, 1], c=model.labels_, s=4, cmap="tab10")
            ax.scatter(*model.cluster_centers_.T, c="black", marker="x", s=60)
            ax.set_title(f"{name} – {title}\n{len(model.cluster_centers_)} cụm, h = {h:.2f}", fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    if not DATA.exists():
        sys.exit("Chưa có dữ liệu giả. Chạy trước: python3 data/make_synthetic.py")

    OUT.mkdir(parents=True, exist_ok=True)
    df = run(args.seeds)
    df.to_csv(OUT / "e1_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT / "e1_summary.csv", index=False)
    plot_examples(OUT / "e1_examples.png")

    same_k = (df.k_custom == df.k_sklearn).mean() * 100
    identical = df.labels_identical.mean() * 100
    print(f"Tổng số lần chạy: {len(df)}")
    print(f"Cùng số cụm:        {same_k:.1f}%")
    print(f"Nhãn giống hệt:     {identical:.1f}%")
    print(f"ARI nhỏ nhất:       {df.ari_custom_vs_sklearn.min():.4f}")
    print(f"Lệch tâm lớn nhất:  {df.center_dev_over_h.max():.2e} · h")
    diff = df[~df.labels_identical]
    if len(diff):
        print("\nCác lần chạy KHÁC sklearn:")
        print(diff[["dataset", "seed", "alpha", "bin_seeding", "search", "k_custom", "k_sklearn",
                    "ari_custom_vs_sklearn"]].to_string(index=False))
    print(f"\nĐã lưu kết quả vào {OUT}")


if __name__ == "__main__":
    main()
