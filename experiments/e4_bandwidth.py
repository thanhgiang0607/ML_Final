"""E4 - Khảo sát độ nhạy của Mean Shift theo bandwidth.

Chạy:
    python experiments/e4_bandwidth.py
Chạy nhanh với 1 seed:
    python experiments/e4_bandwidth.py --seeds 1
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

# Thêm thư mục gốc của project vào đường dẫn tìm kiếm module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import DATASETS, ROOT, load_synthetic
from meanshift.bandwidth import estimate_bandwidth
from meanshift.core import MeanShiftCustom


# 15 giá trị alpha từ 0.1 đến 3 theo thang logarit
ALPHAS = np.geomspace(0.1, 3.0, 15)
DEFAULT_SEEDS = list(range(10))


def load_scaled(dataset, seed):
    """Đọc dữ liệu synthetic và chuẩn hóa giống thí nghiệm E2."""
    X, y = load_synthetic(dataset, seed)
    X = StandardScaler().fit_transform(X)
    return X, y


def run_one(dataset, seed, alpha):
    """Chạy một cấu hình bandwidth và trả về kết quả."""
    X, y_true = load_scaled(dataset, seed)

    # Bandwidth mặc định được ước lượng từ dữ liệu
    h0 = estimate_bandwidth(X)

    # Tránh chạy thuật toán với bandwidth không hợp lệ
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError(
            f"Bandwidth mặc định không hợp lệ: dataset={dataset}, "
            f"seed={seed}, h0={h0}"
        )

    h = float(alpha * h0)

    # Đo thời gian 3 lần, lấy trung vị
    elapsed_times = []
    labels = None

    for _ in range(3):
        model = MeanShiftCustom(
            bandwidth=h,
            bin_seeding=True,
        )

        start = time.perf_counter()
        model.fit(X)
        elapsed_times.append(time.perf_counter() - start)

        labels = model.labels_

    y_pred = np.asarray(labels)
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)

    return {
        "dataset": dataset,
        "seed": seed,
        "alpha": float(alpha),
        "h0": float(h0),
        "bandwidth": h,
        "n_clusters": int(n_clusters),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "time_seconds": float(np.median(elapsed_times)),
    }


def summarize_results(runs):
    """Tổng hợp trung bình và độ lệch chuẩn theo dataset, alpha."""
    summary = (
        runs.groupby(["dataset", "alpha"], as_index=False)
        .agg(
            n_clusters_mean=("n_clusters", "mean"),
            n_clusters_std=("n_clusters", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            time_mean=("time_seconds", "mean"),
            time_std=("time_seconds", "std"),
        )
    )

    # Với trường hợp chỉ có 1 seed, độ lệch chuẩn là NaN.
    std_columns = [col for col in summary.columns if col.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0)

    return summary


def plot_results(summary, output_path):
    """Vẽ số cụm và ARI theo alpha, trung bình ± độ lệch chuẩn."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for dataset in DATASETS:
        data = summary[summary["dataset"] == dataset].sort_values("alpha")

        x = data["alpha"].to_numpy()

        # Số cụm trung bình ± độ lệch chuẩn
        y_clusters = data["n_clusters_mean"].to_numpy()
        err_clusters = data["n_clusters_std"].to_numpy()

        axes[0].plot(x, y_clusters, marker="o", label=dataset)
        axes[0].fill_between(
            x,
            y_clusters - err_clusters,
            y_clusters + err_clusters,
            alpha=0.15,
        )

        # ARI trung bình ± độ lệch chuẩn
        y_ari = data["ari_mean"].to_numpy()
        err_ari = data["ari_std"].to_numpy()

        axes[1].plot(x, y_ari, marker="o", label=dataset)
        axes[1].fill_between(
            x,
            y_ari - err_ari,
            y_ari + err_ari,
            alpha=0.15,
        )

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(
            x=1.0,
            linestyle="--",
            linewidth=1.5,
            label="alpha = 1 (default)",
        )
        ax.set_xlabel("Alpha (h = alpha × h0)")
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Độ nhạy số lượng cụm theo bandwidth")
    axes[0].set_ylabel("Số cụm")
    axes[1].set_title("Độ nhạy ARI theo bandwidth")
    axes[1].set_ylabel("ARI")

    # Tránh lặp nhãn alpha=1 trong chú giải
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="E4: Khảo sát độ nhạy của Mean Shift theo bandwidth."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Số seed chạy cho mỗi dataset (mặc định: 10).",
    )
    args = parser.parse_args()

    if args.seeds < 1 or args.seeds > 10:
        parser.error("--seeds phải nằm trong khoảng từ 1 đến 10.")

    seeds = DEFAULT_SEEDS[:args.seeds]

    output_dir = ROOT / "results" / "e4"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    total = len(DATASETS) * len(seeds) * len(ALPHAS)
    current = 0

    for dataset in DATASETS:
        for seed in seeds:
            for alpha in ALPHAS:
                current += 1
                print(
                    f"[{current}/{total}] "
                    f"dataset={dataset}, seed={seed}, alpha={alpha:.4f}"
                )

                result = run_one(dataset, seed, alpha)
                all_runs.append(result)

    runs = pd.DataFrame(all_runs)
    summary = summarize_results(runs)

    runs_path = output_dir / "e4_synthetic_runs.csv"
    summary_path = output_dir / "e4_synthetic_summary.csv"
    plot_path = output_dir / "e4_synthetic.png"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_results(summary, plot_path)

    # So sánh alpha=1 với bandwidth tốt nhất theo ARI trung bình.
    comparison = (
        summary.groupby("alpha", as_index=False)["ari_mean"]
        .mean()
        .sort_values("ari_mean", ascending=False)
    )
    best = comparison.iloc[0]

    default_idx = (comparison["alpha"] - 1.0).abs().idxmin()
    default = comparison.loc[default_idx]

    print("\n===== TÓM TẮT E4 =====")
    print(f"Số lần chạy: {len(runs)}")
    print(f"Kết quả chi tiết: {runs_path}")
    print(f"Kết quả tổng hợp: {summary_path}")
    print(f"Biểu đồ: {plot_path}")
    print(
        f"Alpha tốt nhất theo ARI trung bình: "
        f"{best['alpha']:.4f} (ARI={best['ari_mean']:.4f})"
    )
    print(
        f"Alpha gần 1 nhất: {default['alpha']:.4f} "
        f"(ARI={default['ari_mean']:.4f})"
    )


if __name__ == "__main__":
    main()