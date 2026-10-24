"""
E4 - Phân tích độ nhạy của bandwidth trong Mean Shift.

Thí nghiệm:
- 5 bộ dữ liệu synthetic.
- 10 seeds mặc định.
- 15 giá trị alpha theo np.geomspace(0.1, 3.0, 15).
- Bổ sung alpha = 1.0 để đánh giá trực tiếp bandwidth mặc định.
- Bandwidth sử dụng: h = alpha * h0.
- h0 = estimate_bandwidth(X).
- Mean Shift sử dụng bin_seeding=True.

Đầu ra:
- results/e4/e4_synthetic_runs.csv
- results/e4/e4_synthetic_summary.csv
- results/e4/e4_synthetic.png
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Thêm thư mục gốc của project vào đường dẫn import.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import DATASETS, load_synthetic
from meanshift.bandwidth import estimate_bandwidth
from meanshift.core import MeanShiftCustom


# Giữ nguyên 15 giá trị alpha theo yêu cầu ban đầu.
ALPHAS = np.geomspace(0.1, 3.0, 15)

# Số lần lặp mặc định cho mỗi bộ dữ liệu.
DEFAULT_SEEDS = 10


def run_one(X, y, dataset, seed, alpha):
    """
    Chạy Mean Shift với một giá trị alpha.

    h0 = estimate_bandwidth(X)
    h = alpha * h0

    Trả về các chỉ số đánh giá và thời gian chạy.
    """
    # Ước lượng bandwidth mặc định.
    h0 = estimate_bandwidth(X)

    # Bảo vệ trường hợp bandwidth không hợp lệ.
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError(
            f"Bandwidth h0 không hợp lệ: {h0}. "
            f"Dataset={dataset}, seed={seed}"
        )

    # Bandwidth thực tế của lần chạy.
    bandwidth = float(alpha * h0)

    # Khởi tạo và chạy Mean Shift.
    model = MeanShiftCustom(
        bandwidth=bandwidth,
        bin_seeding=True
    )

    start_time = time.perf_counter()
    labels_pred = model.fit_predict(X)
    elapsed_time = time.perf_counter() - start_time

    # Đánh giá kết quả phân cụm so với nhãn thật.
    ari = adjusted_rand_score(y, labels_pred)
    nmi = normalized_mutual_info_score(y, labels_pred)
    n_clusters = len(np.unique(labels_pred))

    return {
        "dataset": dataset,
        "seed": seed,
        "alpha": float(alpha),
        "h0": float(h0),
        "bandwidth": bandwidth,
        "n_clusters": int(n_clusters),
        "ari": float(ari),
        "nmi": float(nmi),
        "time_seconds": float(elapsed_time),
    }


def summarize_results(results):
    """
    Tổng hợp kết quả theo dataset và alpha.

    Các chỉ số được tổng hợp gồm:
    - Số cụm trung bình và độ lệch chuẩn.
    - ARI trung bình và độ lệch chuẩn.
    - NMI trung bình và độ lệch chuẩn.
    - Thời gian chạy trung bình và độ lệch chuẩn.
    """
    summary = (
        results.groupby(["dataset", "alpha"], as_index=False)
        .agg(
            n_clusters_mean=("n_clusters", "mean"),
            n_clusters_std=("n_clusters", "std"),
            ari_mean=("ari", "mean"),
            ari_std=("ari", "std"),
            nmi_mean=("nmi", "mean"),
            nmi_std=("nmi", "std"),
            time_seconds_mean=("time_seconds", "mean"),
            time_seconds_std=("time_seconds", "std"),
        )
    )

    # Khi chỉ chạy một seed, độ lệch chuẩn không xác định.
    std_columns = [
        "n_clusters_std",
        "ari_std",
        "nmi_std",
        "time_seconds_std",
    ]
    summary[std_columns] = summary[std_columns].fillna(0.0)

    return summary


def plot_results(summary, output_path):
    """
    Vẽ hai biểu đồ:
    1. Số cụm trung bình theo alpha.
    2. ARI trung bình theo alpha.

    Mỗi bộ dữ liệu có một đường.
    Vùng mờ thể hiện +/- một độ lệch chuẩn.
    Đường alpha = 1 biểu thị bandwidth mặc định.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    metrics = [
        (
            "n_clusters_mean",
            "n_clusters_std",
            "Số cụm trung bình",
        ),
        (
            "ari_mean",
            "ari_std",
            "ARI trung bình",
        ),
    ]

    for ax, (mean_col, std_col, ylabel) in zip(axes, metrics):
        for dataset in DATASETS:
            subset = (
                summary[summary["dataset"] == dataset]
                .sort_values("alpha")
            )

            x = subset["alpha"].to_numpy(dtype=float)
            y = subset[mean_col].to_numpy(dtype=float)
            std = subset[std_col].to_numpy(dtype=float)

            ax.plot(
                x,
                y,
                marker="o",
                markersize=3,
                label=dataset,
            )

            ax.fill_between(
                x,
                y - std,
                y + std,
                alpha=0.15,
            )

        # Đánh dấu bandwidth mặc định alpha = 1.
        ax.axvline(
            x=1.0,
            linestyle="--",
            linewidth=1.5,
            label="alpha = 1 (default)",
        )

        ax.set_xscale("log")
        ax.set_xlabel("Alpha (h = alpha * h0)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " theo alpha")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("E4 - Độ nhạy của bandwidth trong Mean Shift")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="E4 - Phân tích độ nhạy của bandwidth trong Mean Shift"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help="Số seed chạy cho mỗi bộ dữ liệu (mặc định: 10).",
    )
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds phải lớn hơn hoặc bằng 1.")

    seeds = range(args.seeds)

    # Giữ nguyên 15 alpha ban đầu và bổ sung chính xác alpha = 1.0.
    # set() tránh chạy trùng nếu lưới sau này đã có giá trị 1.0.
    alphas_to_run = sorted(set(ALPHAS.tolist() + [1.0]))

    rows = []
    total_runs = len(DATASETS) * len(seeds) * len(alphas_to_run)
    run_number = 0

    print("=" * 65)
    print("E4 - PHÂN TÍCH ĐỘ NHẠY CỦA BANDWIDTH")
    print(f"Số bộ dữ liệu: {len(DATASETS)}")
    print(f"Số seed mỗi bộ dữ liệu: {args.seeds}")
    print(f"Số giá trị alpha: {len(alphas_to_run)}")
    print(f"Tổng số lần chạy dự kiến: {total_runs}")
    print("Alpha được chạy:", [round(a, 6) for a in alphas_to_run])
    print("=" * 65)

    for dataset in DATASETS:
        for seed in seeds:
            X, y = load_synthetic(dataset, seed)

            for alpha in alphas_to_run:
                run_number += 1

                print(
                    f"[{run_number}/{total_runs}] "
                    f"dataset={dataset}, seed={seed}, "
                    f"alpha={alpha:.4f}",
                    flush=True,
                )

                row = run_one(
                    X=X,
                    y=y,
                    dataset=dataset,
                    seed=seed,
                    alpha=alpha,
                )
                rows.append(row)

    # Lưu kết quả chi tiết.
    results = pd.DataFrame(rows)

    result_dir = ROOT / "results" / "e4"
    result_dir.mkdir(parents=True, exist_ok=True)

    runs_path = result_dir / "e4_synthetic_runs.csv"
    summary_path = result_dir / "e4_synthetic_summary.csv"
    plot_path = result_dir / "e4_synthetic.png"

    results.to_csv(runs_path, index=False)

    # Tổng hợp kết quả theo dataset và alpha.
    summary = summarize_results(results)
    summary.to_csv(summary_path, index=False)

    # Vẽ biểu đồ.
    plot_results(summary, plot_path)

    # Tìm alpha tốt nhất theo ARI trung bình trong từng dataset.
    best_by_dataset = (
        summary.loc[
            summary.groupby("dataset")["ari_mean"].idxmax()
        ]
        .sort_values("dataset")
    )

    # Lấy kết quả alpha = 1.0 để so sánh trực tiếp với mặc định.
    default_results = summary[
        np.isclose(summary["alpha"], 1.0)
    ].copy()

    print("\n" + "=" * 65)
    print("TÓM TẮT E4")
    print("=" * 65)
    print(f"Số lần chạy thực tế: {len(results)}")
    print(f"Kết quả chi tiết: {runs_path}")
    print(f"Kết quả tổng hợp: {summary_path}")
    print(f"Biểu đồ: {plot_path}")

    print("\nAlpha tốt nhất theo ARI trung bình của từng dataset:")
    print(
        best_by_dataset[
            ["dataset", "alpha", "ari_mean", "n_clusters_mean"]
        ].to_string(index=False)
    )

    print("\nSo sánh bandwidth mặc định (alpha = 1.0) với alpha tốt nhất:")
    comparison_rows = []

    for dataset in DATASETS:
        best_row = best_by_dataset[
            best_by_dataset["dataset"] == dataset
        ].iloc[0]

        default_row = default_results[
            default_results["dataset"] == dataset
        ].iloc[0]

        comparison_rows.append(
            {
                "dataset": dataset,
                "default_alpha": 1.0,
                "default_ari": default_row["ari_mean"],
                "best_alpha": best_row["alpha"],
                "best_ari": best_row["ari_mean"],
                "ari_difference": (
                    best_row["ari_mean"] - default_row["ari_mean"]
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    print(comparison.to_string(index=False))

    print("\nHoàn tất thí nghiệm E4.")


if __name__ == "__main__":
    main()