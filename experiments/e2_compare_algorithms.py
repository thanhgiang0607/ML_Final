"""E2: So sánh bốn thuật toán phân cụm trên năm bộ dữ liệu tổng hợp.

KMeans và GaussianMixture được cung cấp số cụm đúng đã biết từ bộ sinh dữ liệu
(oracle k), tạo lợi thế so với các phương pháp tự xác định số cụm. Không thuật
toán nào được điều chỉnh bằng nhãn đánh giá. DBSCAN gán -1 cho nhiễu; ARI/NMI
được tính trực tiếp với các nhãn đó, nên điểm số phản ánh cả quyết định loại
điểm khỏi cụm và không thể so sánh như thể -1 là một cụm thông thường.

Giới hạn: dữ liệu chỉ có hai chiều và được chuẩn hóa chung; cấu hình DBSCAN cố
định có thể không phù hợp mọi mật độ. MeanShift phụ thuộc bandwidth và có thể
nhạy với seed/binning; KMeans ưu tiên cụm lồi, GMM giả định hỗn hợp Gaussian,
DBSCAN phụ thuộc eps/min_samples và có thể gộp/tách cụm theo mật độ. Thời gian
là một lần chạy đơn trên môi trường hiện tại, không phải benchmark thống kê.

Chạy: python experiments/e2_compare_algorithms.py [--seeds 10] [--output-dir PATH]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import estimate_bandwidth
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meanshift import MeanShiftCustom

DATA = ROOT / "data" / "synthetic"
DEFAULT_OUT = ROOT / "results" / "e2"
DATASETS = ["blobs", "blobs_varied", "blobs_aniso", "moons", "circles"]
KNOWN_K = {"blobs": 4, "blobs_varied": 3, "blobs_aniso": 3, "moons": 2, "circles": 2}
METHODS = ["MeanShiftCustom", "KMeans", "DBSCAN", "GaussianMixture"]
CONFIG = {
    "MeanShiftCustom": {"kernel": "flat", "neighbor_search": "kdtree", "bandwidth": "0.5 * sklearn estimate_bandwidth(X, quantile=0.3, n_samples=None)", "bin_seeding": True, "cluster_all": True},
    "KMeans": {"n_clusters": "generator-known k (oracle)", "n_init": 10},
    "DBSCAN": {"eps": 0.2, "min_samples": 5},
    "GaussianMixture": {"n_components": "generator-known k (oracle)", "covariance_type": "full", "n_init": 5, "random_state": "seed"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds, starting at 0 (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be a positive integer")
    required = [DATA / f"{name}_seed{seed}.csv" for name in DATASETS for seed in range(args.seeds)]
    missing = [p for p in required if not p.is_file()]
    if missing:
        sys.exit("Thiếu dữ liệu tổng hợp. Tạo trước bằng `python data/make_synthetic.py`. File thiếu: " + ", ".join(str(p) for p in missing[:8]) + (f" (và {len(missing)-8} file khác)" if len(missing) > 8 else ""))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, cache, example_labels = [], {}, {}
    total = len(required)
    with threadpool_limits(limits=1):
        for file_index, path in enumerate(required, 1):
            name = path.name.split("_seed")[0]
            seed = int(path.stem.rsplit("seed", 1)[1])
            frame = pd.read_csv(path)
            if not {"x", "y", "label"}.issubset(frame.columns) or frame.empty:
                sys.exit(f"Dữ liệu không hợp lệ: {path} (cần các cột x,y,label và ít nhất một dòng)")
            X = frame[["x", "y"]].to_numpy(dtype=float)
            y = frame["label"].to_numpy()
            Xs = StandardScaler().fit_transform(X)
            if seed == 0:
                cache[name] = Xs
            bandwidth = 0.5 * estimate_bandwidth(Xs, quantile=0.3, n_samples=None)
            if not np.isfinite(bandwidth) or bandwidth <= 0:
                sys.exit(f"Bandwidth không hợp lệ ({bandwidth}) cho {path}")
            estimators = {
                "MeanShiftCustom": MeanShiftCustom(bandwidth=bandwidth, kernel="flat", neighbor_search="kdtree", bin_seeding=True, cluster_all=True),
                "KMeans": KMeans(n_clusters=KNOWN_K[name], n_init=10, random_state=seed),
                "DBSCAN": DBSCAN(eps=0.2, min_samples=5),
                "GaussianMixture": GaussianMixture(n_components=KNOWN_K[name], covariance_type="full", n_init=5, random_state=seed),
            }
            for method in METHODS:
                model = estimators[method]
                start = time.perf_counter()
                fitted = model.fit(Xs)
                elapsed = time.perf_counter() - start
                labels = fitted.labels_ if method != "GaussianMixture" else fitted.predict(Xs)
                if seed == 0:
                    example_labels[(name, method)] = labels.copy()
                n_clusters = len(set(labels) - {-1})
                noise_fraction = float(np.mean(labels == -1))
                rows.append({
                    "dataset": name, "seed": seed, "method": method,
                    "ari": float(adjusted_rand_score(y, labels)),
                    "nmi": float(normalized_mutual_info_score(y, labels)),
                    "n_clusters": int(n_clusters), "noise_fraction": noise_fraction,
                    "runtime_seconds": elapsed,
                    "n_iter": getattr(fitted, "n_iter_", np.nan),
                    "converged": getattr(fitted, "converged_", np.nan),
                    "bandwidth": bandwidth if method == "MeanShiftCustom" else np.nan,
                    "oracle_k": KNOWN_K[name] if method in ("KMeans", "GaussianMixture") else False,
                })
            print(f"[{file_index}/{total}] Hoàn tất {name}, seed {seed}", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(args.output_dir / "e2_runs.csv", index=False)
    summary = runs.groupby(["dataset", "method"], sort=False).agg(
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std"),
        n_clusters_mean=("n_clusters", "mean"), n_clusters_std=("n_clusters", "std"),
        noise_fraction_mean=("noise_fraction", "mean"), noise_fraction_std=("noise_fraction", "std"),
        runtime_seconds_mean=("runtime_seconds", "mean"), runtime_seconds_std=("runtime_seconds", "std"),
    ).reset_index()
    summary.to_csv(args.output_dir / "e2_summary.csv", index=False)

    fig, axes = plt.subplots(len(DATASETS), len(METHODS), figsize=(15, 16), constrained_layout=True)
    for ri, dataset in enumerate(DATASETS):
        Xs = cache[dataset]
        for ci, method in enumerate(METHODS):
            row = runs[(runs.dataset == dataset) & (runs.seed == 0) & (runs.method == method)].iloc[0]
            labels = example_labels[(dataset, method)]
            ax = axes[ri, ci]
            ax.scatter(Xs[:, 0], Xs[:, 1], c=labels, s=5, cmap="tab20", linewidths=0)
            ax.set_title(f"{dataset} — {method}\nARI={row.ari:.3f}, k={row.n_clusters}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(args.output_dir / "e2_examples.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, metric in zip(axes, ["ari", "nmi"]):
        pivot = summary.pivot(index="dataset", columns="method", values=f"{metric}_mean").reindex(DATASETS)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Mean {metric.upper()} over {args.seeds} seeds")
        ax.set_ylabel(metric.upper()); ax.set_xlabel(""); ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(args.output_dir / "e2_score_comparison.png", dpi=160)
    plt.close(fig)

    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in required}
    metadata = {
        "experiment": "E2 algorithm comparison", "seeds": list(range(args.seeds)),
        "datasets": DATASETS, "input_files_sha256": hashes,
        "config": CONFIG, "standardization": "StandardScaler fit separately to each dataset/seed's two feature columns before every method",
        "metrics": ["adjusted_rand_score", "normalized_mutual_info_score", "n_clusters_excluding_-1", "noise_fraction", "runtime_seconds", "n_iter", "converged"],
        "noise_metric_interpretation": "DBSCAN label -1 is included as an ordinary predicted label by ARI/NMI; n_clusters excludes -1.",
        "oracle_k_disclosure": "KMeans and GaussianMixture use generator-known component counts; this is an oracle-k advantage, not a label-tuned setting.",
        "threadpool_limit": 1, "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    (args.output_dir / "e2_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã lưu kết quả E2 tại {args.output_dir}")


if __name__ == "__main__":
    main()
