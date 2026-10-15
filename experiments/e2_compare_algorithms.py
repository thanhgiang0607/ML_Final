"""E2: So sánh Mean-Shift với K-Means, DBSCAN, GMM trên 5 bộ dữ liệu giả lập.

Người phụ trách: E  ·  Branch: e2-algorithms

Thiết lập
---------
- Dữ liệu: 5 bộ × `--seeds` seed (mặc định 10), chuẩn hoá StandardScaler riêng cho từng (bộ, seed).
- Chọn tham số trên SEED 0, giữ nguyên cho seed 1–9 (theo PHAN_CONG.md):
    * Mean-Shift (h mặc định): h = h0 = estimate_bandwidth(X, quantile=0.3) — không biết số cụm.
    * Mean-Shift (h chọn):     h = α·h0, α chọn trong ALPHA_GRID cho ARI cao nhất trên seed 0.
    * DBSCAN:                  min_samples=5, eps chọn trong EPS_GRID cho ARI cao nhất trên seed 0.
    * K-Means, GMM:            k = số cụm đúng (oracle k) — lợi thế so với các phương pháp tự tìm số cụm.
- Bảng tổng hợp chỉ tính trên seed 1–9 (bỏ seed 0 đã dùng để chọn tham số).
  Chạy với --seeds 1 thì tổng hợp trên seed 0 (chỉ để chạy thử).
- DBSCAN gán -1 cho nhiễu; ARI/NMI coi -1 như một nhãn, còn số cụm không đếm -1.
- Thời gian chỉ đo bước fit, chạy 1 luồng (threadpool_limits=1).

Đầu ra (results/e2/)
--------------------
    e2_params.csv     tham số đã chọn trên seed 0 cho từng (bộ, thuật toán)
    e2_runs.csv       mỗi dòng: bộ, seed, thuật toán, ARI, NMI, số cụm, tỷ lệ nhiễu, thời gian
    e2_summary.csv    ARI, NMI, số cụm, thời gian: trung bình ± std theo (bộ, thuật toán)
    e2_grid.png       lưới 5 bộ × 5 thuật toán, seed 0
    e2_ari_bar.png    ARI trung bình (thanh sai số = std) theo bộ × thuật toán
    e2_metadata.json  cấu hình, phiên bản thư viện, SHA256 dữ liệu đầu vào

Chạy: python3 experiments/e2_compare_algorithms.py [--seeds 10]
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from meanshift import MeanShiftCustom, estimate_bandwidth  # noqa: E402
from common import DATASETS, TRUE_K, load_synthetic, out_dir, synthetic_path  # noqa: E402

OUT = out_dir("e2")
METHODS = ["MeanShift (h mặc định)", "MeanShift (h chọn)", "KMeans", "DBSCAN", "GMM"]
ALPHA_GRID = [0.25, 0.5, 0.75, 1.0, 1.5]
EPS_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
MIN_SAMPLES = 5


def load_scaled(name, seed):
    X, y = load_synthetic(name, seed)
    return StandardScaler().fit_transform(X), y


def make_model(method, name, seed, params, h0):
    if method == "MeanShift (h mặc định)":
        return MeanShiftCustom(bandwidth=h0, bin_seeding=True)
    if method == "MeanShift (h chọn)":
        return MeanShiftCustom(bandwidth=params["alpha"] * h0, bin_seeding=True)
    if method == "KMeans":
        return KMeans(n_clusters=TRUE_K[name], n_init=10, random_state=seed)
    if method == "DBSCAN":
        return DBSCAN(eps=params["eps"], min_samples=MIN_SAMPLES)
    if method == "GMM":
        return GaussianMixture(n_components=TRUE_K[name], covariance_type="full", n_init=5, random_state=seed)
    raise ValueError(method)


def fit_labels(model, X):
    start = time.perf_counter()
    model.fit(X)
    elapsed = time.perf_counter() - start
    labels = model.predict(X) if isinstance(model, GaussianMixture) else model.labels_
    return np.asarray(labels), elapsed


def select_params(name):
    """Chọn α (Mean-Shift) và eps (DBSCAN) cho ARI cao nhất trên seed 0."""
    X, y = load_scaled(name, 0)
    h0 = estimate_bandwidth(X)
    alpha_scores = {}
    for alpha in ALPHA_GRID:
        try:
            labels = MeanShiftCustom(bandwidth=alpha * h0, bin_seeding=True).fit_predict(X)
            alpha_scores[alpha] = adjusted_rand_score(y, labels)
        except ValueError:  # h quá nhỏ, không seed nào có láng giềng
            alpha_scores[alpha] = -np.inf
    eps_scores = {eps: adjusted_rand_score(y, DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(X).labels_)
                  for eps in EPS_GRID}
    alpha = max(alpha_scores, key=alpha_scores.get)
    eps = max(eps_scores, key=eps_scores.get)
    return {"alpha": alpha, "alpha_ari_seed0": alpha_scores[alpha],
            "eps": eps, "eps_ari_seed0": eps_scores[eps]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10, help="số seed, bắt đầu từ 0 (mặc định 10)")
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds phải là số nguyên dương")
    missing = [synthetic_path(n, s) for n in DATASETS for s in range(args.seeds)
               if not synthetic_path(n, s).is_file()]
    if missing:
        sys.exit(f"Thiếu {len(missing)} file dữ liệu giả, ví dụ {missing[0]}. "
                 "Chạy trước: python3 data/make_synthetic.py")

    rows, param_rows, examples = [], [], {}
    with threadpool_limits(limits=1):
        params = {}
        for name in DATASETS:
            params[name] = select_params(name)
            p = params[name]
            param_rows += [
                {"dataset": name, "method": "MeanShift (h mặc định)", "param": "alpha", "value": 1.0, "ari_seed0": np.nan},
                {"dataset": name, "method": "MeanShift (h chọn)", "param": "alpha", "value": p["alpha"], "ari_seed0": p["alpha_ari_seed0"]},
                {"dataset": name, "method": "KMeans", "param": "k", "value": TRUE_K[name], "ari_seed0": np.nan},
                {"dataset": name, "method": "DBSCAN", "param": "eps", "value": p["eps"], "ari_seed0": p["eps_ari_seed0"]},
                {"dataset": name, "method": "GMM", "param": "k", "value": TRUE_K[name], "ari_seed0": np.nan},
            ]
            print(f"[chọn tham số] {name}: α = {p['alpha']}, eps = {p['eps']}", flush=True)

        for name in DATASETS:
            for seed in range(args.seeds):
                X, y = load_scaled(name, seed)
                h0 = estimate_bandwidth(X)
                for method in METHODS:
                    labels, elapsed = fit_labels(make_model(method, name, seed, params[name], h0), X)
                    if seed == 0:
                        examples[(name, method)] = (X, labels)
                    rows.append({
                        "dataset": name, "seed": seed, "method": method,
                        "ari": float(adjusted_rand_score(y, labels)),
                        "nmi": float(normalized_mutual_info_score(y, labels)),
                        "n_clusters": len(set(labels.tolist()) - {-1}),
                        "noise_fraction": float(np.mean(labels == -1)),
                        "runtime_seconds": elapsed,
                    })
                print(f"[{name}] seed {seed + 1}/{args.seeds} xong", flush=True)

    pd.DataFrame(param_rows).to_csv(OUT / "e2_params.csv", index=False)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "e2_runs.csv", index=False)

    eval_runs = runs[runs.seed >= 1] if args.seeds > 1 else runs
    summary = eval_runs.groupby(["dataset", "method"], sort=False).agg(
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std"),
        n_clusters_mean=("n_clusters", "mean"), n_clusters_std=("n_clusters", "std"),
        noise_fraction_mean=("noise_fraction", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"), runtime_seconds_std=("runtime_seconds", "std"),
    ).reset_index()
    summary.to_csv(OUT / "e2_summary.csv", index=False)

    # Lưới 5 bộ × 5 thuật toán, seed 0
    fig, axes = plt.subplots(len(DATASETS), len(METHODS), figsize=(18, 17), constrained_layout=True)
    for ri, name in enumerate(DATASETS):
        for ci, method in enumerate(METHODS):
            X, labels = examples[(name, method)]
            row = runs[(runs.dataset == name) & (runs.seed == 0) & (runs.method == method)].iloc[0]
            ax = axes[ri, ci]
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=4, cmap="tab20", linewidths=0)
            ax.set_title(f"{name} — {method}\nARI = {row.ari:.3f}, số cụm = {row.n_clusters}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(OUT / "e2_grid.png", dpi=150)
    plt.close(fig)

    # ARI trung bình ± std
    mean = summary.pivot(index="dataset", columns="method", values="ari_mean").reindex(index=DATASETS, columns=METHODS)
    std = summary.pivot(index="dataset", columns="method", values="ari_std").reindex(index=DATASETS, columns=METHODS)
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    mean.plot(kind="bar", yerr=std.fillna(0), ax=ax, capsize=2, rot=0)
    seeds_used = "seed 1–9" if args.seeds > 1 else "seed 0"
    ax.set_title(f"ARI trung bình theo bộ dữ liệu ({seeds_used})")
    ax.set_ylabel("ARI"); ax.set_xlabel("Bộ dữ liệu"); ax.legend(fontsize=8)
    fig.savefig(OUT / "e2_ari_bar.png", dpi=150)
    plt.close(fig)

    used = [synthetic_path(n, s) for n in DATASETS for s in range(args.seeds)]
    metadata = {
        "experiment": "E2 – so sánh thuật toán", "seeds": list(range(args.seeds)),
        "summary_seeds": sorted(eval_runs.seed.unique().tolist()), "datasets": DATASETS,
        "selection": "α (Mean-Shift) và eps (DBSCAN) chọn theo ARI cao nhất trên seed 0, giữ nguyên cho các seed khác",
        "alpha_grid": ALPHA_GRID, "eps_grid": EPS_GRID, "min_samples": MIN_SAMPLES,
        "selected_params": params, "true_k": TRUE_K,
        "meanshift": "MeanShiftCustom(flat, kdtree, bin_seeding=True), h0 = meanshift.estimate_bandwidth(X, quantile=0.3)",
        "oracle_k": "KMeans, GMM được biết trước số cụm đúng",
        "standardization": "StandardScaler riêng cho từng (bộ, seed)",
        "noise": "Nhãn -1 của DBSCAN được tính như một nhãn trong ARI/NMI; không đếm vào số cụm",
        "threadpool_limit": 1,
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in used},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
    }
    (OUT / "e2_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nARI trung bình (" + seeds_used + "):")
    print(mean.round(3).to_string())
    print(f"\nĐã lưu kết quả E2 tại {OUT}")


if __name__ == "__main__":
    main()
