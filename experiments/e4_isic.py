"""E4 - Khảo sát độ nhạy của Mean Shift theo bandwidth.

Gồm 2 phần:
    Phần 1 - dữ liệu giả lập: h = alpha × h0, 5 bộ × 10 seed.
    Phần 2 - ISIC tune: 100 ảnh tune, đặc trưng 3D Lab, quét h theo đơn vị Lab.

Chạy:
    python3 experiments/e4_bandwidth.py                  # cả 2 phần
    python3 experiments/e4_bandwidth.py --part synthetic # chỉ phần 1
    python3 experiments/e4_bandwidth.py --part isic      # chỉ phần 2 (cần data/download_isic.py)
Chạy nhanh để thử:
    python3 experiments/e4_bandwidth.py --seeds 1        # phần 1: 1 seed
    python3 experiments/e4_bandwidth.py --part isic --quick
                                                         # phần 2: 10 ảnh đầu, kết quả ghi ra thư mục tạm
                                                         # (không đụng vào results/e4/)

Đầu ra phần 1 (results/e4/):
    e4_synthetic_runs.csv           mỗi dòng: bộ, seed, alpha, h0, h, số cụm, số cụm thật, ARI, NMI, thời gian
    e4_synthetic_summary.csv        trung bình ± std qua các seed theo (bộ, alpha)
    e4_synthetic_alpha_summary.csv  theo từng bộ: alpha tốt nhất, ARI tại alpha = 1, khoảng alpha an toàn
    e4_synthetic.png                số cụm và ARI theo alpha
    e4_metadata.json                cấu hình, phiên bản thư viện, SHA256 dữ liệu đầu vào

Đầu ra phần 2 (results/e4/):
    e4_isic_runs.csv                mỗi dòng: ảnh, h, Dice, IoU, số cụm, thời gian
    e4_isic_summary.csv             trung bình ± std qua các ảnh theo h
    e4_isic_h.png                   Dice, số cụm và thời gian theo h trên các ảnh tune
    e4_isic_examples.png            2 ảnh (trung vị, tệ nhất), mỗi ảnh 4 mức h: ảnh gốc, cụm, mask
    best_h_3d_lab.json              h tốt nhất cho 3D Lab (A dùng ở E5)
    e4_isic_metadata.json           cấu hình, phiên bản thư viện, SHA256 file chia tập
"""

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

# Thêm thư mục gốc của project vào đường dẫn tìm kiếm module
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scipy
import sklearn

from common import DATASETS, ROOT, load_synthetic, synthetic_path, timeit
from meanshift.bandwidth import estimate_bandwidth
from meanshift.core import MeanShiftCustom


# 15 giá trị alpha từ 0.1 đến 3 theo thang logarit
ALPHAS = np.geomspace(0.1, 3.0, 15)
DEFAULT_SEEDS = list(range(10))
# Giống E2: h0 = estimate_bandwidth(X, quantile=0.3)
QUANTILE = 0.3
# Khoảng alpha "an toàn": ARI trung bình >= SAFE_RATIO × ARI tốt nhất của bộ đó
SAFE_RATIO = 0.9

# ----- Phần 2: ISIC tune -----
ISIC_HS = [2, 4, 6, 8, 10, 12, 16, 20]  # bandwidth theo đơn vị Lab
AUTO = "auto"                           # h = estimate_bandwidth của từng ảnh
ISIC_N_SAMPLES = 500   # estimate_bandwidth chỉ lấy mẫu 500 pixel (cả ảnh ~12.000 pixel sẽ rất chậm và tốn RAM)
ISIC_REPEAT = 3        # số lần đo thời gian cho mỗi (ảnh, h), lấy trung vị
QUICK_IMAGES = 10      # --quick: chỉ chạy 10 ảnh đầu của tập tune
N_EXAMPLE_LEVELS = 4   # số mức h trong hình ví dụ


def load_scaled(dataset, seed):
    """Đọc dữ liệu synthetic và chuẩn hóa giống thí nghiệm E2."""
    X, y = load_synthetic(dataset, seed)
    X = StandardScaler().fit_transform(X)
    return X, y


def run_one(dataset, seed, alpha):
    """Chạy một cấu hình bandwidth và trả về kết quả."""
    X, y_true = load_scaled(dataset, seed)

    # Bandwidth mặc định được ước lượng từ dữ liệu
    h0 = estimate_bandwidth(X, quantile=QUANTILE)

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
            neighbor_search="kdtree",
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
        "n_true_clusters": int(len(np.unique(y_true))),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "time_seconds": float(np.median(elapsed_times)),
    }


def summarize_results(runs):
    """Tổng hợp trung bình và độ lệch chuẩn theo dataset, alpha."""
    summary = (
        runs.groupby(["dataset", "alpha"], as_index=False)
        .agg(
            n_true_clusters=("n_true_clusters", "first"),
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


def alpha_table(summary):
    """Theo từng bộ: alpha tốt nhất, ARI tại alpha = 1 và khoảng alpha an toàn.

    Alpha tốt nhất được chọn bằng nhãn thật trên chính các seed đã chạy,
    nên ARI tốt nhất là cận trên (oracle), không phải kết quả dùng được khi không có nhãn.
    """
    rows = []
    for dataset in DATASETS:
        data = summary[summary["dataset"] == dataset].sort_values("alpha")
        best = data.loc[data["ari_mean"].idxmax()]
        default = data.loc[(data["alpha"] - 1.0).abs().idxmin()]
        safe = data[data["ari_mean"] >= SAFE_RATIO * best["ari_mean"]]
        rows.append({
            "dataset": dataset,
            "n_true_clusters": int(best["n_true_clusters"]),
            "best_alpha": float(best["alpha"]),
            "best_ari": float(best["ari_mean"]),
            "best_n_clusters": float(best["n_clusters_mean"]),
            "default_ari": float(default["ari_mean"]),
            "default_n_clusters": float(default["n_clusters_mean"]),
            "ari_gap": float(best["ari_mean"] - default["ari_mean"]),
            "safe_alpha_min": float(safe["alpha"].min()),
            "safe_alpha_max": float(safe["alpha"].max()),
            "safe_n_alphas": int(len(safe)),
            "default_in_safe": bool((safe["alpha"] == default["alpha"]).any()),
        })
    return pd.DataFrame(rows)


def plot_results(summary, output_path):
    """Vẽ số cụm và ARI theo alpha, trung bình ± độ lệch chuẩn."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for i, dataset in enumerate(DATASETS):
        color = f"C{i}"
        data = summary[summary["dataset"] == dataset].sort_values("alpha")

        x = data["alpha"].to_numpy()

        # Số cụm trung bình ± độ lệch chuẩn
        y_clusters = data["n_clusters_mean"].to_numpy()
        err_clusters = data["n_clusters_std"].to_numpy()

        axes[0].plot(x, y_clusters, marker="o", color=color, label=dataset)
        # Trục y dạng log nên chặn dưới ở 1 cụm
        axes[0].fill_between(
            x,
            np.maximum(y_clusters - err_clusters, 1),
            y_clusters + err_clusters,
            color=color,
            alpha=0.15,
        )
        # Số cụm thật của bộ dữ liệu
        axes[0].axhline(
            data["n_true_clusters"].iloc[0],
            color=color,
            linestyle=":",
            linewidth=1,
        )

        # ARI trung bình ± độ lệch chuẩn
        y_ari = data["ari_mean"].to_numpy()
        err_ari = data["ari_std"].to_numpy()

        axes[1].plot(x, y_ari, marker="o", color=color, label=dataset)
        # ARI không vượt quá 1; phía dưới chặn ở 0 cho dễ đọc
        axes[1].fill_between(
            x,
            np.clip(y_ari - err_ari, 0, 1),
            np.clip(y_ari + err_ari, 0, 1),
            color=color,
            alpha=0.15,
        )

    for ax in axes:
        ax.set_xscale("log")
        ax.axvline(
            x=1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="alpha = 1 (h mặc định)",
        )
        ax.set_xlabel("Alpha (h = alpha × h0)")
        ax.grid(True, alpha=0.3)

    axes[0].set_yscale("log")
    axes[0].set_title("Độ nhạy số lượng cụm theo bandwidth\n(đường chấm: số cụm thật)")
    axes[0].set_ylabel("Số cụm (thang log)")
    axes[1].set_title("Độ nhạy ARI theo bandwidth")
    axes[1].set_ylabel("ARI")
    axes[1].set_ylim(-0.02, 1.02)

    # Tránh lặp nhãn alpha=1 trong chú giải
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Phần 2: ISIC tune (3D Lab, bin_seeding=True)
# =====================================================================
def isic_settings(h_auto):
    """Các mức h cần chạy cho một ảnh: (nhãn, h). Nhãn là "2", "4", ... hoặc "auto"."""
    return [(str(h), float(h)) for h in ISIC_HS] + [(AUTO, float(h_auto))]


def fit_labels(X, h):
    """Mean-Shift 3D Lab với bandwidth h (đơn vị Lab), bật bin_seeding."""
    return MeanShiftCustom(bandwidth=h, bin_seeding=True, neighbor_search="kdtree").fit(X)


def run_isic_image(image_id, repeat=ISIC_REPEAT):
    """Chạy mọi mức h trên một ảnh tune. Trả về list dict (mỗi mức h một dòng)."""
    from isic_utils import dice, features, iou, labels_to_mask, load_image

    rgb, gt = load_image(image_id)
    X = features(rgb, "lab3")

    h_auto = estimate_bandwidth(
        X, quantile=QUANTILE, n_samples=ISIC_N_SAMPLES, random_state=0
    )
    if not np.isfinite(h_auto) or h_auto <= 0:
        raise ValueError(f"Bandwidth tự động không hợp lệ: ảnh={image_id}, h={h_auto}")

    rows = []
    for setting, h in isic_settings(h_auto):
        model, seconds = timeit(lambda h=h: fit_labels(X, h), repeat=repeat)
        labels = np.asarray(model.labels_)
        pred = labels_to_mask(labels, rgb)
        rows.append({
            "image_id": image_id,
            "h_setting": setting,
            "h": h,
            "n_clusters": int(len(np.unique(labels[labels != -1]))),
            "dice": dice(pred, gt),
            "iou": iou(pred, gt),
            "empty_mask": bool(not pred.any()),
            "time_seconds": seconds,
        })
    return rows


def summarize_isic(runs):
    """Trung bình ± độ lệch chuẩn qua các ảnh theo mức h (các h cố định theo thứ tự tăng dần, "auto" ở cuối)."""
    summary = (
        runs.groupby("h_setting", as_index=False)
        .agg(
            n_images=("image_id", "nunique"),
            h_mean=("h", "mean"),
            dice_mean=("dice", "mean"),
            dice_std=("dice", "std"),
            dice_median=("dice", "median"),
            iou_mean=("iou", "mean"),
            iou_std=("iou", "std"),
            n_clusters_mean=("n_clusters", "mean"),
            n_clusters_median=("n_clusters", "median"),
            empty_mask_rate=("empty_mask", "mean"),
            time_mean=("time_seconds", "mean"),
            time_std=("time_seconds", "std"),
        )
    )
    std_columns = [col for col in summary.columns if col.endswith("_std")]
    summary[std_columns] = summary[std_columns].fillna(0)

    # "auto" có h_mean là trung bình các h của từng ảnh -> xếp cuối
    summary["_order"] = summary["h_setting"].map(lambda s: np.inf if s == AUTO else float(s))
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def best_fixed_h(summary):
    """Hàng của h cố định (không tính "auto") có Dice trung bình cao nhất."""
    fixed = summary[summary["h_setting"] != AUTO]
    return fixed.loc[fixed["dice_mean"].idxmax()]


def plot_isic_h(summary, output_path, n_images):
    """3 hình con: Dice/IoU, số cụm, thời gian theo h. "auto" vẽ bằng đường nằm ngang."""
    fixed = summary[summary["h_setting"] != AUTO]
    auto = summary[summary["h_setting"] == AUTO].iloc[0]
    best = best_fixed_h(summary)
    x = fixed["h_mean"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Dice, IoU
    ax = axes[0]
    ax.plot(x, fixed["dice_mean"], marker="o", color="C0", label="Dice (trung bình)")
    ax.fill_between(
        x,
        np.clip(fixed["dice_mean"] - fixed["dice_std"], 0, 1),
        np.clip(fixed["dice_mean"] + fixed["dice_std"], 0, 1),
        color="C0", alpha=0.15, label="Dice ± 1 độ lệch chuẩn",
    )
    ax.plot(x, fixed["iou_mean"], marker="s", color="C1", label="IoU (trung bình)")
    ax.axhline(auto["dice_mean"], color="C0", linestyle="--", linewidth=1.2,
               label=f"Dice, h = estimate_bandwidth ({auto['dice_mean']:.3f})")
    ax.axvline(best["h_mean"], color="gray", linestyle=":", linewidth=1.2)
    ax.annotate(f"h tốt nhất = {best['h_mean']:g}\nDice = {best['dice_mean']:.3f}",
                xy=(best["h_mean"], best["dice_mean"]),
                xytext=(10, -45), textcoords="offset points", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "gray"})
    ax.set_title(f"Độ nhạy Dice/IoU theo h ({n_images} ảnh tune, 3D Lab)")
    ax.set_xlabel("Bandwidth h (đơn vị Lab)")
    ax.set_ylabel("Dice / IoU")
    ax.set_ylim(0, 1)

    # Số cụm
    ax = axes[1]
    ax.plot(x, fixed["n_clusters_mean"], marker="o", color="C2", label="Trung bình")
    ax.plot(x, fixed["n_clusters_median"], marker="^", color="C3", linestyle="-", label="Trung vị")
    ax.axhline(auto["n_clusters_median"], color="C3", linestyle="--", linewidth=1.2,
               label=f"Trung vị, h = estimate_bandwidth ({auto['n_clusters_median']:g})")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())  # 2, 5, 10 thay vì 2×10⁰
    ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_title("Số cụm theo h")
    ax.set_xlabel("Bandwidth h (đơn vị Lab)")
    ax.set_ylabel("Số cụm mỗi ảnh (thang log)")

    # Thời gian
    ax = axes[2]
    ax.plot(x, fixed["time_mean"], marker="o", color="C4", label="Thời gian trung bình / ảnh")
    ax.fill_between(
        x,
        np.maximum(fixed["time_mean"] - fixed["time_std"], 0),
        fixed["time_mean"] + fixed["time_std"],
        color="C4", alpha=0.15, label="± 1 độ lệch chuẩn",
    )
    ax.axhline(auto["time_mean"], color="C4", linestyle="--", linewidth=1.2,
               label=f"h = estimate_bandwidth ({auto['time_mean']:.3f} s)")
    ax.set_title("Thời gian chạy theo h")
    ax.set_xlabel("Bandwidth h (đơn vị Lab)")
    ax.set_ylabel("Thời gian (giây)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def example_levels(best_h):
    """N_EXAMPLE_LEVELS mức h trải đều trên lưới ISIC_HS, bảo đảm có mức h tốt nhất."""
    idx = np.linspace(0, len(ISIC_HS) - 1, N_EXAMPLE_LEVELS).round().astype(int)
    levels = [ISIC_HS[i] for i in idx]
    levels[int(np.argmin([abs(v - best_h) for v in levels]))] = best_h
    return sorted(set(levels))


def plot_isic_examples(runs, best_h, output_path):
    """Hình ví dụ: ảnh có Dice trung vị và ảnh có Dice thấp nhất (tại h tốt nhất).

    Mỗi ảnh 2 hàng: hàng trên là ảnh gốc + bản đồ cụm ở từng mức h,
    hàng dưới là mask bác sĩ + mask dự đoán (sau labels_to_mask) kèm Dice.
    """
    from isic_utils import dice, features, labels_to_mask, load_image

    at_best = runs[runs["h_setting"] == f"{best_h:g}"].sort_values(["dice", "image_id"])
    picks = [("trung vị", at_best.iloc[len(at_best) // 2]["image_id"])]
    if len(at_best) > 1:
        picks.append(("tệ nhất", at_best.iloc[0]["image_id"]))

    levels = example_levels(best_h)
    fig, axes = plt.subplots(2 * len(picks), 1 + len(levels),
                             figsize=(3 * (1 + len(levels)), 6 * len(picks)), squeeze=False)

    for p, (tag, image_id) in enumerate(picks):
        rgb, gt = load_image(image_id)
        X = features(rgb, "lab3")
        top, bottom = axes[2 * p], axes[2 * p + 1]
        top[0].imshow(rgb)
        top[0].set_title(f"{image_id}\n(ảnh {tag}, Dice tại h = {best_h:g})", fontsize=9)
        bottom[0].imshow(gt, cmap="gray")
        bottom[0].set_title("Mask bác sĩ", fontsize=9)
        for j, h in enumerate(levels, start=1):
            labels = np.asarray(fit_labels(X, float(h)).labels_)
            pred = labels_to_mask(labels, rgb)
            n_clusters = len(np.unique(labels[labels != -1]))
            top[j].imshow(labels.reshape(rgb.shape[:2]), cmap="tab20", interpolation="nearest")
            top[j].set_title(f"h = {h:g}: {n_clusters} cụm", fontsize=9)
            bottom[j].imshow(pred, cmap="gray")
            bottom[j].set_title(f"Mask dự đoán · Dice {dice(pred, gt):.2f}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Mean-Shift 3D Lab: ảnh hưởng của bandwidth h đến cụm và mask", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_isic(quick=False):
    """Phần 2: quét h trên ảnh tune, lưu bảng, hình và best_h_3d_lab.json."""
    from isic_utils import (BORDER_SHARE, BORDER_WIDTH, MAX_SIDE, MIN_AREA,
                            SPLITS, load_split)

    image_ids = load_split("tune")
    if quick:
        image_ids = image_ids[:QUICK_IMAGES]
        output_dir = Path(tempfile.gettempdir()) / "e4_isic_quick"
        print(f"[--quick] chỉ chạy {len(image_ids)} ảnh; kết quả ghi vào {output_dir}, không đụng results/e4/")
    else:
        output_dir = ROOT / "results" / "e4"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for i, image_id in enumerate(image_ids, start=1):
        print(f"[{i}/{len(image_ids)}] ảnh={image_id}")
        all_rows.extend(run_isic_image(image_id))

    runs = pd.DataFrame(all_rows)
    summary = summarize_isic(runs)
    best = best_fixed_h(summary)
    auto = summary[summary["h_setting"] == AUTO].iloc[0]
    best_h = float(best["h_setting"])

    runs_path = output_dir / "e4_isic_runs.csv"
    summary_path = output_dir / "e4_isic_summary.csv"
    plot_path = output_dir / "e4_isic_h.png"
    examples_path = output_dir / "e4_isic_examples.png"
    best_path = output_dir / "best_h_3d_lab.json"
    metadata_path = output_dir / "e4_isic_metadata.json"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_isic_h(summary, plot_path, len(image_ids))
    plot_isic_examples(runs, best_h, examples_path)

    best_record = {
        "feature": "lab3",
        "unit": "Lab",
        "best_h": best_h,
        "dice_mean": float(best["dice_mean"]),
        "dice_std": float(best["dice_std"]),
        "iou_mean": float(best["iou_mean"]),
        "n_clusters_median": float(best["n_clusters_median"]),
        "n_images": int(best["n_images"]),
        "grid": {r["h_setting"]: round(float(r["dice_mean"]), 4) for _, r in summary.iterrows()},
        "auto_bandwidth": {
            "dice_mean": float(auto["dice_mean"]),
            "mean_h": float(auto["h_mean"]),
            "quantile": QUANTILE,
            "n_samples": ISIC_N_SAMPLES,
        },
        "meanshift": "MeanShiftCustom(flat, kdtree, bin_seeding=True)",
        "note": ("h chọn theo Dice trung bình cao nhất trên tập tune; "
                 "kết quả cuối cùng phải báo cáo trên tập test (E5)."
                 + (" CHẠY --quick: chỉ một phần ảnh tune, không dùng làm kết quả chính." if quick else "")),
    }
    best_path.write_text(json.dumps(best_record, indent=2, ensure_ascii=False), encoding="utf-8")

    split_file = SPLITS / "isic_tune.txt"
    metadata = {
        "experiment": "E4 – độ nhạy theo bandwidth (phần 2: ISIC tune)",
        "quick": bool(quick),
        "n_images": len(image_ids),
        "h_grid_lab": ISIC_HS,
        "h_auto": f"estimate_bandwidth(X, quantile={QUANTILE}, n_samples={ISIC_N_SAMPLES}, random_state=0) trên từng ảnh",
        "features": "3D Lab (isic_utils.features(rgb, 'lab3')), ảnh thu nhỏ cạnh dài về "
                    f"{MAX_SIDE} px",
        "meanshift": "MeanShiftCustom(flat, kdtree, bin_seeding=True)",
        "mask_rule": (f"isic_utils.labels_to_mask (border_width={BORDER_WIDTH}, "
                      f"border_share={BORDER_SHARE}, min_area={MIN_AREA})"),
        "timing": f"time.perf_counter quanh fit, {ISIC_REPEAT} lần, lấy trung vị",
        "best_h": "h cố định có Dice trung bình cao nhất trên các ảnh đã chạy",
        "input_sha256": {
            f"data/splits/{split_file.name}": hashlib.sha256(split_file.read_bytes()).hexdigest()
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n===== TÓM TẮT E4 – ISIC TUNE =====")
    print(f"Số ảnh: {len(image_ids)} · số lần chạy: {len(runs)}")
    print(f"Kết quả: {output_dir}")
    print(summary[[
        "h_setting", "h_mean", "dice_mean", "dice_std", "iou_mean",
        "n_clusters_median", "empty_mask_rate", "time_mean",
    ]].round(3).to_string(index=False))
    print(f"\nh tốt nhất (Lab): {best_h:g} · Dice trung bình {best['dice_mean']:.3f} "
          f"· h = estimate_bandwidth: Dice {auto['dice_mean']:.3f}")
    print(f"=> Gửi chị A dùng cho E5: {best_path.name}")


def run_synthetic(seeds):
    """Phần 1: quét alpha trên dữ liệu giả lập."""
    output_dir = ROOT / "results" / "e4"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []

    # BỔ SUNG: giữ 15 alpha ban đầu và thêm alpha = 1.0
    alphas_to_run = sorted(set(ALPHAS.tolist() + [1.0]))

    # Sửa tổng số lần chạy để tính đủ 16 alpha
    total = len(DATASETS) * len(seeds) * len(alphas_to_run)
    current = 0

    for dataset in DATASETS:
        for seed in seeds:
            for alpha in alphas_to_run:
                current += 1
                print(
                    f"[{current}/{total}] "
                    f"dataset={dataset}, seed={seed}, alpha={alpha:.4f}"
                )

                result = run_one(dataset, seed, alpha)
                all_runs.append(result)

    runs = pd.DataFrame(all_runs)
    summary = summarize_results(runs)
    table = alpha_table(summary)

    runs_path = output_dir / "e4_synthetic_runs.csv"
    summary_path = output_dir / "e4_synthetic_summary.csv"
    table_path = output_dir / "e4_synthetic_alpha_summary.csv"
    plot_path = output_dir / "e4_synthetic.png"
    metadata_path = output_dir / "e4_metadata.json"

    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    table.to_csv(table_path, index=False)
    plot_results(summary, plot_path)

    used = [synthetic_path(d, s) for d in DATASETS for s in seeds]
    metadata = {
        "experiment": "E4 – độ nhạy theo bandwidth (phần 1: dữ liệu giả lập)",
        "datasets": DATASETS,
        "seeds": seeds,
        "alphas": alphas_to_run,
        "bandwidth": f"h = alpha × h0, h0 = meanshift.estimate_bandwidth(X, quantile={QUANTILE})",
        "meanshift": "MeanShiftCustom(flat, kdtree, bin_seeding=True)",
        "standardization": "StandardScaler riêng cho từng (bộ, seed)",
        "timing": "time.perf_counter quanh fit, 3 lần, lấy trung vị",
        "best_alpha": "alpha có ARI trung bình cao nhất trên chính các seed đã chạy (cận trên, dùng nhãn thật)",
        "safe_range": f"các alpha có ARI trung bình >= {SAFE_RATIO} × ARI tốt nhất của bộ đó",
        "input_sha256": {
            str(p.relative_to(ROOT).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in used
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n===== TÓM TẮT E4 =====")
    print(f"Số lần chạy: {len(runs)}")
    print(f"Kết quả chi tiết: {runs_path}")
    print(f"Kết quả tổng hợp: {summary_path}")
    print(f"Bảng alpha theo từng bộ: {table_path}")
    print(f"Biểu đồ: {plot_path}")
    print()
    print(table[[
        "dataset", "best_alpha", "best_ari", "default_ari",
        "safe_alpha_min", "safe_alpha_max", "default_in_safe",
    ]].round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="E4: Khảo sát độ nhạy của Mean Shift theo bandwidth."
    )
    parser.add_argument(
        "--part",
        choices=["synthetic", "isic", "all"],
        default="all",
        help="Phần cần chạy: synthetic (dữ liệu giả), isic (ảnh da), all (cả hai, mặc định).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Phần synthetic: số seed chạy cho mỗi dataset (mặc định: 10).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Phần isic: chỉ chạy {QUICK_IMAGES} ảnh đầu, ghi kết quả ra thư mục tạm.",
    )
    args = parser.parse_args()

    if args.seeds < 1 or args.seeds > 10:
        parser.error("--seeds phải nằm trong khoảng từ 1 đến 10.")

    if args.part in ("synthetic", "all"):
        run_synthetic(DEFAULT_SEEDS[:args.seeds])
    if args.part in ("isic", "all"):
        run_isic(quick=args.quick)


if __name__ == "__main__":
    main()