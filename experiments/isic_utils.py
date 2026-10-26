"""Công cụ dùng chung cho các thực nghiệm ISIC 2018 (E4 phần 2, E5, E6).

Người phụ trách: A  ·  Branch: isic-utils

Mọi phương pháp (Mean-Shift, Otsu, K-Means, GMM) đi qua CÙNG một quy trình:

    load_image (thu nhỏ cạnh dài về 128 px)
      -> features(rgb, mode)              ma trận điểm (H*W, d)
      -> thuật toán phân cụm              nhãn cụm cho từng pixel
      -> labels_to_mask(labels, rgb)      chọn cụm tổn thương bằng một quy tắc chung
      -> dice / iou so với mask thật

Nhờ vậy, khác biệt về Dice chỉ đến từ thuật toán, không đến từ cách xử lý ảnh.

Dữ liệu: chạy trước `python3 data/download_isic.py`.
    tune = 100 ảnh Validation (chọn tham số), test = 200 ảnh lấy từ Test (chấm điểm cuối).

Kiểm tra nhanh trên vài ảnh tune (lưu results/e5/check_utils.png):
    python3 experiments/isic_utils.py [--n 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab

ROOT = Path(__file__).resolve().parents[1]
ISIC = ROOT / "data" / "isic2018"
SPLITS = ROOT / "data" / "splits"

# Ảnh và mask của tập tune (Validation) và tập test (Test)
IMAGE_DIRS = [ISIC / "ISIC2018_Task1-2_Validation_Input", ISIC / "ISIC2018_Task1-2_Test_Input"]
MASK_DIRS = [ISIC / "ISIC2018_Task1_Validation_GroundTruth", ISIC / "ISIC2018_Task1_Test_GroundTruth"]

MAX_SIDE = 128
MODES = ("rgb3", "lab3", "lab5")
# labels_to_mask: viền ảnh dày BORDER_WIDTH × cạnh ngắn. Một cụm bị bỏ nếu
#   - phủ >= BORDER_SHARE dải viền (da lành bao quanh), hoặc
#   - >= BORDER_SHARE số pixel của chính nó nằm trên viền (góc tối ống kính, thước đo), hoặc
#   - nhỏ hơn MIN_AREA diện tích ảnh (đốm vụn, lông).
# Hai điều kiện sau được thêm sau khi xem 100 ảnh tune: với Mean-Shift 3D Lab h=8,
# Dice trung bình tăng 0.472 -> 0.567; K-Means k=2 không đổi (0.718).
BORDER_WIDTH = 0.05
BORDER_SHARE = 0.5
MIN_AREA = 0.01


def load_split(name):
    """Danh sách mã ảnh của tập "tune" hoặc "test" (đọc data/splits/isic_<name>.txt)."""
    if name not in ("tune", "test"):
        raise ValueError(f"name phải là 'tune' hoặc 'test', nhận {name!r}")
    path = SPLITS / f"isic_{name}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy trước: python3 data/download_isic.py")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _find(image_id, folders, suffix):
    for folder in folders:
        path = folder / f"{image_id}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Không tìm thấy {image_id}{suffix}. Chạy trước: python3 data/download_isic.py")


def load_image(image_id, max_side=MAX_SIDE):
    """Đọc ảnh và mask thật, thu nhỏ để cạnh dài = max_side (giữ tỷ lệ).

    Trả về (rgb: ndarray (H, W, 3) uint8, mask: ndarray (H, W) bool).
    Ảnh thu nhỏ bằng LANCZOS; mask thu nhỏ bằng NEAREST để giữ giá trị 0/1.
    """
    image = Image.open(_find(image_id, IMAGE_DIRS, ".jpg")).convert("RGB")
    mask = Image.open(_find(image_id, MASK_DIRS, "_segmentation.png")).convert("L")
    if image.size != mask.size:
        raise ValueError(f"{image_id}: ảnh {image.size} và mask {mask.size} khác kích thước")

    scale = max_side / max(image.size)
    if scale < 1:
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8), np.asarray(mask) > 127


def to_lab(rgb):
    """RGB uint8 (H, W, 3) -> Lab float64 (H, W, 3): L trong [0, 100], a và b khoảng [-128, 127]."""
    return rgb2lab(np.asarray(rgb, dtype=np.uint8))


def features(rgb, mode, hs=None, hr=None):
    """Ma trận điểm (H*W, d) cho thuật toán phân cụm, theo thứ tự pixel hàng-trước.

    - "rgb3": (R, G, B), đơn vị 0–255.
    - "lab3": (L, a, b), đơn vị Lab (không chuẩn hoá, để h có nghĩa theo đơn vị màu).
    - "lab5": (x/hs, y/hs, L/hr, a/hr, b/hr); khi đó Mean-Shift dùng bandwidth = 1.
    """
    if mode not in MODES:
        raise ValueError(f"mode phải là một trong {MODES}, nhận {mode!r}")
    h, w = rgb.shape[:2]
    if mode == "rgb3":
        return rgb.reshape(-1, 3).astype(np.float64)
    lab = to_lab(rgb).reshape(-1, 3)
    if mode == "lab3":
        return lab
    if hs is None or hr is None or hs <= 0 or hr <= 0:
        raise ValueError("mode 'lab5' cần hs > 0 và hr > 0")
    yy, xx = np.mgrid[0:h, 0:w]
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    return np.hstack([xy / hs, lab / hr])


def border_ring(shape, width=BORDER_WIDTH):
    """Mask bool (H, W) của dải viền ảnh, dày max(1, width × cạnh ngắn) pixel."""
    h, w = shape
    k = max(1, int(round(width * min(h, w))))
    ring = np.zeros((h, w), dtype=bool)
    ring[:k, :] = ring[-k:, :] = True
    ring[:, :k] = ring[:, -k:] = True
    return ring


def labels_to_mask(labels, rgb, border_width=BORDER_WIDTH, border_share=BORDER_SHARE,
                   min_area=MIN_AREA):
    """Chọn cụm tổn thương từ nhãn cụm — quy tắc chung cho MỌI phương pháp.

    1. Bỏ các cụm: phủ >= border_share dải viền ảnh (da lành), hoặc có >= border_share
       số pixel nằm trên viền (góc tối ống kính, thước đo), hoặc nhỏ hơn min_area diện tích ảnh.
    2. Trong các cụm còn lại, chọn cụm tối nhất (L trung bình nhỏ nhất).
    3. Lấp lỗ (binary_fill_holes), rồi giữ vùng liền lớn nhất (ndimage.label, 4-láng giềng).

    labels: ndarray (H*W,) hoặc (H, W); nhãn -1 (nhiễu) không bao giờ được chọn.
    Không còn cụm ứng viên nào thì trả về mask rỗng.
    """
    h, w = rgb.shape[:2]
    labels = np.asarray(labels).reshape(h, w)
    lightness = to_lab(rgb)[..., 0]
    ring = border_ring((h, w), border_width)
    n_ring = ring.sum()

    best, best_l = None, np.inf
    for cluster in np.unique(labels):
        if cluster == -1:
            continue
        region = labels == cluster
        size = region.sum()
        on_ring = (region & ring).sum()
        if on_ring >= border_share * n_ring or on_ring >= border_share * size or size < min_area * h * w:
            continue
        mean_l = lightness[region].mean()
        if mean_l < best_l:
            best, best_l = cluster, mean_l
    if best is None:
        return np.zeros((h, w), dtype=bool)

    mask = ndimage.binary_fill_holes(labels == best)
    components, n = ndimage.label(mask)
    if n > 1:
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        mask = components == sizes.argmax()
    return mask


def dice(pred, gt):
    """Dice = 2|P ∩ G| / (|P| + |G|); cả hai rỗng thì bằng 1."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else float(2 * (pred & gt).sum() / total)


def iou(pred, gt):
    """IoU = |P ∩ G| / |P ∪ G|; cả hai rỗng thì bằng 1."""
    pred, gt = np.asarray(pred, dtype=bool), np.asarray(gt, dtype=bool)
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float((pred & gt).sum() / union)


def _check(n_images):
    """Chạy thử K-Means k=2 và Mean-Shift 3D Lab (h=8) trên vài ảnh tune, lưu hình kiểm tra."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    sys.path.insert(0, str(ROOT))
    from meanshift import MeanShiftCustom

    ids = load_split("tune")[:n_images]
    fig, axes = plt.subplots(len(ids), 4, figsize=(12, 3 * len(ids)), squeeze=False)
    for row, image_id in zip(axes, ids):
        rgb, gt = load_image(image_id)
        X = features(rgb, "lab3")
        km = labels_to_mask(KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X), rgb)
        ms_model = MeanShiftCustom(bandwidth=8, bin_seeding=True).fit(X)
        ms = labels_to_mask(ms_model.labels_, rgb)
        n_ms = len(ms_model.cluster_centers_)
        print(f"{image_id} {rgb.shape[1]}x{rgb.shape[0]}  K-Means Dice={dice(km, gt):.3f}  "
              f"Mean-Shift h=8 ({n_ms} cụm) Dice={dice(ms, gt):.3f}")
        panels = [(rgb, image_id), (gt, "Mask thật"),
                  (km, f"K-Means k=2 · Dice {dice(km, gt):.2f}"),
                  (ms, f"Mean-Shift h=8 ({n_ms} cụm) · Dice {dice(ms, gt):.2f}")]
        for ax, (img, title) in zip(row, panels):
            ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle("Kiểm tra isic_utils: đặc trưng 3D Lab, quy tắc labels_to_mask chung")
    fig.tight_layout()
    out = ROOT / "results" / "e5"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "check_utils.png", dpi=150)
    plt.close(fig)
    print(f"Đã lưu {out / 'check_utils.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra nhanh isic_utils trên vài ảnh tune")
    parser.add_argument("--n", type=int, default=4, help="Số ảnh tune để kiểm tra (mặc định 4)")
    _check(parser.parse_args().n)
