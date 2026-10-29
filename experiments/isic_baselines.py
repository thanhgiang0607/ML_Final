"""Baseline phân đoạn cho E5: Otsu, K-Means, GMM.

Người phụ trách: B  ·  Branch: isic-baselines

Mỗi hàm trả về nhãn cụm cho từng pixel, dạng ndarray (H*W,) theo thứ tự hàng-trước, giống
MeanShiftCustom.labels_, để đưa chung vào isic_utils.labels_to_mask:

    rgb, gt = load_image(image_id)
    X = features(rgb, "lab3")
    pred = labels_to_mask(kmeans_labels(X), rgb)

Kiểm tra nhanh trên vài ảnh tune (in Dice từng phương pháp):
    python3 experiments/isic_baselines.py [--n 10]
"""
import argparse

import numpy as np
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from isic_utils import to_lab


def otsu_labels(rgb):
    """Ngưỡng Otsu trên kênh L (Lab). Nhãn 1 = sáng hơn ngưỡng, 0 = tối hơn hoặc bằng.

    Ảnh có L không đổi thì trả về toàn nhãn 0.
    """
    lightness = to_lab(rgb)[..., 0].ravel()
    if lightness.max() == lightness.min():
        return np.zeros(lightness.size, dtype=int)
    return (lightness > threshold_otsu(lightness)).astype(int)


def kmeans_labels(feat, k=2):
    """K-Means k cụm trên ma trận đặc trưng (H*W, d); random_state=0, n_init=10."""
    return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(feat)


def gmm_labels(feat, k=2):
    """Gaussian Mixture k thành phần (hiệp phương sai đầy đủ) trên (H*W, d); random_state=0."""
    return GaussianMixture(n_components=k, covariance_type="full", random_state=0).fit_predict(feat)


def _check(n_images):
    """In Dice của 3 baseline (đặc trưng 3D Lab) trên vài ảnh tune."""
    from isic_utils import dice, features, labels_to_mask, load_image, load_split

    methods = {
        "Otsu": lambda rgb, X: otsu_labels(rgb),
        "K-Means": lambda rgb, X: kmeans_labels(X),
        "GMM": lambda rgb, X: gmm_labels(X),
    }
    scores = {name: [] for name in methods}
    for image_id in load_split("tune")[:n_images]:
        rgb, gt = load_image(image_id)
        X = features(rgb, "lab3")
        row = []
        for name, fn in methods.items():
            scores[name].append(dice(labels_to_mask(fn(rgb, X), rgb), gt))
            row.append(f"{name} {scores[name][-1]:.3f}")
        print(f"{image_id}  " + "  ".join(row))
    print("Dice trung bình: " + "  ".join(f"{name} {np.mean(v):.3f}" for name, v in scores.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra nhanh baseline trên vài ảnh tune")
    parser.add_argument("--n", type=int, default=10, help="Số ảnh tune (mặc định 10)")
    _check(parser.parse_args().n)
