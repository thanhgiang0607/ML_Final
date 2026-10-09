# Đồ án ML – Mean-Shift

**Đề tài:** Thuật toán Mean-Shift: hiện thực OOP, kiểm chứng với sklearn và ứng dụng phân đoạn tổn thương da (ảnh dermoscopy).

**Ý tưởng:** Tự cài Mean-Shift từ đầu, chạy đối chiếu với `sklearn` xem giống hay khác, đo tốc độ và độ nhạy theo bandwidth `h`, rồi đem đi tách vùng tổn thương da và chấm điểm bằng Dice/IoU so với mask của chuyên gia.

Chi tiết đầy đủ: `DeCuong_DoAn_MeanShift.docx` · Bản tóm tắt cho nhóm: `Proposal_MeanShift.docx`

---

## Cấu trúc thư mục

```
ML_Final/
├── meanshift/                 # Package tự cài (chỉ dùng NumPy/SciPy)
│   ├── core.py                #   MeanShiftCustom: fit / predict / fit_predict
│   ├── base.py                #   Lớp cơ sở: get_params, set_params, kiểm tra input
│   ├── kernels.py             #   FlatKernel, GaussianKernel
│   ├── neighbors.py           #   BruteForceSearch, KDTreeSearch
│   ├── bandwidth.py           #   estimate_bandwidth
│   └── seeding.py             #   get_bin_seeds (bin seeding)
├── tests/                     # Unit test (pytest)
├── experiments/
│   └── e1_compare_sklearn.py  # E1: đối chiếu với sklearn
├── results/e1/                # Kết quả E1 (csv + hình)
├── data/
│   ├── make_synthetic.py      # Sinh dữ liệu giả
│   ├── download_isic.py       # Tải ISIC 2018 Task 1 + tạo file chia tập
│   ├── splits/                # ID ảnh tập tune (100) / test (200) — commit để cả nhóm dùng chung
│   ├── synthetic/             # 5 bộ × 10 seed (sinh ra, không cần commit)
│   └── isic2018/              # Ảnh da ISIC 2018 Task 1 (tải về, không commit)
└── requirements.txt
```



## Chạy

```bash
pip install -r requirements.txt
python3 data/make_synthetic.py             # sinh dữ liệu giả
python3 data/download_isic.py              # tải ảnh da ISIC (~2,5 GB; thêm --no-test-images để chỉ tải ~240 MB)
python3 -m pytest tests                    # chạy unit test
python3 experiments/e1_compare_sklearn.py  # E1 (thêm --seeds 2 để chạy nhanh)
```



## Dùng thử

```python
from meanshift import MeanShiftCustom, estimate_bandwidth

h = estimate_bandwidth(X)
ms = MeanShiftCustom(bandwidth=h, kernel="flat", neighbor_search="kdtree").fit(X)
ms.cluster_centers_, ms.labels_
```

---



## Các bước làm

1. **Lý thuyết** – Viết phần KDE, vector Mean-Shift, hội tụ, so sánh với K-Means/DBSCAN/GMM.
2. **Dữ liệu** – Sinh dữ liệu giả (blobs, moons, circles). Tải ISIC 2018 Task 1 hoặc HAM10000 kèm mask, chia 50 ảnh để tune, 200 ảnh để test.
3. **Code** `MeanShiftCustom` – Có `fit / predict / fit_predict`, flat + Gaussian kernel, bản brute-force + KD-Tree, bin seeding. Viết unit test.
4. **E1 – Đối chiếu sklearn** – Chạy cả hai trên cùng dữ liệu, cùng `h`; so số cụm, tâm cụm, nhãn (ARI) để xem giống hay khác, khác thì giải thích vì sao.
5. **E2 – So thuật toán** – Mean-Shift vs K-Means, DBSCAN, GMM trên dữ liệu giả (ARI, NMI).
6. **E3 – Tốc độ** – Đo thời gian và bộ nhớ khi tăng `n` và `h`: brute vs KD-Tree vs sklearn.
7. **E4 – Độ nhạy** `h` – Quét `h`, xem số cụm và chất lượng thay đổi thế nào.
8. **E5 – Phân đoạn da** – Chuyển ảnh sang Lab, thử 3D và 5D (thêm vị trí pixel). So với Otsu, K-Means, GMM bằng Dice/IoU trên tập test.
9. **E6 – Ablation** – So flat vs Gaussian, bin seeding bật/tắt.
10. **Viết báo cáo + slide** – Theo outline trong file docx, trả lời từng câu hỏi nghiên cứu bằng số liệu.

