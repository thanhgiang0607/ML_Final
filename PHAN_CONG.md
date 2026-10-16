# Phân công nhóm – Đồ án Mean-Shift

> Đọc trước: `Proposal_MeanShift.docx` (đề tài, mục tiêu) · `README.md` (cấu trúc, lệnh chạy)


| Ký hiệu | Thành viên   | Vai trò chính                                           |
| ------- | ------------ | ------------------------------------------------------- |
| A       | *(điền tên)* | Trưởng nhóm · `core.py` · ghép code · case study ảnh da |
| B       | *(điền tên)* | `kernels.py` · Chương 1 lý thuyết                       |
| C       | *(điền tên)* | `neighbors.py` · đo tốc độ                              |
| D       | *(điền tên)* | `bandwidth.py`, `seeding.py` · độ nhạy h                |
| E       | *(điền tên)* | Đối chiếu sklearn (E1) · so sánh thuật toán (E2)        |


---



## Giai đoạn 1 – Code OOP (tuần 1–2)

Các file trong `meanshift/` đã có **khung sẵn (stub)**: tên lớp, tên hàm, tham số và mô tả đầu vào/đầu ra đã chốt. Mỗi người chỉ cần viết phần ruột ở chỗ có `# TODO(<ký hiệu>)`.


| Người | File cần code                                    | File test                                          | Branch                  | Xong khi                                              |
| ----- | ------------------------------------------------ | -------------------------------------------------- | ----------------------- | ----------------------------------------------------- |
| **A** | `meanshift/core.py`                              | `tests/test_core.py`                               | `oop-core`              | 2 blob xa nhau → đúng 2 cụm; `predict` khớp `labels_` |
| **B** | `meanshift/kernels.py`                           | `tests/test_kernels.py`                            | `oop-kernels`           | Trọng số Flat = 1; Gaussian đúng `exp(-u/2)`          |
| **C** | `meanshift/neighbors.py`                         | `tests/test_neighbors.py`                          | `oop-neighbors`         | Brute-force và KD-Tree trả về cùng láng giềng         |
| **D** | `meanshift/bandwidth.py`, `meanshift/seeding.py` | `tests/test_bandwidth.py`, `tests/test_seeding.py` | `oop-bandwidth-seeding` | Ra cùng kết quả với sklearn                           |
| **E** | `experiments/e1_compare_sklearn.py`              | `tests/test_vs_sklearn.py`                         | `e1-compare`            | Bảng + hình đối chiếu sklearn trên 5 bộ × 10 seed     |


`meanshift/base.py` (lớp cơ sở, `check_array`) **đã cài sẵn**, mọi người dùng chung.

**Thứ tự merge:** B, C, D (độc lập, ai xong trước merge trước) → A → E.
Trong lúc chờ: A có thể viết tạm bản flat + brute-force ngay trong `core.py` để chạy thử; E viết sẵn script và chạy khi `core.py` đã merge.

### Giao diện đã chốt — KHÔNG tự ý đổi

```
kernels.py    FlatKernel / GaussianKernel(cutoff=3.0)
                .support(bandwidth) -> float
                .weights(sq_dist, bandwidth) -> ndarray
              make_kernel(kernel, gaussian_cutoff=3.0)

neighbors.py  BruteForceSearch(chunk_size=512) / KDTreeSearch()
                .fit(X) -> self
                .query_radius(points, r) -> list[(idx, sq_dist)]
              make_neighbor_search(name)

bandwidth.py  estimate_bandwidth(X, quantile=0.3, n_samples=None, random_state=0) -> float
seeding.py    get_bin_seeds(X, bin_size, min_bin_freq=1) -> ndarray

core.py       MeanShiftCustom(bandwidth=None, kernel="flat", neighbor_search="kdtree", ...)
                .fit(X) -> self   .predict(X)   .fit_predict(X)
                .cluster_centers_ .labels_ .n_iter_ ...
```

`tests/test_interface.py` tự kiểm tra các chữ ký này. Muốn đổi → bàn với cả nhóm, sửa luôn `test_interface.py` trong cùng PR.

---



## Giai đoạn 2 – Thực nghiệm (tuần 3–5)



### Tổng quan


| Người | Thực nghiệm                                         | Trả lời  | Script                                                                                           | Dữ liệu                                            | Đầu ra (`results/…`) | Branch                          |
| ----- | --------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------- | -------------------- | ------------------------------- |
| **E** | **E1** – Đối chiếu sklearn (làm nốt từ giai đoạn 1) | RQ1      | `experiments/e1_compare_sklearn.py`, `experiments/common.py`                                     | `data/synthetic/` (5 bộ × 10 seed)                 | `results/e1/`        | `e1-compare`                    |
| **E** | **E2** – So với K-Means, DBSCAN, GMM                | RQ2      | `experiments/e2_compare_algorithms.py`                                                           | `data/synthetic/` (5 bộ × 10 seed)                 | `results/e2/`        | `e2-algorithms`                 |
| **C** | **E3** – Tốc độ & bộ nhớ                            | RQ3      | `experiments/e3_runtime.py`                                                                      | Giả lập sinh riêng trong script (n = 500 → 50.000) | `results/e3/`        | `e3-runtime`                    |
| **D** | **E4** – Độ nhạy theo h                             | RQ4      | `experiments/e4_bandwidth.py`                                                                    | `data/synthetic/` + ISIC tune (100 ảnh)            | `results/e4/`        | `e4-bandwidth`                  |
| **A** | **E5** – Phân đoạn tổn thương da                    | RQ5      | `experiments/isic_utils.py`, `experiments/e5_isic_segmentation.py`                               | ISIC tune (100) + ISIC test (200)                  | `results/e5/`        | `isic-utils`, `e5-isic`         |
| **B** | **E6** – Ablation + hình quỹ đạo + baseline cho E5  | RQ3, RQ5 | `experiments/e6_ablation.py`, `experiments/fig_trajectories.py`, `experiments/isic_baselines.py` | `data/synthetic/` + ISIC tune                      | `results/e6/`        | `e6-ablation`, `isic-baselines` |




### Thứ tự và phụ thuộc

**Hầu hết làm song song, độc lập.** `core.py` đã merge nên ai cũng gọi được `MeanShiftCustom`. Chỉ phần ảnh ISIC có phụ thuộc.

**Làm ngay, không chờ ai:**


| Người | Phần làm độc lập                                                |
| ----- | --------------------------------------------------------------- |
| **E** | E1, E2 (toàn bộ)                                                |
| **C** | E3 (toàn bộ, vì tự sinh dữ liệu trong script)                   |
| **D** | E4 phần 1: dữ liệu giả lập                                      |
| **B** | `isic_baselines.py`, E6 bước 2 (giả lập), bước 4 (hình quỹ đạo) |
| **A** | `isic_utils.py`                                                 |


**Các chỗ phụ thuộc:**

```
common.py (E) ─────────► C, D, B dùng để đọc dữ liệu giả            (không chặn)
isic_utils.py (A) ─────► D: E4 phần 2 · B: E6 bước 3 · A: E5        (CHẶN — làm trước)
isic_baselines.py (B) ─► A: E5 bước 2                                (không chặn)
h tốt nhất 3D Lab (D) ─► A: E5 bước 2                                (không chặn)
```


| Phụ thuộc               | Người cần | Cách để không phải chờ                                                                                                                           |
| ----------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `common.py` (E)         | C, D, B   | Chưa merge thì tạm tự đọc CSV ngay trong script (vài dòng `np.loadtxt`); merge xong thì đổi sang dùng chung.                                     |
| `isic_utils.py` (A)     | D, B, A   | **Phụ thuộc thật duy nhất.** A ưu tiên merge trong **2–3 ngày đầu tuần 3**. Trong lúc đó D, B làm phần giả lập (phần ISIC vốn xếp vào tuần 4–5). |
| `isic_baselines.py` (B) | A         | File ngắn, B làm ~1 ngày. A chỉ cần ở bước chỉnh tham số E5 — lúc đó đã có.                                                                      |
| h tốt nhất 3D Lab (D)   | A         | Không chặn: A tự quét lưới h khi chỉnh tham số trên tập tune; kết quả của D dùng để đối chiếu.                                                   |


**Lịch tổng:**

```
Tuần 3 (2–3 ngày đầu)  A: isic_utils.py · B: isic_baselines.py · E: common.py   → merge TRƯỚC
Tuần 3–4               E1, E2, E3, phần giả lập của E4 và E6                     (song song)
Tuần 4–5               phần ISIC của E4, E6; E5 chỉnh tham số → chấm test        (cần isic_utils.py)
Cuối tuần 5            mọi thực nghiệm xong, chép hình/bảng chốt vào report/
```



#### Giao diện file dùng chung — chốt TRƯỚC khi code

Ai cần dùng thì **viết sẵn code gọi theo đúng chữ ký dưới đây**, không cần chờ file được merge. Muốn đổi → báo cả nhóm.

```
experiments/common.py            (E)
  DATASETS = ["blobs", "blobs_varied", "blobs_aniso", "moons", "circles"]
  load_synthetic(name, seed) -> (X: ndarray (n, 2), y: ndarray (n,))
  timeit(fn, repeat=3) -> (kết quả của fn(), thời gian trung vị (giây))
  out_dir(name) -> Path            # results/<name>/, tự tạo nếu chưa có
  TRUE_K = {"blobs": 4, "blobs_varied": 3, ...}   # số cụm đúng của từng bộ
  synthetic_path(name, seed) -> Path               # đường dẫn file CSV
  # Dùng trong script: from common import DATASETS, load_synthetic, out_dir

experiments/isic_utils.py        (A)
  load_split(name) -> list[str]                # name: "tune" | "test"
  load_image(image_id, max_side=128) -> (rgb: ndarray (H, W, 3) uint8, mask: ndarray (H, W) bool)
  to_lab(rgb) -> ndarray (H, W, 3) float
  features(rgb, mode, hs=None, hr=None) -> ndarray (H*W, d)
                                               # mode: "rgb3" | "lab3" | "lab5" (lab5 cần hs, hr)
  labels_to_mask(labels, rgb) -> ndarray (H, W) bool
                                               # labels: ndarray (H*W,) — nhãn cụm của từng pixel
  dice(pred, gt) -> float
  iou(pred, gt) -> float

experiments/isic_baselines.py    (B)
  otsu_labels(rgb) -> ndarray (H*W,)           # Otsu trên kênh L, nhãn 0/1
  kmeans_labels(feat, k=2) -> ndarray (H*W,)   # random_state=0
  gmm_labels(feat, k=2) -> ndarray (H*W,)      # random_state=0
```

Ví dụ một vòng E5 / E4 phần 2 (ai cũng viết được ngay từ bây giờ):

```python
from isic_utils import load_split, load_image, features, labels_to_mask, dice
from meanshift import MeanShiftCustom

for image_id in load_split("tune"):
    rgb, gt = load_image(image_id)
    X = features(rgb, "lab3")
    labels = MeanShiftCustom(bandwidth=8, bin_seeding=True).fit_predict(X)
    pred = labels_to_mask(labels, rgb)
    print(image_id, dice(pred, gt))
```



### Quy ước chung cho mọi script

1. **Chạy bằng 1 lệnh**: `python3 experiments/eX_....py`. Có cờ chạy bản rút gọn để người review chạy thử: `--seeds 1` (dữ liệu giả, như E1/E2) hoặc `--quick` (ít ảnh).
2. **Seed cố định** (`random_state=0` hoặc theo seed của bộ dữ liệu) → chạy lại ra cùng số.
3. **Lưu 2 loại CSV**: `eX_runs.csv` (dữ liệu thô, mỗi dòng 1 lần chạy) và `eX_summary.csv` (gộp: trung bình ± độ lệch chuẩn).
4. **Hình**: PNG, `dpi=150`, có tiêu đề trục, chú thích (legend), chữ tiếng Việt có dấu.
5. **Đo thời gian**: `time.perf_counter()`, lặp 3 lần lấy **trung vị**. sklearn luôn chạy `n_jobs=1` cho công bằng.
6. **In tóm tắt ra màn hình** khi chạy xong (vài dòng số liệu chính).
7. **Viết** `report/THUC_NGHIEM_EX.md` (có lên git): cấu hình đã chạy, bảng số liệu chính, nhận xét, giới hạn → dùng để viết báo cáo. Mẫu: `report/THUC_NGHIEM_E1_E2.md`.
8. **Kết quả nào lên git** (đã cấu hình trong `.gitignore`): trong `results/eX/` chỉ đưa lên `*.png`, `*_summary.csv`, `*_params.csv`, `*_metadata.json`. `*_runs.csv`, mask và ảnh phân đoạn từng ảnh ISIC thì không lên git, vì chạy lại script là có. Commit kết quả **cùng PR với script** đã sinh ra nó. Không cần chép hình sang `report/figures/` nữa, báo cáo trỏ thẳng tới `results/eX/`.
9. Script trong `experiments/` không bắt buộc có test, nhưng vẫn đi qua PR và review như giai đoạn 1.
10. **Chọn tham số trên nhiều seed/ảnh, không chỉ 1.** Bài học từ E2: trên blobs_aniso, α chọn theo seed 0 lại kém h mặc định ở seed 1–9 (ARI 0.528 so với 0.657). Khi chọn h (E4, E5, E6) nên lấy trung bình trên nhiều seed/ảnh, và **báo cáo kết quả trên phần dữ liệu không dùng để chọn**.

---



### E – E1 (làm nốt) và E2

**E1 – Đối chiếu sklearn** · yêu cầu chi tiết đã ghi trong docstring `experiments/e1_compare_sklearn.py`.

- [x] Viết `experiments/common.py` dùng chung cho cả nhóm:
  - `DATASETS = ["blobs", "blobs_varied", "blobs_aniso", "moons", "circles"]`
  - `load_synthetic(name, seed) -> (X, y)` (đọc `data/synthetic/<name>_seed<k>.csv`)
  - `timeit(fn, repeat=3) -> (kết quả, thời gian trung vị)`
  - `out_dir(name) -> Path` (tạo `results/<name>/` nếu chưa có)
- [x] Chạy lưới: 5 bộ × 10 seed × α ∈ {0.25, 0.5, 1.0} × bin_seeding {tắt, bật} × {brute, kdtree} = 600 lần.
- [x] Liệt kê và **giải thích** các lần chạy khác sklearn (nếu có).


| Đầu ra                       | Nội dung                                                                                                                                                                                                |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results/e1/e1_runs.csv`     | Mỗi dòng 1 lần chạy: bộ, seed, α, h, bin_seeding, search, số cụm 2 bên, ARI giữa 2 bên, lệch tâm lớn nhất / h, nhãn giống hệt (True/False), n_iter 2 bên, thời gian 2 bên, ARI mỗi bên so với nhãn đúng |
| `results/e1/e1_summary.csv`  | Gộp theo (bộ, α, bin_seeding, search): % cùng số cụm, % nhãn giống hệt, ARI nhỏ nhất, lệch tâm lớn nhất                                                                                                 |
| `results/e1/e1_examples.png` | 5 hàng (5 bộ) × 2 cột (custom | sklearn), seed 0, α = 0.5                                                                                                                                               |
| `report/THUC_NGHIEM_E1_E2.md` | Kết luận RQ1 (đã có)                                                                                                                                                                                            |


> **Kết quả E1 (đã xong):** 600/600 lần chạy cùng số cụm và nhãn giống hệt sklearn, lệch tâm lớn nhất ≈ 1.6·10⁻¹⁴·h. E1 vẽ thêm quỹ đạo Gaussian trên KDE và mật độ theo vòng lặp (`e1_kde_trajectory.png`, `e1_density_steps.png`) — B dùng lại cho Bước 4 của E6.

**E2 – So với K-Means, DBSCAN, GMM**

- [x] Dữ liệu: 5 bộ × 10 seed (`load_synthetic`). Chuẩn hoá `StandardScaler` trước khi chạy (ghi rõ trong NOTES).
- [x] Cấu hình từng thuật toán (chọn trên **seed 0**, giữ nguyên cho seed 1–9):
  - **Mean-Shift** (`MeanShiftCustom`): (a) h = `estimate_bandwidth(X)` mặc định — không biết số cụm; (b) h tốt nhất trong lưới α·h₀, α ∈ {0.25, 0.5, 0.75, 1, 1.5}.
  - **K-Means**: k = số cụm đúng, `n_init=10`.
  - **GMM**: k = số cụm đúng, `covariance_type="full"`.
  - **DBSCAN**: `min_samples=5`, eps chọn trong lưới (ví dụ 0.05 → 0.5) cho ARI cao nhất trên seed 0. Điểm nhiễu (−1) giữ nguyên khi tính ARI.
- [x] Đo: ARI, NMI (so với nhãn đúng), số cụm tìm được, thời gian chạy.
- [x] Nhận xét: thuật toán nào thắng ở bộ nào, vì sao (moons/circles không lồi; blobs_varied mật độ khác nhau; K-Means/GMM được cho biết trước k).


| Đầu ra                      | Nội dung                                                                          |
| --------------------------- | --------------------------------------------------------------------------------- |
| `results/e2/e2_params.csv`  | Tham số đã chọn trên seed 0 cho từng (bộ, thuật toán)                             |
| `results/e2/e2_runs.csv`    | Mỗi dòng: bộ, seed, thuật toán, ARI, NMI, số cụm, thời gian                       |
| `results/e2/e2_summary.csv` | Bảng ARI, NMI trung bình ± std: hàng = bộ, cột = thuật toán                       |
| `results/e2/e2_grid.png`    | Lưới 5 hàng (bộ) × 5 cột (MS mặc định, MS tốt nhất, K-Means, DBSCAN, GMM), seed 0 |
| `results/e2/e2_ari_bar.png` | Biểu đồ cột ARI trung bình (có thanh sai số) theo bộ × thuật toán                 |
| `report/THUC_NGHIEM_E1_E2.md` | Kết luận RQ2 (đã có)                                                                      |

> **Kết quả E2 (đã xong, ARI trung bình seed 1–9):** GMM tốt nhất trên 3 bộ blobs; DBSCAN ≈ 1.0 trên moons và circles; Mean-Shift với h chọn gần K-Means/GMM trên blobs (0.902) dù không biết số cụm, nhưng không tách được moons/circles. Chi tiết: `report/THUC_NGHIEM_E1_E2.md`.


---

### C – E3: Tốc độ & bộ nhớ

- [ ] Sinh dữ liệu ngay trong script: `make_blobs(n, centers=5, cluster_std=1.0, random_state=0)`, d = 2.
- [ ] **Theo n**: n ∈ {500, 1.000, 2.000, 5.000, 10.000, 20.000, 50.000}. h **cố định** = `estimate_bandwidth` tính một lần trên mẫu 2.000 điểm (để chỉ n thay đổi).
- [ ] **Theo h**: n = 5.000, h = α·h₀ với α ∈ {0.25, 0.5, 1, 2}.
- [ ] Các bản cần đo:
  - custom brute · custom kdtree · custom kdtree + bin_seeding
  - sklearn `MeanShift(n_jobs=1)` · sklearn + bin_seeding
- [ ] Thời gian: trung vị 3 lần. **Bộ nhớ đỉnh**: `tracemalloc` (bắt đầu trước `fit`, lấy `get_traced_memory()[1]`).
- [ ] Giới hạn: lần chạy nào > 10 phút thì bỏ các n lớn hơn của bản đó, ghi "quá thời gian" vào CSV.
- [ ] Ghi cấu hình máy (CPU, RAM, phiên bản Python/NumPy/sklearn) vào `env.txt`.
- [ ] Nhận xét: KD-Tree nhanh hơn brute từ n bao nhiêu; khi h lớn KD-Tree còn lợi không; bin seeding giảm số seed và thời gian bao nhiêu.


| Đầu ra                          | Nội dung                                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------------------------- |
| `results/e3/e3_runs.csv`        | Mỗi dòng: thí nghiệm (n / h), n, h, bản, lần lặp, thời gian (s), bộ nhớ đỉnh (MB), số seed, số cụm |
| `results/e3/e3_summary.csv`     | Trung vị thời gian, bộ nhớ theo (n, bản) và (h, bản); hệ số tăng tốc so với brute                  |
| `results/e3/e3_time_vs_n.png`   | Thời gian theo n, trục log–log, mỗi bản 1 đường                                                    |
| `results/e3/e3_memory_vs_n.png` | Bộ nhớ đỉnh theo n                                                                                 |
| `results/e3/e3_time_vs_h.png`   | Thời gian theo h (n = 5.000)                                                                       |
| `results/e3/env.txt`            | Cấu hình máy và phiên bản thư viện                                                                 |
| `report/THUC_NGHIEM_E3.md`           | Kết luận RQ3                                                                                       |


---



### D – E4: Độ nhạy theo h

**Phần 1 – Dữ liệu giả lập** (làm ngay tuần 3)

- [ ] 5 bộ × 10 seed; h = α·h₀ với h₀ = `estimate_bandwidth(X)`, α = `np.geomspace(0.1, 3, 15)`.
- [ ] Mỗi lần chạy ghi: số cụm, ARI, NMI, thời gian (bật `bin_seeding` cho nhanh, ghi rõ).
- [ ] Đánh dấu α = 1 (h mặc định) trên hình; so h mặc định với h tốt nhất.

**Phần 2 – ISIC tune** (tuần 4–5, sau khi `isic_utils.py` merge)

- [ ] 100 ảnh tune, đặc trưng 3D Lab (`isic_utils`), Mean-Shift `bin_seeding=True`.
- [ ] h ∈ {2, 4, 6, 8, 10, 12, 16, 20} (đơn vị Lab) + h = `estimate_bandwidth` của từng ảnh.
- [ ] Mỗi (ảnh, h) ghi: Dice, IoU, số cụm, thời gian. Tìm h cho Dice trung bình cao nhất → **gửi A dùng cho E5 (Mean-Shift 3D Lab)**.


| Đầu ra                             | Nội dung                                                                               |
| ---------------------------------- | -------------------------------------------------------------------------------------- |
| `results/e4/e4_synthetic_runs.csv` | Mỗi dòng: bộ, seed, α, h, số cụm, ARI, NMI, thời gian                                  |
| `results/e4/e4_synthetic.png`      | 2 hình con: số cụm theo α và ARI theo α, mỗi bộ 1 đường (trung bình ± std qua 10 seed) |
| `results/e4/e4_isic_runs.csv`      | Mỗi dòng: ảnh, h, Dice, IoU, số cụm, thời gian                                         |
| `results/e4/e4_isic_h.png`         | Dice trung bình (± std) theo h trên 100 ảnh tune                                       |
| `results/e4/e4_isic_examples.png`  | 1–2 ảnh, mỗi ảnh 4 mức h: ảnh gốc, cụm, mask                                           |
| `results/e4/best_h_3d_lab.json`    | h tốt nhất cho 3D Lab (A dùng ở E5)                                                    |
| `report/THUC_NGHIEM_E4.md`              | Kết luận RQ4                                                                           |


---



### A – E5: Phân đoạn tổn thương da

**Bước 1 –** `experiments/isic_utils.py` (merge **đầu tuần 3**, B và D dùng chung)

- [ ] `load_split(name) -> list[id]` đọc `data/splits/isic_tune.txt` / `isic_test.txt`
- [ ] `load_image(id) -> (ảnh RGB, mask)`: thu nhỏ cạnh dài về 128 px (mask thu nhỏ cùng cỡ, nearest)
- [ ] `to_lab(rgb)`: đổi RGB → Lab (thêm `scikit-image` vào `requirements.txt`, ghi trong PR)
- [ ] `features(img, mode, hs=None, hr=None)`: `"rgb3"` = (R, G, B); `"lab3"` = (L, a, b); `"lab5"` = (x/hs, y/hs, L/hr, a/hr, b/hr)
- [ ] `labels_to_mask(labels, img)`: quy tắc chung cho **mọi** phương pháp — bỏ cụm phủ phần lớn viền ảnh → chọn cụm tối nhất (L trung bình nhỏ nhất) → lấp lỗ (`scipy.ndimage.binary_fill_holes`) → giữ vùng liền lớn nhất (`scipy.ndimage.label`)
- [ ] `dice(pred, gt)`, `iou(pred, gt)`
- [ ] Test nhanh trên 3–5 ảnh, lưu hình kiểm tra `results/e5/check_utils.png`

**Bước 2 – Chỉnh tham số trên 100 ảnh tune**

- [ ] Mean-Shift 3D RGB, 3D Lab: lưới h (lấy kết quả từ D cho 3D Lab).
- [ ] Mean-Shift 5D Lab: lưới hs ∈ {8, 16, 32}, hr ∈ {4, 8, 12} (bandwidth = 1 sau khi chia tỷ lệ).
- [ ] Baseline (của B): Otsu kênh L, K-Means k = 2, GMM k = 2 — không cần chỉnh.
- [ ] Giữ bộ tham số có Dice trung bình cao nhất cho mỗi phương pháp.

**Bước 3 – Chấm trên 200 ảnh test (chạy 1 lần duy nhất)**

- [ ] 6 phương pháp: Otsu, K-Means, GMM, MS 3D RGB, MS 3D Lab, MS 5D Lab.
- [ ] Đo Dice, IoU, thời gian/ảnh.
- [ ] Wilcoxon (`scipy.stats.wilcoxon`) giữa Mean-Shift tốt nhất và từng phương pháp còn lại.
- [ ] Chạy thêm sklearn `MeanShift` với cùng tham số trên ~20 ảnh để xem mask giống/khác bản tự viết.
- [ ] Phân tích lỗi: xem ảnh tốt nhất, trung vị, tệ nhất; xếp nhóm nguyên nhân (lông, viền đen ống kính, tổn thương nhạt màu, nhiều màu).


| Đầu ra                           | Nội dung                                                             |
| -------------------------------- | -------------------------------------------------------------------- |
| `results/e5/check_utils.png`     | Kiểm tra pipeline trên vài ảnh                                       |
| `results/e5/e5_tune_runs.csv`    | Mỗi dòng: ảnh, phương pháp, tham số, Dice, IoU, thời gian            |
| `results/e5/e5_best_params.json` | Tham số tốt nhất của từng phương pháp                                |
| `results/e5/e5_test_runs.csv`    | 200 ảnh × 6 phương pháp: Dice, IoU, thời gian                        |
| `results/e5/e5_test_summary.csv` | Bảng kết quả chính: Dice, IoU (trung bình ± std), thời gian/ảnh      |
| `results/e5/e5_wilcoxon.csv`     | p-value Mean-Shift tốt nhất vs từng phương pháp                      |
| `results/e5/e5_vs_sklearn.csv`   | Dice giữa mask bản tự viết và mask sklearn trên ~20 ảnh              |
| `results/e5/e5_examples.png`     | Ảnh tốt / trung vị / tệ: ảnh gốc, mask bác sĩ, mask từng phương pháp |
| `results/e5/e5_failures.png`     | Ví dụ từng nhóm lỗi                                                  |
| `results/e5/e5_dice_box.png`     | Boxplot Dice của 6 phương pháp trên 200 ảnh test                     |
| `report/THUC_NGHIEM_E5.md`            | Kết luận RQ5                                                         |


---



### B – E6: Ablation + hình quỹ đạo + baseline cho E5

**Bước 1 –** `experiments/isic_baselines.py` (merge **đầu tuần 3**, A dùng cho E5)

- [ ] `otsu_labels(img)`: Otsu trên kênh L
- [ ] `kmeans_labels(feat, k=2)`, `gmm_labels(feat, k=2)` (`random_state=0`)
- [ ] Trả về nhãn cùng dạng với Mean-Shift để đưa chung vào `labels_to_mask` của A.

**Bước 2 – Ablation trên dữ liệu giả lập** (5 bộ × 10 seed, h = h₀)

- [ ] **Flat vs Gaussian**: ARI, số cụm, n_iter_, thời gian.
- [ ] **Gaussian cutoff** ∈ {2, 3, 4}: kết quả đổi bao nhiêu, thời gian đổi bao nhiêu.
- [ ] **Bin seeding tắt/bật**: số seed, thời gian, ARI, có đổi số cụm không.

**Bước 3 – Ablation trên ISIC tune** (sau khi `isic_utils.py` merge)

- [ ] 3D Lab, h tốt nhất của D: Flat vs Gaussian, bin seeding tắt/bật → Dice, thời gian/ảnh.

**Bước 4 – Hình minh hoạ cho Chương 1 và mục 3.2** (`experiments/fig_trajectories.py`)

> E1 đã có `results/e1/e1_kde_trajectory.png` (quỹ đạo 1 seed trên KDE) và `e1_density_steps.png` (mật độ theo vòng lặp). Xem trước; nếu đủ dùng cho báo cáo thì **bỏ Bước 4**, chỉ làm phần còn thiếu dưới đây.

- [ ] Quỹ đạo của **nhiều** seed (~20) trên `blobs` (`record_trajectories=True`) vẽ chồng lên đường đồng mức KDE — hình E1 chỉ có 1 seed.
- [ ] (Không bắt buộc) f̂(yₜ) theo vòng lặp t của vài seed — đã có ở `e1_density_steps.png`.


| Đầu ra                           | Nội dung                                                           |
| -------------------------------- | ------------------------------------------------------------------ |
| `results/e6/e6_kernel.csv`       | Flat vs Gaussian: bộ, seed, kernel, ARI, số cụm, n_iter, thời gian |
| `results/e6/e6_cutoff.csv`       | Gaussian cutoff 2/3/4: ARI, số cụm, thời gian                      |
| `results/e6/e6_binseed.csv`      | Bin seeding tắt/bật: số seed, thời gian, ARI, số cụm               |
| `results/e6/e6_summary.csv`      | Bảng gộp các ablation (trung bình ± std)                           |
| `results/e6/e6_isic.csv`         | Ablation trên 100 ảnh tune: Dice, thời gian/ảnh                    |
| `results/e6/e6_trajectories.png` | Quỹ đạo seed trên đường đồng mức KDE                               |
| `results/e6/e6_density_path.png` | f̂(yₜ) theo vòng lặp t                                             |
| `report/THUC_NGHIEM_E6.md`            | Kết luận ablation (bổ sung RQ3, RQ5)                               |


---



### Xong giai đoạn 2 khi

- [ ] 6 script chạy được bằng 1 lệnh, đã merge vào `main`.
- [ ] Mỗi `results/eX/` có đủ file ở bảng trên; mỗi thực nghiệm có `report/THUC_NGHIEM_EX.md`.
- [ ] Hình và bảng tổng hợp trong `results/eX/` đã lên git (quy ước 8).
- [ ] Mỗi RQ1–RQ5 có ít nhất 1 bảng hoặc 1 hình trả lời.



## Giai đoạn 3 – Báo cáo & slide (tuần 6–7)

- Ai làm thực nghiệm nào viết phần kết quả của thực nghiệm đó.
- **Chương 1 (lý thuyết):** B, bắt đầu ngay từ tuần 1.
- **Chương 2 (thiết kế OOP):** A + mỗi người 1 đoạn mô tả file mình code.
- A gộp báo cáo, làm slide; cả nhóm duyệt lần cuối.

---



## Quy trình làm việc với Git (từ đầu đến cuối)



### Lần đầu tiên (mỗi người làm 1 lần)

```bash
git clone https://github.com/thanhgiang0607/ML_Final.git
cd ML_Final
pip install -r requirements.txt
python3 data/make_synthetic.py           # sinh dữ liệu giả
python3 data/download_isic.py            # chỉ người làm E4/E5/E6 cần
python3 -m pytest                        # kiểm tra môi trường: không có FAILED
```



### Mỗi phần việc — ví dụ B làm `kernels.py`



#### Phía thành viên

**1. Lấy code mới nhất, tạo branch riêng**

```bash
git checkout main
git pull
git checkout -b oop-kernels
```

**2. Code và test xen kẽ**

- Viết phần ruột ở các chỗ `# TODO(B)` trong `meanshift/kernels.py`.
- Viết thêm test vào `tests/test_kernels.py`.
- Chạy liên tục trong lúc code: `python3 -m pytest tests/test_kernels.py` — sai thì sửa ngay.

**3. Code xong → kiểm tra lần cuối trên máy**

```bash
git pull origin main          # gộp những gì người khác đã merge vào main
python3 -m pytest             # toàn bộ test: không có FAILED
```

Nếu `git pull` báo **conflict**: mở file bị xung đột, giữ phần đúng, rồi `git add` + `git commit`.

**4. Commit và push lên branch của mình**

```bash
git add meanshift/kernels.py tests/test_kernels.py
git commit -m "Cài FlatKernel, GaussianKernel và test"
git push -u origin oop-kernels
```

**5. Tạo Pull Request trên GitHub**

- Vào repo → bấm **Compare & pull request**.
- Tiêu đề: *"B – kernels.py"*. Mô tả: đã làm gì, kết quả pytest trên máy (ví dụ *9 passed, 22 skipped*).
- **Reviewers**: chọn trưởng nhóm (hoặc 1 thành viên khác).
- Bấm **Create pull request**.

**6. Đợi GitHub tự chạy test (~1–2 phút)**

- ✅ → chờ review.
- ❌ → bấm **Details** xem test nào sai → sửa trên máy → `git add` / `git commit` / `git push` lên **cùng branch**. PR tự cập nhật, GitHub chạy lại test — **không tạo PR mới**.



#### Phía người review (trưởng nhóm)

**7. Review**

- Xem dấu ✅. Bấm *Details* kiểm tra test của phần vừa code đã **passed** (không còn **skipped**).
- Mở tab **Files changed** đọc code.
- Cần sửa → comment vào dòng đó, chọn **Request changes** → thành viên sửa, push lại (quay lại bước 6).
- Ổn → **Approve**.

**8. Merge**

- Bấm **Merge pull request** → **Confirm**.
- Bấm **Delete branch** để xoá branch trên GitHub.
- Đánh dấu ☑ ở mục *Tiến độ* cuối file này.



#### Sau khi merge

**9. Thành viên dọn dẹp, chuẩn bị việc tiếp theo**

```bash
git checkout main
git pull                       # main giờ đã có code vừa merge
git branch -d oop-kernels      # xoá branch cũ trên máy
```

Việc mới → quay lại bước 1 với branch mới (ví dụ `e6-ablation`).

**10. Các thành viên khác** `git pull` để nhận code mới (đang ở branch riêng thì `git pull origin main`).

### Tóm tắt

```
Thành viên:  tạo branch → code + test → pytest trên máy → push → tạo PR
                                                                    ↓
GitHub:      tự chạy test → ✅ / ❌   (❌: sửa, push lại cùng branch)
                                                                    ↓
Review:      đọc code → Approve → Merge → xoá branch
                                                                    ↓
Cả nhóm:     git pull để nhận code mới
```

**Lưu ý:** ✅ chỉ có nghĩa là *các test hiện có đều pass*, không có nghĩa code chắc chắn đúng — người review vẫn phải đọc code.

## Quy tắc

1. **Không push thẳng lên** `main`**.** Mọi thay đổi đi qua Pull Request, có ít nhất 1 người khác review.
2. **Chỉ merge khi test pass.** GitHub tự chạy `pytest` cho mỗi PR (GitHub Actions) — xem dấu ✅/❌ ở cuối trang PR, ❌ thì bấm *Details* để xem test nào sai. Nên tự chạy `python3 -m pytest` trên máy trước khi push cho nhanh. Test của phần chưa code hiện `SKIPPED (Chưa cài)` — bình thường.
3. **Code trong** `meanshift/` **phải có test** ở `tests/test_<tên file>.py`. Đã có vài test mẫu, mỗi người bổ sung thêm. Script trong `experiments/` không bắt buộc có test.
4. **Chỉ sửa file của mình.** Cần sửa file người khác → nhắn người đó hoặc nêu trong PR.
5. **Chỉ dùng NumPy/SciPy trong** `meanshift/`**.** sklearn chỉ được dùng trong `tests/` và `experiments/` để đối chiếu.
6. **Commit message tiếng Việt, ngắn gọn**, nói rõ đã làm gì.



## Tiến độ (cập nhật khi xong)

- [x] B – `kernels.py` merged
- [x] C – `neighbors.py` merged
- [x] D – `bandwidth.py`, `seeding.py` merged
- [x] A – `core.py` merged, toàn bộ test pass
- [x] E – E1 chạy xong, có bảng + hình · `common.py` merged
- [x] E2 · [ ] E3 · [ ] E4 · [ ] E5 · [ ] E6
- [ ] Báo cáo · [ ] Slide