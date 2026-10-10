# Phân công nhóm – Đồ án Mean-Shift

> Đọc trước: `Proposal_MeanShift.docx` (đề tài, mục tiêu) · `README.md` (cấu trúc, lệnh chạy)

| Ký hiệu | Thành viên | Vai trò chính |
|---|---|---|
| A | _(điền tên)_ | Trưởng nhóm · `core.py` · ghép code · case study ảnh da |
| B | _(điền tên)_ | `kernels.py` · Chương 1 lý thuyết |
| C | _(điền tên)_ | `neighbors.py` · đo tốc độ |
| D | _(điền tên)_ | `bandwidth.py`, `seeding.py` · độ nhạy h |
| E | _(điền tên)_ | Đối chiếu sklearn (E1) · so sánh thuật toán (E2) |

---

## Giai đoạn 1 – Code OOP (tuần 1–2)

Các file trong `meanshift/` đã có **khung sẵn (stub)**: tên lớp, tên hàm, tham số và mô tả đầu vào/đầu ra đã chốt. Mỗi người chỉ cần viết phần ruột ở chỗ có `# TODO(<ký hiệu>)`.

| Người | File cần code | File test | Branch | Xong khi |
|---|---|---|---|---|
| **A** | `meanshift/core.py` | `tests/test_core.py` | `oop-core` | 2 blob xa nhau → đúng 2 cụm; `predict` khớp `labels_` |
| **B** | `meanshift/kernels.py` | `tests/test_kernels.py` | `oop-kernels` | Trọng số Flat = 1; Gaussian đúng `exp(-u/2)` |
| **C** | `meanshift/neighbors.py` | `tests/test_neighbors.py` | `oop-neighbors` | Brute-force và KD-Tree trả về cùng láng giềng |
| **D** | `meanshift/bandwidth.py`, `meanshift/seeding.py` | `tests/test_bandwidth.py`, `tests/test_seeding.py` | `oop-bandwidth-seeding` | Ra cùng kết quả với sklearn |
| **E** | `experiments/e1_compare_sklearn.py` | `tests/test_vs_sklearn.py` | `e1-compare` | Bảng + hình đối chiếu sklearn trên 5 bộ × 10 seed |

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

| Người | Thực nghiệm | Script | Dữ liệu |
|---|---|---|---|
| A | **E5** – Phân đoạn tổn thương da (Lab, 3D/5D, tạo mask, so Otsu/K-Means/GMM) | `experiments/e5_isic_segmentation.py` | ISIC: `data/splits/` |
| B | **E6** – Ablation (flat vs Gaussian, bật/tắt bin seeding) + hỗ trợ E5 | `experiments/e6_ablation.py` | Giả lập + ISIC tune |
| C | **E3** – Tốc độ & bộ nhớ (n = 500 → 50.000; theo h) | `experiments/e3_runtime.py` | Giả lập sinh riêng |
| D | **E4** – Độ nhạy theo h | `experiments/e4_bandwidth.py` | Giả lập + ISIC tune |
| E | **E2** – So với K-Means, DBSCAN, GMM | `experiments/e2_compare_algorithms.py` | `data/synthetic/` |

Kết quả lưu vào `results/<tên thực nghiệm>/` (thư mục này không đẩy lên git — gửi hình/bảng cuối cùng qua nhóm hoặc đưa thẳng vào báo cáo).

## Giai đoạn 3 – Báo cáo & slide (tuần 6–7)

- Ai làm thực nghiệm nào viết phần kết quả của thực nghiệm đó.
- **Chương 1 (lý thuyết):** B, bắt đầu ngay từ tuần 1.
- **Chương 2 (thiết kế OOP):** A + mỗi người 1 đoạn mô tả file mình code.
- A gộp báo cáo, làm slide; cả nhóm duyệt lần cuối.

---

## Quy trình Git cho mỗi người

```bash
# Lần đầu
git clone https://github.com/thanhgiang0607/ML_Final.git
cd ML_Final
pip install -r requirements.txt
python3 data/make_synthetic.py           # sinh dữ liệu giả
python3 data/download_isic.py            # chỉ người làm E4/E5/E6 cần

# Mỗi lần bắt đầu việc
git checkout main && git pull
git checkout -b <tên-branch>             # ví dụ: oop-kernels

# Code xong
python3 -m pytest                        # phải không có FAILED
git add <các file mình sửa>
git commit -m "Mô tả ngắn việc đã làm"
git push -u origin <tên-branch>
# -> Lên GitHub bấm "Compare & pull request", nhờ 1 người review
```

## Quy tắc

1. **Không push thẳng lên `main`.** Mọi thay đổi đi qua Pull Request, có ít nhất 1 người khác review.
2. **Chỉ merge khi `python3 -m pytest` không có test FAILED.** Test của phần chưa code sẽ hiện `SKIPPED (Chưa cài)` — bình thường.
3. **Code trong `meanshift/` phải có test** ở `tests/test_<tên file>.py`. Đã có vài test mẫu, mỗi người bổ sung thêm. Script trong `experiments/` không bắt buộc có test.
4. **Chỉ sửa file của mình.** Cần sửa file người khác → nhắn người đó hoặc nêu trong PR.
5. **Chỉ dùng NumPy/SciPy trong `meanshift/`.** sklearn chỉ được dùng trong `tests/` và `experiments/` để đối chiếu.
6. **Commit message tiếng Việt, ngắn gọn**, nói rõ đã làm gì.

## Tiến độ (cập nhật khi xong)

- [ ] B – `kernels.py` merged
- [ ] C – `neighbors.py` merged
- [ ] D – `bandwidth.py`, `seeding.py` merged
- [ ] A – `core.py` merged, toàn bộ test pass
- [ ] E – E1 chạy xong, có bảng + hình
- [ ] E2 · [ ] E3 · [ ] E4 · [ ] E5 · [ ] E6
- [ ] Báo cáo · [ ] Slide
