# THỰC NGHIỆM E5: PHÂN ĐOẠN TỔN THƯƠNG DA ISIC 2018

## 1. Mục tiêu

Trả lời **RQ5**: Mean-Shift phân đoạn tổn thương da trên ảnh soi da (ISIC 2018 Task 1) tốt đến đâu so với các phương pháp đơn giản phổ biến (Otsu, K-Means, GMM), và cách biểu diễn pixel (RGB, Lab, Lab + vị trí) ảnh hưởng thế nào.

Script: `experiments/e5_isic_segmentation.py`. Hàm dùng chung: `experiments/isic_utils.py`; baseline: `experiments/isic_baselines.py`.

## 2. Dữ liệu và quy trình

- **Tune:** 100 ảnh (`data/splits/isic_tune.txt`, lấy từ ISIC 2018 Validation) — chỉ dùng để chọn tham số.
- **Test:** 200 ảnh (`data/splits/isic_test.txt`, lấy từ ISIC 2018 Test) — chạy **một lần** sau khi đã chốt tham số.
- Ảnh thu nhỏ cạnh dài về 128 px; mask bác sĩ thu nhỏ cùng cỡ (nearest).

Mọi phương pháp đi qua **cùng một quy trình**, nên chênh lệch Dice chỉ đến từ thuật toán:

```
ảnh RGB -> đặc trưng -> thuật toán -> nhãn cụm từng pixel -> labels_to_mask -> Dice, IoU
```

`labels_to_mask` (quy tắc chung): bỏ cụm phủ ≥ 50% dải viền ảnh, cụm có ≥ 50% pixel nằm trên viền và cụm nhỏ hơn 1% ảnh → chọn cụm tối nhất (L trung bình nhỏ nhất) → lấp lỗ → giữ vùng liền lớn nhất.

| Phương pháp | Đặc trưng | Tham số |
|---|---|---|
| Otsu | Kênh L | không có |
| K-Means k = 2 | 3D Lab | `random_state=0`, `n_init=10` |
| GMM k = 2 | 3D Lab | hiệp phương sai đầy đủ, `random_state=0` |
| MS 3D RGB | (R, G, B), 0–255 | h chỉnh trên tune |
| MS 3D Lab | (L, a, b) | h = 8, lấy từ E4 (`results/e4/best_h_3d_lab.json`) |
| MS 5D Lab | (x/hs, y/hs, L/hr, a/hr, b/hr), bandwidth = 1 | hs, hr chỉnh trên tune |

Mean-Shift luôn là `MeanShiftCustom` (kernel phẳng, KD-Tree, `bin_seeding=True`). Ký hiệu hs (bán kính theo vị trí, pixel) và hr (bán kính theo màu, đơn vị Lab) theo Comaniciu & Meer (2002).

Thời gian đo từ ảnh RGB tới nhãn cụm (gồm cả bước tạo đặc trưng), trên Apple M1, 1 tiến trình; ở test lấy trung vị 3 lần chạy.

## 3. Chỉnh tham số trên tập tune

Lưới: MS 3D RGB h ∈ {12, 16, 20, 24, 32, 40, 48}; MS 5D hs ∈ {8, 16, 32, 64} × hr ∈ {4, 8, 12, 16, 20}. Số liệu đầy đủ ở `results/e5/e5_tune_summary.csv`.

**Mở rộng lưới MS 5D.** Lưới ban đầu theo kế hoạch là hs ∈ {8, 16, 32}, hr ∈ {4, 8, 12}. Bản chạy thử cho tốt nhất ở hs = 32 (mép lưới) nên thêm hs = 64; tune 100 ảnh lại cho tốt nhất ở hr = 12 (mép lưới) nên thêm hr = 16, 20 rồi chạy lại toàn bộ. Kết quả cuối, Dice trung bình trên tune:

| hs \ hr | 4 | 8 | 12 | 16 | 20 |
|---|---:|---:|---:|---:|---:|
| 8 | 0,043 | 0,129 | 0,156 | 0,160 | 0,182 |
| 16 | 0,146 | 0,309 | 0,372 | 0,466 | 0,504 |
| 32 | 0,247 | 0,506 | 0,607 | 0,636 | **0,646** |
| 64 | 0,247 | 0,544 | 0,587 | 0,505 | 0,395 |

hr = 20 vẫn nằm ở mép lưới, nhưng nhóm dừng ở đây: với hs = 32, mỗi lần tăng hr chỉ còn thêm +0,03 rồi +0,01; và ở hàng hs = 64, Dice đã đạt đỉnh ở hr = 12 rồi giảm mạnh (tỷ lệ mask rỗng tăng lên 31–46%) vì tổn thương bị gộp vào da. hs hoặc hr quá nhỏ thì ảnh bị cắt vụn (hs = 8, hr = 4: trung vị 255 cụm/ảnh).

**MS 3D RGB:** Dice tăng từ 0,284 (h = 12) lên 0,464 (h = 24) rồi giảm còn 0,378 (h = 48); chọn h = 24.

**Tham số chốt** (`results/e5/e5_best_params.json`):

| Phương pháp | Tham số | Dice tune |
|---|---|---:|
| Otsu | – | 0,568 |
| K-Means k = 2 | – | 0,718 |
| GMM k = 2 | – | 0,735 |
| MS 3D RGB | h = 24 | 0,464 |
| MS 3D Lab | h = 8 | 0,567 |
| MS 5D Lab | hs = 32, hr = 20 | **0,646** (Mean-Shift tốt nhất) |

## 4. Kết quả trên 200 ảnh test

### 4.1. Bảng chính

Lấy từ `results/e5/e5_test_summary.csv`. "Ảnh hỏng" = Dice < 0,1.

| Phương pháp | Dice TB ± std | Dice trung vị | IoU TB | Ảnh hỏng | Thời gian/ảnh (trung vị) | Dice tune |
|---|---:|---:|---:|---:|---:|---:|
| **GMM k = 2** | **0,702 ± 0,318** | **0,868** | **0,619** | 8,0% | 0,025 s | 0,735 |
| K-Means k = 2 | 0,686 ± 0,323 | 0,864 | 0,601 | 9,0% | 0,029 s | 0,718 |
| **MS 5D Lab** | **0,631 ± 0,328** | 0,772 | 0,538 | 11,0% | 0,875 s | 0,646 |
| MS 3D Lab | 0,579 ± 0,379 | 0,747 | 0,502 | 23,5% | 0,548 s | 0,567 |
| Otsu | 0,551 ± 0,326 | 0,587 | 0,451 | 12,0% | 0,001 s | 0,568 |
| MS 3D RGB | 0,477 ± 0,359 | 0,572 | 0,389 | 28,0% | 0,601 s | 0,464 |

Thứ hạng trên test giống hệt trên tune, Dice test lệch tune không quá 0,033 ở mọi phương pháp, nên việc chọn tham số trên 100 ảnh tune không bị khớp quá mức.

Hình `results/e5/e5_dice_box.png`: boxplot Dice của 6 phương pháp.

### 4.2. Kiểm định Wilcoxon

Wilcoxon cặp (cùng ảnh), hai phía, giữa MS 5D Lab (Mean-Shift tốt nhất, chọn trên tune) và từng phương pháp còn lại; p-value hiệu chỉnh Holm cho 5 phép so sánh (`results/e5/e5_wilcoxon.csv`).

| MS 5D Lab so với | Chênh Dice TB | Chênh trung vị | MS 5D tốt hơn / kém hơn / bằng (số ảnh) | p (Holm) | Kết luận (α = 0,05) |
|---|---:|---:|:-:|---:|---|
| GMM | −0,071 | −0,047 | 71 / 125 / 4 | 0,0002 | MS 5D **kém hơn** có ý nghĩa |
| K-Means | −0,055 | −0,046 | 55 / 143 / 2 | 2·10⁻⁶ | MS 5D **kém hơn** có ý nghĩa |
| Otsu | +0,080 | +0,003 | 106 / 92 / 2 | 0,067 | **Không** khác biệt có ý nghĩa |
| MS 3D Lab | +0,052 | +0,012 | 105 / 86 / 9 | 0,034 | MS 5D **tốt hơn** có ý nghĩa |
| MS 3D RGB | +0,154 | +0,086 | 147 / 42 / 11 | 1,5·10⁻¹³ | MS 5D **tốt hơn** có ý nghĩa |

### 4.3. Đối chiếu với sklearn

Trên 20 ảnh test đầu, chạy `sklearn.cluster.MeanShift(bin_seeding=True)` với cùng đặc trưng và bandwidth cho cả 3 biến thể Mean-Shift (`results/e5/e5_vs_sklearn.csv`): **ARI giữa nhãn hai bên = 1 và Dice giữa hai mask = 1 ở cả 60/60 lần chạy**, tức `MeanShiftCustom` cho kết quả trùng khớp sklearn trên ảnh thật, khớp với kết luận E1 trên dữ liệu giả lập.

## 5. Phân tích lỗi

Hình `results/e5/e5_examples.png`: ảnh tốt nhất, trung vị, tệ nhất của MS 5D Lab, kèm mask của cả 6 phương pháp. Hình `results/e5/e5_failures.png`: 8 ảnh tệ nhất.

Xem toàn bộ 22 ảnh test mà MS 5D Lab hỏng (Dice < 0,1) và xếp theo nguyên nhân chính (`results/e5/e5_failure_groups.csv`, phân loại bằng mắt):

| Nhóm | Số ảnh | Mô tả | Dice TB của GMM trên cùng ảnh |
|---|---:|---|---:|
| Cắt theo vị trí | 12 | Tổn thương nhỏ (so với hs = 32) bị gộp chung với vùng da quanh nó thành một khối lớn; ranh giới cụm là các đường gần thẳng cắt ngang ảnh, cụm được chọn chứa chủ yếu là da | 0,41 |
| Viền đen ống kính | 5 | Vùng tối quanh ảnh tròn bị chia thành nhiều mảnh theo vị trí, có mảnh không thỏa điều kiện viền nên được chọn thay cho tổn thương | 0,15 |
| Thước đo | 3 | Vạch đen của thước đo là vùng tối nhất ảnh; tổn thương nhạt màu gần da | 0,00 |
| Tổn thương lớn chạm viền | 2 | Tổn thương phủ gần hết ảnh hoặc chạm viền nên bị quy tắc viền loại | 0,44 |

5/200 ảnh hỏng ở cả 6 phương pháp (tổn thương nhạt, có thước đo hoặc viền ống kính); 14/200 ảnh hỏng ở cả 3 biến thể Mean-Shift.

Ngoài ra, ở **tổn thương nhiều màu** (ví dụ ảnh trung vị ISIC_0036120 trong `e5_examples.png`), MS 3D Lab tách lõi tối và vùng nâu ngoài thành 2 cụm và chỉ giữ lõi (Dice 0,24), trong khi K-Means/GMM với k = 2 buộc phải gộp cả tổn thương vào một cụm (Dice 0,95–0,97); MS 5D giảm được lỗi này nhờ hr lớn (Dice 0,77).

## 6. Nhận xét

**Mean-Shift không thắng baseline phân cụm có biết trước k.** MS 5D Lab, biến thể tốt nhất, đạt Dice 0,631, thấp hơn GMM (0,702) và K-Means (0,686) có ý nghĩa thống kê, trong khi chậm hơn khoảng 30–35 lần. Lý do chính: ảnh soi da gần như luôn gồm 2 vùng (da, tổn thương), nên giả định k = 2 của K-Means/GMM khớp sẵn với bài toán; Mean-Shift phải tự tìm số cụm và hay tách tổn thương thành nhiều cụm (h nhỏ) hoặc gộp vào da (h lớn). Ưu điểm "không cần biết k" của Mean-Shift không phát huy khi k đã biết.

**Biểu diễn pixel quan trọng hơn thuật toán.** Cùng là Mean-Shift, chuyển RGB → Lab tăng Dice từ 0,477 lên 0,579, thêm vị trí (5D) tăng tiếp lên 0,631 và giảm tỷ lệ ảnh hỏng từ 23,5% xuống 11%. Lab tách độ sáng (L) khỏi màu (a, b) và có khoảng cách gần với cảm nhận màu hơn RGB, nên một bán kính h dùng được cho nhiều ảnh hơn. Thêm vị trí giúp các pixel liền nhau gom về một cụm, ít mảnh vụn hơn.

**Nhưng thêm vị trí tạo kiểu lỗi mới.** Với hs = 32 trên ảnh 128 px, tổn thương nhỏ bị gộp vào một khối da lớn (nhóm "cắt theo vị trí", 12/22 ảnh hỏng), và đây là các ảnh mà GMM vẫn làm được (GMM đạt Dice trung bình 0,41 trên nhóm này, 6/12 ảnh GMM > 0,5). Một hs duy nhất không phù hợp cả tổn thương lớn lẫn nhỏ — cùng vấn đề "một h cho mọi ảnh" đã thấy ở E4.

**Otsu ngang Mean-Shift tốt nhất về trung vị.** MS 5D hơn Otsu 0,080 Dice trung bình nhưng chênh trung vị chỉ 0,003 và Wilcoxon không có ý nghĩa (p = 0,067): Mean-Shift chỉ hơn Otsu rõ ở một số ảnh, còn phần lớn ảnh hai bên tương đương, trong khi Otsu nhanh hơn khoảng 850 lần.

**Lỗi chung của mọi phương pháp đến từ quy tắc chọn cụm.** Thước đo, viền đen ống kính và tổn thương chạm viền làm hỏng cả baseline lẫn Mean-Shift, vì `labels_to_mask` chọn "cụm tối nhất không nằm trên viền". Cải thiện tiền xử lý (cắt viền ống kính, xóa lông/thước) sẽ có lợi cho mọi phương pháp, không riêng Mean-Shift.

**Trả lời RQ5.** Trên 200 ảnh test ISIC 2018, Mean-Shift tốt nhất (5D Lab + vị trí, hs = 32, hr = 20) đạt Dice 0,631 / IoU 0,538, tốt hơn có ý nghĩa so với Mean-Shift 3D Lab và 3D RGB, tương đương Otsu, nhưng kém có ý nghĩa so với K-Means và GMM k = 2 (Dice 0,686 và 0,702) và chậm hơn khoảng 30 lần. Với bài toán 2 vùng như phân đoạn tổn thương da, Mean-Shift không phải lựa chọn tốt nhất; giá trị của nó nằm ở việc không cần biết số cụm, và kết quả phụ thuộc mạnh vào cách biểu diễn pixel và bandwidth.

## 7. Giới hạn

- Tham số được chọn theo Dice trung bình trên 100 ảnh tune; mỗi phương pháp dùng một bộ tham số chung cho mọi ảnh. E4 cho thấy nếu chọn được h riêng từng ảnh (dùng mask thật), Dice của MS 3D Lab có thể đạt 0,747, nên khoảng cách giữa Mean-Shift và baseline một phần do dùng tham số chung.
- MS 5D: tham số tốt nhất nằm ở mép lưới hr (đã mở rộng lưới 2 lần, Dice đã chững, xem mục 3). Lưới thưa (hs cách nhau 2 lần).
- Baseline K-Means/GMM không được chỉnh tham số (k = 2 cố định, theo kế hoạch). Chỉ thử K-Means/GMM trên 3D Lab.
- Kết quả phụ thuộc quy tắc `labels_to_mask` (chọn một cụm tối nhất); phương pháp gộp nhiều cụm có thể đổi thứ hạng. Quy tắc được điều chỉnh sau khi xem 100 ảnh tune (xem `isic_utils.py`), không dùng ảnh test.
- Ảnh thu nhỏ còn 128 px; không có tiền xử lý xóa lông, cắt viền ống kính.
- Phân nhóm lỗi làm bằng mắt, chỉ trên 22 ảnh hỏng của MS 5D Lab.
- Thời gian đo trên Apple M1, 1 tiến trình, ảnh 128 px; Mean-Shift dùng KD-Tree và bin seeding.

## 8. Tái lập

```
python3 data/download_isic.py                                # tải ảnh (1 lần)
python3 experiments/e4_isic.py                               # h tốt nhất cho 3D Lab (E4 phần 2)
python3 experiments/e5_isic_segmentation.py --stage tune     # 100 ảnh tune, ~40 phút (Apple M1)
python3 experiments/e5_isic_segmentation.py --stage test     # 200 ảnh test, ~30 phút
```

Thêm `--quick` để chạy 10 ảnh, ghi ra thư mục tạm (không đụng `results/e5/`).

## 9. Đầu ra

| Tệp (results/e5/) | Nội dung |
|---|---|
| `e5_tune_runs.csv` | Mỗi (ảnh tune, phương pháp, tham số): Dice, IoU, số cụm, mask rỗng, thời gian; không lên git |
| `e5_tune_summary.csv` | Trung bình ± std theo (phương pháp, tham số) trên tune |
| `e5_best_params.json` | Tham số chốt của từng phương pháp, Mean-Shift tốt nhất |
| `e5_test_runs.csv` | 200 ảnh test × 6 phương pháp; không lên git |
| `e5_test_summary.csv` | Bảng kết quả chính trên test |
| `e5_wilcoxon.csv` | Wilcoxon MS 5D Lab so với từng phương pháp, p-value thô và Holm |
| `e5_vs_sklearn.csv` | So nhãn và mask với `sklearn.cluster.MeanShift` trên 20 ảnh test |
| `e5_failure_groups.csv` | 22 ảnh hỏng của MS 5D Lab, nhóm nguyên nhân, Dice các phương pháp |
| `e5_dice_box.png` | Boxplot Dice 6 phương pháp trên test |
| `e5_examples.png` | Ảnh tốt nhất, trung vị, tệ nhất (theo MS 5D Lab) |
| `e5_failures.png` | 8 ảnh test tệ nhất của MS 5D Lab |
| `check_utils.png` | Kiểm tra pipeline `isic_utils` trên 5 ảnh tune |
| `e5_metadata.json` | Cấu hình, phiên bản thư viện, SHA256 file chia tập |
