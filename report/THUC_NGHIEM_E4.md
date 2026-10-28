# THỰC NGHIỆM E4: KHẢO SÁT ĐỘ NHẠY CỦA MEAN SHIFT THEO BANDWIDTH

## 1. Mục tiêu

Thực nghiệm E4 khảo sát ảnh hưởng của tham số bandwidth đến kết quả phân cụm của thuật toán Mean Shift. Mục tiêu là đánh giá sự thay đổi về số lượng cụm, chất lượng phân cụm và thời gian thực thi khi bandwidth thay đổi, từ đó xác định mức bandwidth phù hợp cho từng bộ dữ liệu.

E4 gồm 2 phần: **phần 1** (mục 2–9) trên 5 bộ dữ liệu tổng hợp, script `experiments/e4_bandwidth.py`; **phần 2** (mục 10) trên 100 ảnh ISIC tune, script `experiments/e4_isic.py`, để chọn h cho Mean-Shift 3D Lab ở E5.

## 2. Dữ liệu thực nghiệm

Thực nghiệm sử dụng 5 bộ dữ liệu tổng hợp, mỗi bộ gồm 1.500 điểm dữ liệu và được tạo với 10 seed khác nhau.

| Bộ dữ liệu | Đặc điểm | Số cụm tham chiếu |
|---|---|---:|
| blobs | Các cụm có dạng tương đối đẳng hướng | 4 |
| blobs_varied | Các cụm có độ trải khác nhau | 3 |
| blobs_aniso | Các cụm biến đổi tuyến tính, bất đẳng hướng | 3 |
| moons | Hai cụm có dạng hình lưỡi liềm | 2 |
| circles | Hai cụm có dạng vòng tròn đồng tâm | 2 |

Trước khi phân cụm, dữ liệu được chuẩn hóa bằng StandardScaler. Việc này giúp đưa các đặc trưng về cùng thang đo, hạn chế ảnh hưởng của sự khác biệt về đơn vị đo đến khoảng cách giữa các điểm.

## 3. Cấu hình thực nghiệm

Bandwidth được điều chỉnh dựa trên giá trị bandwidth cơ sở do hàm `estimate_bandwidth()` ước lượng. Gọi giá trị cơ sở là `h0`, bandwidth sử dụng trong mỗi lần chạy được tính như sau:

`h = alpha * h0`

Trong đó:
- `h0`: bandwidth cơ sở được ước lượng từ dữ liệu.
- `alpha`: hệ số điều chỉnh bandwidth.
- `h`: bandwidth thực tế được sử dụng để phân cụm.

Thực nghiệm khảo sát 15 giá trị alpha được tạo bằng `np.geomspace(0.1, 3.0, 15)`, đồng thời bổ sung alpha = 1 để đánh dấu cấu hình bandwidth cơ sở. Do đó, mỗi bộ dữ liệu và seed được chạy với 16 giá trị alpha.

Các cấu hình chính:

| Tham số | Giá trị |
|---|---|
| Số bộ dữ liệu | 5 |
| Số seed mỗi bộ dữ liệu | 10 |
| Số giá trị alpha | 16 |
| Khoảng alpha | Từ 0,1 đến 3,0 |
| Phương pháp chọn bandwidth | `h = alpha * h0` |
| `h0` | `estimate_bandwidth(X, quantile=0.3)`, giống E2 |
| Mean Shift | `MeanShiftCustom`, kernel phẳng, tìm láng giềng bằng KD-Tree |
| `bin_seeding` | `True` |
| Số lần chạy để đo thời gian cho mỗi cấu hình | 3 |
| Thống kê thời gian | Trung vị của 3 lần chạy |

Tổng số cấu hình thực nghiệm là 5 × 10 × 16 = 800. Với mỗi cấu hình, thuật toán được chạy 3 lần để đo thời gian thực thi.

## 4. Chỉ số đánh giá

Thực nghiệm sử dụng các chỉ số sau:

- **Số cụm:** Cho biết thuật toán phát hiện được bao nhiêu cụm, không tính các điểm được gán nhãn nhiễu `-1` nếu thuật toán có sử dụng nhãn này.
- **ARI (Adjusted Rand Index):** Đánh giá mức độ tương đồng giữa kết quả phân cụm và nhãn tham chiếu. Giá trị càng cao thường cho thấy kết quả càng phù hợp với nhãn tham chiếu.
- **NMI (Normalized Mutual Information):** Đánh giá mức độ tương đồng giữa hai cách phân chia dữ liệu. Giá trị càng cao thể hiện sự tương đồng càng lớn.
- **Thời gian thực thi:** Được tính bằng giây và lấy trung vị của 3 lần chạy cho mỗi cấu hình.

Các kết quả được tổng hợp theo từng bộ dữ liệu và giá trị alpha. Báo cáo sử dụng giá trị trung bình và độ lệch chuẩn qua 10 seed để đánh giá kết quả cũng như mức độ biến động giữa các lần sinh dữ liệu.

## 5. Kết quả thực nghiệm

### 5.1. Ảnh hưởng của alpha đến kết quả phân cụm

Biểu đồ `e4_synthetic.png` gồm 2 hình: bên trái là số cụm theo alpha (trục y thang log, đường chấm là số cụm thật), bên phải là ARI theo alpha. Mỗi đường là trung bình qua 10 seed, dải màu là ± 1 độ lệch chuẩn. Đường đứt nét đen đánh dấu alpha = 1 (h mặc định).

Hình bên trái cho thấy xu hướng chung trên cả 5 bộ: alpha nhỏ thì số cụm tăng vọt (ví dụ blobs_varied ra khoảng 270 cụm ở alpha = 0,1), alpha lớn thì các cụm bị gộp dần về 1 cụm. Số cụm cắt đường số cụm thật ở alpha nhỏ hơn 1 với blobs, blobs_varied và blobs_aniso; với cả ba bộ này, tại alpha = 1 thuật toán chỉ còn trung bình 2,3 cụm.

### 5.2. Alpha tốt nhất và khoảng alpha an toàn

Bảng dưới lấy từ `results/e4/e4_synthetic_alpha_summary.csv`. "Alpha tốt nhất" là alpha có ARI trung bình (qua 10 seed) cao nhất. "Khoảng an toàn" gồm các alpha có ARI trung bình ≥ 90% ARI tốt nhất của bộ đó: khoảng càng rộng thì thuật toán càng ít nhạy với việc chọn h.

| Bộ dữ liệu | Số cụm thật | Alpha tốt nhất | ARI tốt nhất (số cụm) | ARI tại alpha = 1 (số cụm) | Chênh lệch ARI | Khoảng an toàn | alpha = 1 nằm trong? |
|---|---:|---:|---:|---:|---:|---|:-:|
| blobs | 4 | 0,26 | 0,962 (4,1) | 0,442 (2,3) | +0,520 | 0,16 – 0,43 | Không |
| blobs_varied | 3 | 0,70 | 0,712 (3,5) | 0,627 (2,3) | +0,085 | 0,43 – 0,89 | Không |
| blobs_aniso | 3 | 0,55 | 0,741 (2,9) | 0,640 (2,3) | +0,101 | 0,43 – 0,70 | Không |
| moons | 2 | 1,00 | 0,546 (2,0) | 0,546 (2,0) | 0 | 0,70 – 1,14 | Có |
| circles | 2 | 0,55 | 0,132 (8,8) | 0,000 (1,0) | +0,132 | (không có ý nghĩa) | – |

Alpha tốt nhất được chọn bằng chính nhãn thật trên cùng 10 seed dùng để báo cáo, nên cột "ARI tốt nhất" là **cận trên (oracle)**: đó là mức tốt nhất có thể đạt nếu biết trước h đúng, không phải kết quả dùng được khi không có nhãn.

## 6. Nhận xét

**h mặc định thường quá lớn.** Ở 4/5 bộ dữ liệu, alpha = 1 nằm ngoài khoảng an toàn, và alpha tốt nhất luôn ≤ 1 (từ 0,26 đến 1,00). Nghĩa là `estimate_bandwidth(quantile=0.3)` có xu hướng ước lượng h lớn hơn mức cần thiết, khiến các cụm gần nhau bị gộp lại: tại alpha = 1, blobs chỉ ra trung bình 2,3 cụm so với 4 cụm thật, và ARI chỉ còn 0,442, chưa bằng một nửa mức 0,962 đạt được khi alpha = 0,26. Kết quả này khớp với E2 (ARI 0,455 với h mặc định trên blobs).

**Độ nhạy khác nhau theo dạng dữ liệu.**
- blobs có khoảng an toàn rộng nhất (0,16 – 0,43, gồm 5 mức alpha): khi các cụm tách bạch và cùng kích thước, có nhiều giá trị h cho kết quả tốt, chỉ cần h đủ nhỏ để không gộp cụm.
- blobs_aniso hẹp nhất (0,43 – 0,70, chỉ 3 mức) và có độ lệch chuẩn giữa các seed rất lớn (dải xanh lá trên hình): cụm bị kéo dài nên một h đẳng hướng duy nhất khó vừa với mọi hướng.
- blobs_varied cũng khó vì các cụm có độ trải khác nhau: h nhỏ thì cụm rộng bị vỡ (khoảng 270 cụm ở alpha = 0,1), h lớn thì cụm hẹp bị gộp. ARI tốt nhất chỉ 0,712.
- moons là bộ duy nhất mà h mặc định trùng alpha tốt nhất, nhưng ARI cao nhất cũng chỉ 0,546, do Mean-Shift tìm mode mật độ nên không tách được hai hình lưỡi liềm lồng vào nhau.
- circles thất bại ở mọi h: ARI không vượt quá 0,132, và mức đó đạt được chỉ vì vòng tròn bị cắt thành khoảng 9 mảnh. Từ alpha ≈ 0,7, ARI đã về 0, và từ alpha ≈ 0,9 trở lên toàn bộ dữ liệu gộp thành 1 cụm. Đây là giới hạn của Mean-Shift với cụm không lồi, không phải lỗi cài đặt.

**Thời gian** giảm khi alpha tăng (blobs: khoảng 0,047 s ở alpha = 0,1 xuống khoảng 0,002 s ở alpha = 3), vì h lớn hơn thì lưới bin seeding thô hơn, số seed ít hơn. Với n = 1500 điểm, mọi cấu hình đều chạy dưới 0,06 s nên thời gian không phải yếu tố quyết định khi chọn h.

**Trả lời RQ4.** Mean-Shift khá nhạy với h: trên các bộ dạng cụm, khoảng alpha cho kết quả tốt chỉ rộng khoảng 1,6 – 2,7 lần (ví dụ 0,43 – 0,70), và h mặc định do `estimate_bandwidth(quantile=0.3)` đưa ra thường rơi ra ngoài khoảng này theo hướng quá lớn. Khi dùng Mean-Shift trên dữ liệu thật, nên coi h mặc định là điểm xuất phát và thử thêm các h nhỏ hơn (khoảng 0,3 – 0,7 lần h0), chọn trên tập riêng có nhãn hoặc dựa vào số cụm hợp lý.

## 7. Giới hạn thực nghiệm

- Alpha tốt nhất và khoảng an toàn được xác định bằng nhãn thật trên chính 10 seed đã báo cáo, nên ARI tốt nhất là cận trên. Ở dữ liệu thật không có nhãn, không thể chọn h theo cách này.
- Chỉ khảo sát alpha từ 0,1 đến 3,0 với 16 mức, trên 5 bộ dữ liệu 2 chiều, 1500 điểm. Ngưỡng 90% cho khoảng an toàn là quy ước của nhóm.
- Khoảng an toàn được báo cáo bằng alpha nhỏ nhất và lớn nhất thỏa ngưỡng; với cả 5 bộ, các alpha này liền nhau trên lưới.
- h0 phụ thuộc tham số `quantile` của `estimate_bandwidth`. Nhận xét "h mặc định quá lớn" chỉ đúng với quantile = 0.3 đã dùng (giống E2).

## 8. Tái lập thực nghiệm

Chạy đầy đủ 10 seed bằng lệnh:

`python3 experiments/e4_bandwidth.py`

Chạy thử với một seed để kiểm tra nhanh:

`python3 experiments/e4_bandwidth.py --seeds 1`

## 9. Đầu ra

Kết quả được lưu trong thư mục `results/e4/`:

| Tệp | Nội dung |
|---|---|
| `e4_synthetic_runs.csv` | Kết quả chi tiết của từng (bộ, seed, alpha); không lên git |
| `e4_synthetic_summary.csv` | Trung bình ± độ lệch chuẩn theo bộ dữ liệu và alpha |
| `e4_synthetic_alpha_summary.csv` | Theo từng bộ: alpha tốt nhất, ARI tại alpha = 1, khoảng alpha an toàn |
| `e4_synthetic.png` | Biểu đồ số cụm và ARI theo alpha |
| `e4_metadata.json` | Cấu hình, phiên bản thư viện, SHA256 dữ liệu đầu vào |

## 10. Phần 2: ISIC tune (đặc trưng 3D Lab)

### 10.1. Cấu hình

| Tham số | Giá trị |
|---|---|
| Ảnh | 100 ảnh tune (`load_split("tune")`, lấy từ ISIC 2018 Validation), thu nhỏ cạnh dài về 128 px |
| Đặc trưng | `features(rgb, "lab3")`: (L, a, b), không chuẩn hoá, nên h tính theo đơn vị Lab |
| Mean Shift | `MeanShiftCustom`, kernel phẳng, KD-Tree, `bin_seeding=True` |
| h cố định | 2, 4, 6, 8, 10, 12, 16, 20 (đơn vị Lab) |
| h tự động | `estimate_bandwidth(X, quantile=0.3, n_samples=500, random_state=0)` riêng cho từng ảnh |
| Chọn mask | `isic_utils.labels_to_mask` (quy tắc chung với E5, E6) |
| Chỉ số | Dice, IoU so với mask bác sĩ; số cụm; tỷ lệ mask rỗng; thời gian (trung vị 3 lần fit) |

Tổng cộng 100 ảnh × 9 mức h = 900 lần chạy.

### 10.2. Kết quả

Bảng lấy từ `results/e4/e4_isic_summary.csv` (trung bình qua 100 ảnh).

| h | Dice TB ± std | IoU TB | Số cụm (trung vị) | Mask rỗng | Ảnh Dice < 0,1 | Thời gian TB (s) |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0,049 ± 0,096 | 0,028 | 126 | 1% | 84 | 1,11 |
| 4 | 0,309 ± 0,293 | 0,226 | 14,5 | 1% | 34 | 1,39 |
| 6 | 0,449 ± 0,334 | 0,356 | 6 | 6% | 23 | 1,02 |
| **8** | **0,567 ± 0,358** | **0,480** | 3 | 10% | **19** | 0,66 |
| 10 | 0,543 ± 0,385 | 0,467 | 2 | 20% | 26 | 0,45 |
| 12 | 0,458 ± 0,421 | 0,401 | 2 | 36% | 42 | 0,32 |
| 16 | 0,354 ± 0,430 | 0,318 | 1 | 54% | 57 | 0,19 |
| 20 | 0,259 ± 0,403 | 0,235 | 1 | 67% | 69 | 0,13 |
| tự động (TB 10,9) | 0,559 ± 0,390 | 0,487 | 2 | 19% | 26 | 0,44 |

**h tốt nhất = 8** (Dice 0,567, trung vị 0,662), ghi vào `results/e4/best_h_3d_lab.json` để dùng cho E5.

Hình `e4_isic_h.png`: Dice/IoU, số cụm và thời gian theo h. Hình `e4_isic_examples.png`: ảnh có Dice trung vị và ảnh tệ nhất tại h = 8, mỗi ảnh ở 4 mức h.

### 10.3. Nhận xét

**Dice có dạng hình chuông theo h, đỉnh ở h = 8–10.** Hai phía đỉnh hỏng theo hai kiểu ngược nhau:
- **h nhỏ (2–4): tổn thương bị vỡ vụn.** Ở h = 2, mỗi ảnh ra trung vị 126 cụm; `labels_to_mask` chỉ chọn một cụm tối nhất nên mask chỉ là một mảnh nhỏ của tổn thương (Dice 0,049, 84/100 ảnh Dice < 0,1).
- **h lớn (≥ 12): tổn thương bị gộp vào da.** Số cụm về 1–2, tỷ lệ mask rỗng tăng từ 10% (h = 8) lên 67% (h = 20), vì khi chỉ còn 1 cụm phủ cả viền ảnh thì không còn cụm ứng viên nào.

Khoảng h cho Dice ≥ 90% mức tốt nhất (≥ 0,510) chỉ gồm h = 8 và 10, tức Mean-Shift trên ảnh thật cũng nhạy với h như trên dữ liệu giả lập (phần 1).

**h tự động gần bằng h tốt nhất.** `estimate_bandwidth` cho h trung bình 10,9 (từ 3,0 đến 21,5 tùy ảnh) và Dice 0,559, chỉ thấp hơn h = 8 là 0,008; tính theo từng ảnh, h tự động tốt hơn h = 8 ở 49 ảnh, kém hơn ở 36 ảnh. Khác với phần 1 (h mặc định quá lớn trên dữ liệu chuẩn hoá), ở đây không gian Lab có đơn vị cố định và mỗi ảnh chủ yếu gồm 2 vùng màu (da, tổn thương), nên quantile = 0,3 cho h vừa phải. Nhóm vẫn chọn h = 8 cố định cho E5 vì Dice cao nhất, ít ảnh hỏng nhất (19 so với 26) và không phụ thuộc vào việc lấy mẫu của `estimate_bandwidth`.

**Một h chung cho mọi ảnh còn xa mức tối ưu từng ảnh.** Nếu chọn được h tốt nhất riêng cho từng ảnh (cận trên, dùng mask thật), Dice trung bình đạt 0,747 so với 0,567. h tốt nhất của từng ảnh trải đều từ 4 đến 20: ví dụ ảnh trung vị trong `e4_isic_examples.png` đạt Dice 0,67 ở h = 8 nhưng 0,86 ở h = 20, vì h = 8 tách lõi đen và viền nâu của nốt ruồi thành 2 cụm và chỉ giữ lõi.

**Ảnh tương phản thấp thất bại ở mọi h.** Ảnh tệ nhất (ISIC_0012346) có tổn thương nhạt gần màu da: từ h = 8 trở lên cả ảnh chỉ còn 1 cụm. 5/100 ảnh có Dice < 0,1 ở mọi mức h; đây là giới hạn của phân cụm chỉ dựa vào màu.

**Thời gian** giảm khi h tăng từ 4 (1,39 s/ảnh) xuống 20 (0,13 s/ảnh), vì ít seed hơn sau bin seeding. h = 2 lại nhanh hơn h = 4 dù nhiều cụm hơn; nhiều khả năng vì mỗi seed chỉ có rất ít láng giềng nên mỗi vòng lặp rẻ (nhóm chưa đo riêng để kiểm chứng). Với h = 8, một ảnh 128 px mất khoảng 0,66 s.

**Trả lời RQ4 (phần ảnh thật).** Dice của Mean-Shift 3D Lab thay đổi từ 0,05 đến 0,57 theo h; chỉ h = 8–10 cho kết quả tốt. h = 8 được chuyển sang E5, kết quả cuối cùng báo cáo trên 200 ảnh test.

### 10.4. Giới hạn

- h được chọn và Dice được đo trên cùng 100 ảnh tune, nên Dice 0,567 có thể cao hơn mức đạt được trên tập test; E5 báo cáo trên 200 ảnh test không dùng để chọn h.
- Chỉ xét đặc trưng 3D Lab, ảnh thu nhỏ 128 px; không xét 5D (Lab + vị trí). Lưới h thưa ở đoạn 12–20.
- Dice phụ thuộc quy tắc `labels_to_mask` (chọn một cụm tối nhất); một phương pháp gộp nhiều cụm có thể cho h tốt nhất khác.
- `estimate_bandwidth` chỉ lấy mẫu 500 pixel mỗi ảnh nên h tự động có chút ngẫu nhiên (cố định bằng `random_state=0`).
- Thời gian đo trên Apple M1, chạy 1 tiến trình.

### 10.5. Tái lập và đầu ra

```
python3 data/download_isic.py          # tải ảnh (1 lần)
python3 experiments/e4_isic.py         # 100 ảnh, ~25 phút trên Apple M1
python3 experiments/e4_isic.py --quick # 10 ảnh, ghi ra thư mục tạm, không đụng results/e4/
```

| Tệp (results/e4/) | Nội dung |
|---|---|
| `e4_isic_runs.csv` | Mỗi (ảnh, h): Dice, IoU, số cụm, mask rỗng, thời gian; không lên git |
| `e4_isic_summary.csv` | Trung bình ± độ lệch chuẩn theo h |
| `e4_isic_h.png` | Dice/IoU, số cụm, thời gian theo h |
| `e4_isic_examples.png` | Ảnh trung vị và ảnh tệ nhất ở 4 mức h |
| `best_h_3d_lab.json` | h tốt nhất cho 3D Lab (dùng ở E5) |
| `e4_isic_metadata.json` | Cấu hình, phiên bản thư viện, SHA256 file chia tập |
