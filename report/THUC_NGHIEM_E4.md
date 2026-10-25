# THỰC NGHIỆM E4: KHẢO SÁT ĐỘ NHẠY CỦA MEAN SHIFT THEO BANDWIDTH

## 1. Mục tiêu

Thực nghiệm E4 khảo sát ảnh hưởng của tham số bandwidth đến kết quả phân cụm của thuật toán Mean Shift. Mục tiêu là đánh giá sự thay đổi về số lượng cụm, chất lượng phân cụm và thời gian thực thi khi bandwidth thay đổi, từ đó xác định mức bandwidth phù hợp cho từng bộ dữ liệu.

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
