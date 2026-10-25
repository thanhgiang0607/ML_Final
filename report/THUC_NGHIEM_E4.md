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

Biểu đồ `e4_synthetic.png` thể hiện sự thay đổi của số cụm và ARI theo alpha trên cả 5 bộ dữ liệu. Đường đánh dấu alpha = 1 thể hiện cấu hình sử dụng bandwidth cơ sở.

Nhìn chung, bandwidth ảnh hưởng trực tiếp đến kết quả phân cụm. Khi bandwidth nhỏ, thuật toán có thể phát hiện nhiều cụm nhỏ; khi bandwidth tăng, các cụm gần nhau có thể bị gộp lại. Vì vậy, lựa chọn bandwidth cần cân nhắc đặc điểm phân bố của từng bộ dữ liệu.

### 5.2. Alpha cho ARI trung bình cao nhất theo từng bộ dữ liệu

Theo kết quả tổng hợp của lần chạy đủ 10 seed, các giá trị alpha có ARI trung bình cao nhất theo từng bộ dữ liệu như sau:

| Bộ dữ liệu | Alpha tốt nhất | ARI trung bình | Số cụm trung bình |
|---|---:|---:|---:|
| blobs | 0,264262 | 0,962113 | 4,1 |
| blobs_aniso | 0,547723 | 0,741091 | 2,9 |
| blobs_varied | 0,698344 | 0,711799 | 3,5 |
| circles | 0,547723 | 0,131864 | 8,8 |
| moons | 1,000000 | 0,546367 | 2,0 |

Kết quả cho thấy không có một giá trị alpha duy nhất đạt kết quả tốt nhất trên tất cả các bộ dữ liệu. Điều này cho thấy việc lựa chọn bandwidth cần xét đến cấu trúc và đặc điểm phân bố của dữ liệu.

Lưu ý: Cần đối chiếu các giá trị trong bảng với file `results/e4/e4_synthetic_summary.csv` trước khi chốt báo cáo.

### 5.3. So sánh bandwidth cơ sở với giá trị tốt nhất

Bandwidth cơ sở tương ứng với alpha = 1. So sánh ARI trung bình tại alpha = 1 với ARI trung bình cao nhất của từng bộ dữ liệu cho kết quả:

| Bộ dữ liệu | ARI tại alpha = 1 | ARI tốt nhất | Chênh lệch ARI |
|---|---:|---:|---:|
| blobs | 0,442359 | 0,962113 | +0,519754 |
| blobs_varied | 0,627023 | 0,711799 | +0,084776 |
| blobs_aniso | 0,639829 | 0,741091 | +0,101263 |
| moons | 0,546367 | 0,546367 | 0,000000 |
| circles | 0,000000 | 0,131864 | +0,131864 |

Các kết quả cho thấy việc điều chỉnh bandwidth có thể cải thiện chất lượng phân cụm so với bandwidth cơ sở. Mức cải thiện khác nhau giữa các bộ dữ liệu, cho thấy hiệu quả của việc điều chỉnh phụ thuộc vào đặc điểm dữ liệu.


## 6. Nhận xét

Kết quả thực nghiệm cho thấy bandwidth ảnh hưởng đáng kể đến chất lượng phân cụm của Mean Shift. Giá trị alpha có ARI trung bình cao nhất khác nhau giữa các bộ dữ liệu, cho thấy không có một cấu hình bandwidth duy nhất phù hợp với mọi dạng phân bố.

Đối với blobs, việc điều chỉnh bandwidth giúp ARI trung bình tăng từ 0,442359 lên 0,962113. Với blobs_varied và blobs_aniso, mức cải thiện lần lượt là 0,084776 và 0,101263. Trong khi đó, moons đạt ARI trung bình cao nhất tại alpha = 1, nên việc điều chỉnh bandwidth trong phạm vi khảo sát không giúp cải thiện chỉ số này. Với circles, ARI tốt nhất vẫn thấp, cho thấy cấu hình được khảo sát chưa tạo ra kết quả phân cụm tốt theo nhãn tham chiếu.

Nhìn chung, bandwidth cơ sở có thể được dùng làm điểm xuất phát, nhưng nên khảo sát thêm các giá trị khác khi áp dụng Mean Shift. Việc lựa chọn tham số cần dựa trên đặc điểm dữ liệu và nhiều chỉ số đánh giá, thay vì chỉ dựa vào số lượng cụm.

## 7. Giới hạn thực nghiệm

Thực nghiệm chỉ khảo sát alpha trong khoảng từ 0,1 đến 3,0 trên năm bộ dữ liệu tổng hợp. Giá trị tốt nhất được báo cáo là giá trị có ARI trung bình cao nhất trong phạm vi đã thử nghiệm, không đồng nghĩa với giá trị tối ưu cho mọi bộ dữ liệu.

Ngoài ra, nhãn tham chiếu chỉ có sẵn vì đây là dữ liệu tổng hợp. Với dữ liệu thực tế không có nhãn, việc lựa chọn bandwidth cần kết hợp thêm các tiêu chí đánh giá khác.

## 8. Tái lập thực nghiệm

Chạy đầy đủ 10 seed bằng lệnh:

`python experiments/e4_bandwidth.py`

Chạy thử với một seed để kiểm tra nhanh:

`python experiments/e4_bandwidth.py --seeds 1`

## 9. Đầu ra

Kết quả được lưu trong thư mục `results/e4/`:

| Tệp | Nội dung |
|---|---|
| `e4_synthetic_runs.csv` | Kết quả chi tiết của từng cấu hình |
| `e4_synthetic_summary.csv` | Kết quả tổng hợp theo bộ dữ liệu và alpha |
| `e4_synthetic.png` | Biểu đồ số cụm và ARI theo alpha |

Các bảng số liệu trong báo cáo cần được đối chiếu với những tệp đầu ra tương ứng để bảo đảm thống nhất với lần chạy thực nghiệm cuối cùng.