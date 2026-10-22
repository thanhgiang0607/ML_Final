# Thực nghiệm E3 — Tốc độ và bộ nhớ

## 1. Mục tiêu
Trả lời RQ3: KD-Tree và bin seeding giảm chi phí tính toán bao nhiêu, và trong điều kiện nào?

## 2. Cấu hình thực nghiệm
- Dữ liệu: `make_blobs`, centers=5, cluster_std=1.0, d=2, random_state=0.
- Bandwidth tham chiếu: h0=1.5541824, ước lượng một lần trên mẫu n=2000 điểm với quantile=0.1.
- Quét n: 500, 1000, 2000, 5000, 10000, 20000, 50000; giữ h=h0.
- Quét h: n=5000; alpha=0.25, 0.5, 1.0, 2.0; h=alpha*h0.
- Số lượt đo thời gian mỗi cấu hình: 3; timeout: 600 giây cho từng lượt fit.
- Các biến thể: Custom brute; Custom KD-Tree; Custom KD-Tree + bin seeding; sklearn MeanShift (bin seeding off/on, n_jobs=1).
- **Thời gian:** `time.perf_counter()` quanh `fit(X)`, lặp N lần, không bật `tracemalloc` và không gắn wrapper thống kê; báo cáo median và IQR.
- **Bộ nhớ:** một lượt `fit(X)` riêng cho mỗi cấu hình, bật `tracemalloc` ngay trước fit và không gắn wrapper thống kê. IQR bộ nhớ không báo cáo khi chỉ có một quan sát.
- **Chẩn đoán:** một lượt fit riêng không bật `tracemalloc`; chỉ lượt này gắn wrapper để đếm truy vấn/láng giềng và số vòng lặp. Thời gian của lượt chẩn đoán không dùng trong benchmark.
- Môi trường: CPU Intel64 Family 6 Model 186 Stepping 3, GenuineIntel; RAM tổng (GB) 15.692; Python 3.13.6 (tags/v3.13.6:4e66535, Aug  6 2025, 14:36:00) [MSC v.1944 64 bit (AMD64)]; NumPy 2.5.4; SciPy 1.18.1; scikit-learn 1.9.1.

## 3. Kết quả tóm tắt
Bảng chi tiết nằm ở `../results/e3/e3_summary.csv`; từng phép đo nằm ở `../results/e3/e3_runs.csv` (không commit theo quy ước nhóm).

### 3.1. Thời gian, bộ nhớ và số cụm theo n
Bảng có cột **Số cụm trung vị** để kiểm tra bandwidth có đang gộp quá nhiều cụm hay không.

| n | Phương pháp | Median thời gian (s) | IQR thời gian (s) | Bộ nhớ đỉnh (MB) | Số cụm trung vị | Số seed trung vị | Láng giềng TB/truy vấn | Vòng lặp TB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 500 | Custom brute | 0.1077 | 0.04136 | 4.547 | 6 | 500 | 70.57 | 7.794 |
| 500 | Custom KD-Tree | 0.1394 | 0.05008 | 2.203 | 6 | 500 | 70.57 | 7.794 |
| 500 | Custom KD-Tree + bin seeding | 0.0914 | 0.007053 | 0.2264 | 5 | 48 | 65.33 | 8.771 |
| 500 | sklearn (bin seeding off) | 6.512 | 0.2692 | 0.8895 | 6 | 500 | 70.49 | 7.794 |
| 500 | sklearn (bin seeding on) | 0.6823 | 0.04841 | 0.7431 | 5 | 48 | 64.67 | 8.771 |
| 1000 | Custom brute | 0.6149 | 0.151 | 15.21 | 5 | 1000 | 152 | 9.584 |
| 1000 | Custom KD-Tree | 0.4546 | 0.003305 | 9.587 | 5 | 1000 | 152 | 9.584 |
| 1000 | Custom KD-Tree + bin seeding | 0.1451 | 0.0009871 | 0.5393 | 4 | 56 | 133.2 | 9.75 |
| 1000 | sklearn (bin seeding off) | 15.72 | 0.1178 | 1.03 | 5 | 1000 | 151.9 | 9.584 |
| 1000 | sklearn (bin seeding on) | 0.8725 | 0.007826 | 0.7664 | 4 | 56 | 132.4 | 9.75 |
| 2000 | Custom brute | 1.945 | 0.5414 | 38.99 | 6 | 2000 | 293.8 | 10.85 |
| 2000 | Custom KD-Tree | 1.667 | 0.202 | 39.83 | 6 | 2000 | 293.8 | 10.85 |
| 2000 | Custom KD-Tree + bin seeding | 0.2429 | 0.00657 | 1.186 | 6 | 60 | 265.9 | 11.92 |
| 2000 | sklearn (bin seeding off) | 35.71 | 0.1644 | 1.379 | 6 | 2000 | 293.7 | 10.85 |
| 2000 | sklearn (bin seeding on) | 1.134 | 0.01799 | 0.8176 | 6 | 60 | 263.9 | 11.92 |
| 5000 | Custom brute | 32.25 | 13.93 | 165.4 | 6 | 5000 | 756.1 | 16.64 |
| 5000 | Custom KD-Tree | 25.85 | 1.044 | 253.1 | 6 | 5000 | 756.1 | 16.64 |
| 5000 | Custom KD-Tree + bin seeding | 0.6322 | 0.007185 | 3.495 | 5 | 68 | 690.7 | 16.24 |
| 5000 | sklearn (bin seeding off) | 138.3 | 0.5371 | 2.505 | 6 | 5000 | 756 | 16.64 |
| 5000 | sklearn (bin seeding on) | 1.891 | 0.01222 | 0.9101 | 5 | 68 | 687.8 | 16.24 |
| 10000 | Custom brute | 62.14 | 0.7651 | 562.3 | 4 | 10000 | 1484 | 14.92 |
| 10000 | Custom KD-Tree | 86.88 | 0.4595 | — | 4 | 10000 | — | — |
| 10000 | Custom KD-Tree + bin seeding | 1.187 | 0.04865 | 7.497 | 5 | 73 | 1360 | 16.05 |
| 10000 | sklearn (bin seeding off) | 267.7 | 6.009 | — | 4 | 10000 | — | — |
| 10000 | sklearn (bin seeding on) | 2.252 | 0.06055 | 1.053 | 5 | 73 | 1354 | 16.05 |
| 20000 | Custom brute | 492.2 | 22.11 | — | 4 | 20000 | — | — |
| 20000 | Custom KD-Tree | — | — | — | — | — | — | — |
| 20000 | Custom KD-Tree + bin seeding | 2.183 | 0.03152 | 16.64 | 5 | 81 | 2654 | 14.51 |
| 20000 | sklearn (bin seeding off) | 524.3 | 1.883 | — | 4 | 20000 | — | — |
| 20000 | sklearn (bin seeding on) | 2.696 | 0.02596 | 1.338 | 5 | 81 | 2644 | 14.51 |
| 50000 | Custom brute | — | — | — | — | — | — | — |
| 50000 | Custom KD-Tree | — | — | — | — | — | — | — |
| 50000 | Custom KD-Tree + bin seeding | 5.462 | 0.003638 | 43.79 | 6 | 87 | 6496 | 13.89 |
| 50000 | sklearn (bin seeding off) | — | — | — | — | — | — | — |
| 50000 | sklearn (bin seeding on) | 4.062 | 0.0248 | 2.151 | 6 | 87 | 6466 | 13.89 |

### 3.2. Thời gian, bộ nhớ và số cụm theo h
| alpha | h | Phương pháp | Median thời gian (s) | IQR thời gian (s) | Bộ nhớ đỉnh (MB) | Số cụm trung vị | Số seed trung vị | Láng giềng TB/truy vấn | Vòng lặp TB |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.38855 | Custom brute | 16.37 | 0.02492 | — | 210 | 5000 | — | — |
| 0.25 | 0.38855 | Custom KD-Tree | 11.1 | 4.932 | 18.11 | 210 | 5000 | 55.14 | 10.75 |
| 0.25 | 0.38855 | Custom KD-Tree + bin seeding | 0.3582 | 0.1878 | 1.61 | 188 | 651 | 41.73 | 8.972 |
| 0.25 | 0.38855 | sklearn (bin seeding off) | 50.29 | 11.78 | 2.231 | 210 | 5000 | 54.97 | 10.75 |
| 0.25 | 0.38855 | sklearn (bin seeding on) | 2.432 | 1.049 | 1.034 | 188 | 651 | 40.63 | 8.972 |
| 0.5 | 0.77709 | Custom brute | — | — | — | — | — | — | — |
| 0.5 | 0.77709 | Custom KD-Tree | 6.846 | 11.72 | 82.49 | 12 | 5000 | 236.1 | 21.57 |
| 0.5 | 0.77709 | Custom KD-Tree + bin seeding | 0.3187 | 0.1215 | 3.358 | 11 | 204 | 215.3 | 24.8 |
| 0.5 | 0.77709 | sklearn (bin seeding off) | 102.4 | 7.496 | 2.635 | 12 | 5000 | 236.1 | 21.57 |
| 0.5 | 0.77709 | sklearn (bin seeding on) | 3.962 | 1.29 | 0.9708 | 11 | 204 | 214.9 | 24.8 |
| 1 | 1.5542 | Custom brute | — | — | — | — | — | — | — |
| 1 | 1.5542 | Custom KD-Tree | 14.93 | 2.485 | 253.1 | 6 | 5000 | 756.1 | 16.64 |
| 1 | 1.5542 | Custom KD-Tree + bin seeding | 0.1412 | 0.01979 | 3.495 | 5 | 68 | 690.7 | 16.24 |
| 1 | 1.5542 | sklearn (bin seeding off) | 85.43 | 1.387 | 2.503 | 6 | 5000 | 756 | 16.64 |
| 1 | 1.5542 | sklearn (bin seeding on) | 0.3843 | 0.09142 | 0.9094 | 5 | 68 | 687.8 | 16.24 |
| 2 | 3.1084 | Custom brute | — | — | — | — | — | — | — |
| 2 | 3.1084 | Custom KD-Tree | 39.35 | 7.413 | 605.4 | 2 | 5000 | 2132 | 14.85 |
| 2 | 3.1084 | Custom KD-Tree + bin seeding | 0.1084 | 0.06567 | 2.555 | 2 | 23 | 1989 | 14.3 |
| 2 | 3.1084 | sklearn (bin seeding off) | 80.67 | 0.8974 | 2.458 | 2 | 5000 | 2132 | 14.85 |
| 2 | 3.1084 | sklearn (bin seeding on) | 0.2059 | 0.03734 | 0.9072 | 2 | 23 | 1978 | 14.3 |

## 4. Trả lời định lượng các câu hỏi RQ3

### 4.1. KD-Tree bắt đầu nhanh hơn brute ở n nào?
Tốc độ tương đối được tính bằng `median_time(brute) / median_time(KD-Tree)`. Giá trị > 1 nghĩa là KD-Tree nhanh hơn; giá trị < 1 nghĩa là KD-Tree chậm hơn trong cấu hình và môi trường đã đo.

| n | Brute median (s) | KD-Tree median (s) | Brute/KD-Tree (×) | Láng giềng TB/truy vấn của KD-Tree |
|---:|---:|---:|---:|---:|
| 500 | 0.10772 | 0.13937 | 0.773 | 70.57 |
| 1000 | 0.61487 | 0.45459 | 1.353 | 152 |
| 2000 | 1.9449 | 1.6671 | 1.167 | 293.8 |
| 5000 | 32.249 | 25.852 | 1.247 | 756.1 |
| 10000 | 62.137 | 86.883 | 0.715 | — |
| 20000 | — | — | — | — |
| 50000 | — | — | — | — |

**Kết quả:** trong các mức n được so sánh hợp lệ, KD-Tree lần đầu nhanh hơn brute tại **n = 1,000**; nhưng lợi thế không duy trì ở mọi mức n lớn hơn; KD-Tree không nhanh hơn brute tại n=10,000. Đây là kết quả quan sát trên máy và bộ dữ liệu này, không phải ngưỡng phổ quát.

**Độ dốc log–log thực nghiệm** (hệ số góc của log10(thời gian median) theo log10(n), chỉ tính tại các điểm đo hợp lệ):
- Custom brute: độ dốc = **2.243**, dựa trên 6 mức n.
- Custom KD-Tree: độ dốc = **2.237**, dựa trên 5 mức n.
- Custom KD-Tree + bin seeding: độ dốc = **0.902**, dựa trên 7 mức n.
- sklearn (bin seeding off): độ dốc = **1.213**, dựa trên 6 mức n.
- sklearn (bin seeding on): độ dốc = **0.388**, dựa trên 7 mức n.
Độ dốc là mô tả thực nghiệm trong khoảng n đã đo; không nên đồng nhất trực tiếp với độ phức tạp tiệm cận lý thuyết.

### 4.2. Khi h tăng, số láng giềng và thời gian thay đổi thế nào?
Bảng sau so sánh mức alpha nhỏ nhất và lớn nhất có số liệu hợp lệ của từng phương pháp. Phần trăm thay đổi được tính từ điểm đầu đến điểm cuối; phần nhận xét về xu hướng dùng tất cả điểm alpha có dữ liệu.

| Phương pháp | alpha đầu → cuối | Láng giềng TB/truy vấn đầu → cuối | Thay đổi láng giềng | Thời gian median đầu → cuối (s) | Thay đổi thời gian | Xu hướng thời gian qua các alpha |
|---|---:|---:|---:|---:|---:|---|
| Custom brute | 0.25 → 0.25 | không thu được | không tính được | 16.374 → 16.374 | +0.0% | chưa đủ điểm hợp lệ để xác định xu hướng |
| Custom KD-Tree | 0.25 → 2 | 55.14 → 2132 | +3766.4% | 11.104 → 39.347 | +254.3% | biến động không đơn điệu qua các mức đã đo |
| Custom KD-Tree + bin seeding | 0.25 → 2 | 41.73 → 1989 | +4667.1% | 0.35816 → 0.10841 | -69.7% | giảm hoặc giữ nguyên qua các mức đã đo |
| sklearn (bin seeding off) | 0.25 → 2 | 54.97 → 2132 | +3778.3% | 50.294 → 80.666 | +60.4% | biến động không đơn điệu qua các mức đã đo |
| sklearn (bin seeding on) | 0.25 → 2 | 40.63 → 1978 | +4768.8% | 2.4316 → 0.2059 | -91.5% | biến động không đơn điệu qua các mức đã đo |

**Diễn giải KD-Tree:** số láng giềng trung bình tăng hoặc giữ nguyên qua các mức đã đo theo các alpha đã đo; từ alpha=0.25 đến alpha=2, giá trị đổi từ 55.14 sang 2132 (+3766.4%).
Thời gian median của KD-Tree biến động không đơn điệu qua các mức đã đo theo alpha; từ alpha=0.25 đến alpha=2, thời gian đổi từ 11.104 s sang 39.347 s (+254.3%).
Số láng giềng lớn hơn có thể làm giảm lợi thế của KD-Tree vì truy vấn cần trả về nhiều điểm hơn; chỉ kết luận cơ chế này phù hợp với kết quả nếu số láng giềng thực đo và thời gian cùng ủng hộ nhận xét, không suy ra quan hệ nhân quả chỉ từ lý thuyết.

### 4.3. Bin seeding giảm bao nhiêu seed và thời gian?
So sánh `custom_kdtree_bin` với `custom_kdtree` tại cùng n và h. Phần trăm tiết kiệm seed = (seed không bin − seed có bin) / seed không bin; phần trăm tiết kiệm thời gian được tính tương tự. Số âm nghĩa là cấu hình có bin seeding tốn nhiều tài nguyên/thời gian hơn ở phép đo đó.

| n | Seed không bin | Seed có bin | Seed giảm | Seed giảm (%) | Số cụm không bin | Số cụm có bin | Thời gian không bin (s) | Thời gian có bin (s) | Thời gian tiết kiệm (s) | Thời gian tiết kiệm (%) | Speedup có bin (×) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 500 | 48 | 452 | +90.4% | 6 | 5 | 0.13937 | 0.0914 | +0.047972 | +34.4% | 1.525 |
| 1000 | 1000 | 56 | 944 | +94.4% | 5 | 4 | 0.45459 | 0.14512 | +0.30946 | +68.1% | 3.132 |
| 2000 | 2000 | 60 | 1940 | +97.0% | 6 | 6 | 1.6671 | 0.24293 | +1.4242 | +85.4% | 6.862 |
| 5000 | 5000 | 68 | 4932 | +98.6% | 6 | 5 | 25.852 | 0.63222 | +25.219 | +97.6% | 40.890 |
| 10000 | 10000 | 73 | 9927 | +99.3% | 4 | 5 | 86.883 | 1.1869 | +85.696 | +98.6% | 73.202 |
| 20000 | — | 81 | — | — | — | 5 | — | 2.183 | — | — | — |
| 50000 | — | 87 | — | — | — | 6 | — | 5.462 | — | — | — |

Mức giảm seed lớn nhất theo tỷ lệ là +99.3% tại n=10,000 (từ 10000 xuống 73 seed).
Mức tiết kiệm thời gian lớn nhất theo tỷ lệ là +98.6% tại n=10,000 (+85.696 s; speedup=73.202×). Số âm nghĩa là bin seeding chậm hơn KD-Tree không bin ở cấu hình đó.
Cần đọc đồng thời cột số cụm: giảm seed/tăng tốc không tự động có nghĩa là kết quả phân cụm giữ nguyên. Bảng E3 ghi số cụm để đối chiếu; đánh giá sai khác chất lượng phân cụm chi tiết thuộc các thực nghiệm phù hợp như E1/E6.

## 5. Biểu đồ
- `../results/e3/e3_time_vs_n.png` — thời gian theo n, trục log–log.
- `../results/e3/e3_memory_vs_n.png` — bộ nhớ đỉnh theo n.
- `../results/e3/e3_time_vs_h.png` — thời gian theo h.

## 6. Giới hạn và lưu ý diễn giải
- `tracemalloc` ghi nhận bộ nhớ Python-traced; có thể không bao quát đầy đủ bộ nhớ native do NumPy, SciPy, KD-Tree hoặc sklearn cấp phát. Không diễn giải số đo này như RSS tổng của tiến trình.
- Thời gian báo cáo chỉ lấy từ các lượt fit không bật tracemalloc và không gắn wrapper. Lượt đo bộ nhớ và lượt chẩn đoán là các lần fit riêng; thời gian của chúng không tham gia tính median thời gian.
- Bộ nhớ được đo một lần riêng cho mỗi cấu hình. Do chỉ có một quan sát, median được báo cáo nhưng Q1/Q3/IQR bộ nhớ để trống.
- Các lượt timeout/skip được ghi trong CSV; chúng không được dùng để tính median thời gian thành công.
- Số láng giềng và vòng lặp lấy từ lượt chẩn đoán. Nếu API thư viện thay đổi khiến không thu được số liệu, để trống/NA thay vì tự suy ra.
- Số cụm được lấy từ `len(cluster_centers_)` sau mỗi lượt fit; cần đọc cột này cùng bandwidth h để xác định liệu các cụm có bị gộp quá mức hay không.

## 7. File đầu ra
- `../results/e3/e3_runs.csv` — dữ liệu từng phép đo, có cột `measurement` (`time`, `memory`, `diagnostics`).
- `../results/e3/e3_summary.csv` — median/Q1/Q3/IQR cho thời gian, peak memory khi có đủ quan sát, speedup so với custom brute và số cụm trung vị.
- `../results/e3/env.txt` và `../results/e3/e3_metadata.json` — môi trường và cấu hình chạy.
