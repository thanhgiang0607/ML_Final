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
| 500 | Custom brute | 0.2018 | 0.8461 | 4.547 | 6 | 500 | 70.57 | 7.794 |
| 500 | Custom KD-Tree | 0.2072 | 0.01274 | 2.203 | 6 | 500 | 70.57 | 7.794 |
| 500 | Custom KD-Tree + bin seeding | 0.01241 | 0.003355 | 0.2264 | 5 | 48 | 65.33 | 8.771 |
| 500 | sklearn (bin seeding off) | 1.138 | 0.1493 | 0.8895 | 6 | 500 | 70.49 | 22 |
| 500 | sklearn (bin seeding on) | 0.1638 | 0.05159 | 0.7485 | 5 | 48 | 64.67 | 18 |
| 1000 | Custom brute | 0.5732 | 0.03641 | 15.21 | 5 | 1e+03 | 152 | 9.584 |
| 1000 | Custom KD-Tree | 0.6361 | 0.02916 | 9.587 | 5 | 1e+03 | 152 | 9.584 |
| 1000 | Custom KD-Tree + bin seeding | 0.02924 | 0.01289 | 0.5393 | 4 | 56 | 133.2 | 9.75 |
| 1000 | sklearn (bin seeding off) | 3.583 | 0.1637 | 1.031 | 5 | 1e+03 | 151.9 | 36 |
| 1000 | sklearn (bin seeding on) | 0.2583 | 0.0204 | 0.7671 | 4 | 56 | 132.4 | 18 |
| 2000 | Custom brute | 2.058 | 0.121 | 38.99 | 6 | 2e+03 | 293.8 | 10.85 |
| 2000 | Custom KD-Tree | 2.222 | 0.02929 | 39.83 | 6 | 2e+03 | 293.8 | 10.85 |
| 2000 | Custom KD-Tree + bin seeding | 0.03636 | 0.007926 | 1.186 | 6 | 60 | 265.9 | 11.92 |
| 2000 | sklearn (bin seeding off) | 7.947 | 0.672 | 1.381 | 6 | 2e+03 | 293.7 | 38 |
| 2000 | sklearn (bin seeding on) | 0.5018 | 0.1656 | 0.8115 | 6 | 60 | 263.9 | 19 |
| 5000 | Custom brute | 14.51 | 0.9822 | 165.4 | 6 | 5e+03 | 756.1 | 16.64 |
| 5000 | Custom KD-Tree | 17.95 | 5.693 | 253.1 | 6 | 5e+03 | 756.1 | 16.64 |
| 5000 | Custom KD-Tree + bin seeding | 0.1115 | 0.006944 | 3.495 | 5 | 68 | 690.7 | 16.24 |
| 5000 | sklearn (bin seeding off) | 28.07 | 0.7794 | 2.502 | 6 | 5e+03 | 756 | 84 |
| 5000 | sklearn (bin seeding on) | 0.561 | 0.1455 | 0.9095 | 5 | 68 | 687.8 | 41 |
| 10000 | Custom brute | 92.15 | 13.49 | 562.3 | 4 | 1e+04 | 1484 | 14.92 |
| 10000 | Custom KD-Tree | 40.42 | 0.2291 | 1019 | 4 | 1e+04 | 1484 | 14.92 |
| 10000 | Custom KD-Tree + bin seeding | 0.3125 | 0.06557 | 7.497 | 5 | 73 | 1360 | 16.05 |
| 10000 | sklearn (bin seeding off) | 131.6 | 39.74 | 4.131 | 4 | 1e+04 | 1484 | 59 |
| 10000 | sklearn (bin seeding on) | 0.9963 | 0.1935 | 1.039 | 5 | 73 | 1354 | 37 |
| 20000 | Custom brute | — | — | — | — | — | — | — |
| 20000 | Custom KD-Tree | 445.3 | 41.91 | — | 4 | 2e+04 | — | — |
| 20000 | Custom KD-Tree + bin seeding | 0.4974 | 0.06118 | 16.64 | 5 | 81 | 2654 | 14.51 |
| 20000 | sklearn (bin seeding off) | 255.9 | 17.01 | 7.321 | 4 | 2e+04 | 2969 | 66 |
| 20000 | sklearn (bin seeding on) | 0.7431 | 0.3988 | 1.331 | 5 | 81 | 2644 | 45 |
| 50000 | Custom brute | — | — | — | — | — | — | — |
| 50000 | Custom KD-Tree | — | — | — | — | — | — | — |
| 50000 | Custom KD-Tree + bin seeding | 1.226 | 0.264 | 43.79 | 6 | 87 | 6496 | 13.89 |
| 50000 | sklearn (bin seeding off) | 446.8 | 6.292 | — | 5 | 5e+04 | — | — |
| 50000 | sklearn (bin seeding on) | 2.196 | 0.7871 | 2.157 | 6 | 87 | 6466 | 37 |

### 3.2. Thời gian, bộ nhớ và số cụm theo h
| alpha | h | Phương pháp | Median thời gian (s) | IQR thời gian (s) | Bộ nhớ đỉnh (MB) | Số cụm trung vị | Số seed trung vị | Láng giềng TB/truy vấn | Vòng lặp TB |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.38855 | Custom brute | 11.51 | 1.01 | 67.79 | 210 | 5e+03 | 55.14 | 10.75 |
| 0.25 | 0.38855 | Custom KD-Tree | 1.471 | 1.766 | 18.11 | 210 | 5e+03 | 55.14 | 10.75 |
| 0.25 | 0.38855 | Custom KD-Tree + bin seeding | 0.6533 | 0.2999 | 1.61 | 188 | 651 | 41.73 | 8.972 |
| 0.25 | 0.38855 | sklearn (bin seeding off) | 60.14 | 2.085 | 2.228 | 210 | 5e+03 | 54.97 | 59 |
| 0.25 | 0.38855 | sklearn (bin seeding on) | 5.022 | 0.935 | 1.035 | 188 | 651 | 40.63 | 46 |
| 0.5 | 0.77709 | Custom brute | 26.48 | 1.707 | 95.1 | 12 | 5e+03 | 236.1 | 21.57 |
| 0.5 | 0.77709 | Custom KD-Tree | 15.55 | 4.076 | — | 12 | 5e+03 | — | — |
| 0.5 | 0.77709 | Custom KD-Tree + bin seeding | 0.707 | 0.194 | 3.358 | 11 | 204 | 215.3 | 24.8 |
| 0.5 | 0.77709 | sklearn (bin seeding off) | 124.8 | 10.8 | 2.635 | 12 | 5e+03 | 236.1 | 67 |
| 0.5 | 0.77709 | sklearn (bin seeding on) | 2.921 | 1.515 | 0.9746 | 11 | 204 | 214.9 | 63 |
| 1 | 1.5542 | Custom brute | 23.67 | 2.783 | 165.4 | 6 | 5e+03 | 756.1 | 16.64 |
| 1 | 1.5542 | Custom KD-Tree | — | — | — | — | — | — | — |
| 1 | 1.5542 | Custom KD-Tree + bin seeding | 0.596 | 0.1914 | 3.495 | 5 | 68 | 690.7 | 16.24 |
| 1 | 1.5542 | sklearn (bin seeding off) | 49 | 25.72 | 2.502 | 6 | 5e+03 | 756 | 84 |
| 1 | 1.5542 | sklearn (bin seeding on) | 0.3727 | 0.1096 | 0.9092 | 5 | 68 | 687.8 | 41 |
| 2 | 3.1084 | Custom brute | 31.8 | 0.8988 | 319.6 | 2 | 5e+03 | 2132 | 14.85 |
| 2 | 3.1084 | Custom KD-Tree | — | — | — | — | — | — | — |
| 2 | 3.1084 | Custom KD-Tree + bin seeding | 0.1526 | 0.03676 | 2.555 | 2 | 23 | 1989 | 14.3 |
| 2 | 3.1084 | sklearn (bin seeding off) | 68.49 | 7.41 | 2.456 | 2 | 5e+03 | 2132 | 29 |
| 2 | 3.1084 | sklearn (bin seeding on) | 0.2258 | 0.08189 | 0.9085 | 2 | 23 | 1978 | 28 |

## 4. Trả lời định lượng các câu hỏi RQ3

### 4.1. KD-Tree bắt đầu nhanh hơn brute ở n nào?
Tốc độ tương đối được tính bằng `median_time(brute) / median_time(KD-Tree)`. Giá trị > 1 nghĩa là KD-Tree nhanh hơn; giá trị < 1 nghĩa là KD-Tree chậm hơn trong cấu hình và môi trường đã đo.

| n | Brute median (s) | KD-Tree median (s) | Brute/KD-Tree (×) | Láng giềng TB/truy vấn của KD-Tree |
|---:|---:|---:|---:|---:|
| 500 | 0.20181 | 0.20716 | 0.974 | 70.57 |
| 1000 | 0.57317 | 0.63612 | 0.901 | 152 |
| 2000 | 2.0581 | 2.2217 | 0.926 | 293.8 |
| 5000 | 14.506 | 17.947 | 0.808 | 756.1 |
| 10000 | 92.149 | 40.419 | 2.280 | 1484 |
| 20000 | — | — | — | — |
| 50000 | — | — | — | — |

**Kết quả:** trong các mức n được so sánh hợp lệ, KD-Tree lần đầu nhanh hơn brute tại **n = 10,000**; và KD-Tree vẫn nhanh hơn ở tất cả mức n lớn hơn có số liệu hợp lệ. Đây là kết quả quan sát trên máy và bộ dữ liệu này, không phải ngưỡng phổ quát.

**Độ dốc log–log thực nghiệm** (hệ số góc của log10(thời gian median) theo log10(n), chỉ tính tại các điểm đo hợp lệ):
- Custom brute: độ dốc = **2.041**, dựa trên 5 mức n.
- Custom KD-Tree: độ dốc = **2.014**, dựa trên 6 mức n.
- Custom KD-Tree + bin seeding: độ dốc = **1.009**, dựa trên 7 mức n.
- sklearn (bin seeding off): độ dốc = **1.363**, dựa trên 7 mức n.
- sklearn (bin seeding on): độ dốc = **0.495**, dựa trên 7 mức n.
Độ dốc là mô tả thực nghiệm trong khoảng n đã đo; không nên đồng nhất trực tiếp với độ phức tạp tiệm cận lý thuyết.

### 4.2. Khi h tăng, số láng giềng và thời gian thay đổi thế nào?
Bảng sau so sánh mức alpha nhỏ nhất và lớn nhất có số liệu hợp lệ của từng phương pháp. Phần trăm thay đổi được tính từ điểm đầu đến điểm cuối; phần nhận xét về xu hướng dùng tất cả điểm alpha có dữ liệu.

| Phương pháp | alpha đầu → cuối | Láng giềng TB/truy vấn đầu → cuối | Thay đổi láng giềng | Thời gian median đầu → cuối (s) | Thay đổi thời gian | Xu hướng thời gian qua các alpha |
|---|---:|---:|---:|---:|---:|---|
| Custom brute | 0.25 → 2 | 55.14 → 2132 | +3766.4% | 11.508 → 31.802 | +176.3% | biến động không đơn điệu qua các mức đã đo |
| Custom KD-Tree | 0.25 → 0.5 | 55.14 → 55.14 | +0.0% | 1.4713 → 15.554 | +957.1% | tăng hoặc giữ nguyên qua các mức đã đo |
| Custom KD-Tree + bin seeding | 0.25 → 2 | 41.73 → 1989 | +4667.1% | 0.6533 → 0.15263 | -76.6% | biến động không đơn điệu qua các mức đã đo |
| sklearn (bin seeding off) | 0.25 → 2 | 54.97 → 2132 | +3778.3% | 60.135 → 68.487 | +13.9% | biến động không đơn điệu qua các mức đã đo |
| sklearn (bin seeding on) | 0.25 → 2 | 40.63 → 1978 | +4768.8% | 5.0218 → 0.22583 | -95.5% | giảm hoặc giữ nguyên qua các mức đã đo |

**Diễn giải KD-Tree:** không đủ ít nhất hai giá trị `mean_neighbors_per_query` hợp lệ để kết luận xu hướng láng giềng khi h tăng. Hãy kiểm tra `e3_runs.csv` xem lượt diagnostics có ghi được bộ đếm hay không.
Thời gian median của KD-Tree tăng hoặc giữ nguyên qua các mức đã đo theo alpha; từ alpha=0.25 đến alpha=0.5, thời gian đổi từ 1.4713 s sang 15.554 s (+957.1%).
Số láng giềng lớn hơn có thể làm giảm lợi thế của KD-Tree vì truy vấn cần trả về nhiều điểm hơn; chỉ kết luận cơ chế này phù hợp với kết quả nếu số láng giềng thực đo và thời gian cùng ủng hộ nhận xét, không suy ra quan hệ nhân quả chỉ từ lý thuyết.

### 4.3. Bin seeding giảm bao nhiêu seed và thời gian?
So sánh `custom_kdtree_bin` với `custom_kdtree` tại cùng n và h. Phần trăm tiết kiệm seed = (seed không bin − seed có bin) / seed không bin; phần trăm tiết kiệm thời gian được tính tương tự. Số âm nghĩa là cấu hình có bin seeding tốn nhiều tài nguyên/thời gian hơn ở phép đo đó.

| n | Seed không bin | Seed có bin | Seed giảm | Seed giảm (%) | Số cụm không bin | Số cụm có bin | Thời gian không bin (s) | Thời gian có bin (s) | Thời gian tiết kiệm (s) | Thời gian tiết kiệm (%) | Speedup có bin (×) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 500 | 48 | 452 | +90.4% | 6 | 5 | 0.20716 | 0.012405 | +0.19476 | +94.0% | 16.700 |
| 1000 | 1e+03 | 56 | 944 | +94.4% | 5 | 4 | 0.63612 | 0.029237 | +0.60688 | +95.4% | 21.757 |
| 2000 | 2e+03 | 60 | 1940 | +97.0% | 6 | 6 | 2.2217 | 0.036363 | +2.1854 | +98.4% | 61.099 |
| 5000 | 5e+03 | 68 | 4932 | +98.6% | 6 | 5 | 17.947 | 0.11153 | +17.835 | +99.4% | 160.914 |
| 10000 | 1e+04 | 73 | 9927 | +99.3% | 4 | 5 | 40.419 | 0.31255 | +40.107 | +99.2% | 129.323 |
| 20000 | 2e+04 | 81 | 1.992e+04 | +99.6% | 4 | 5 | 445.33 | 0.49739 | +444.83 | +99.9% | 895.326 |
| 50000 | — | 87 | — | — | — | 6 | — | 1.226 | — | — | — |

Mức giảm seed lớn nhất theo tỷ lệ là +99.6% tại n=20,000 (từ 20000 xuống 81 seed).
Mức tiết kiệm thời gian lớn nhất theo tỷ lệ là +99.9% tại n=20,000 (+444.83 s; speedup=895.326×). Số âm nghĩa là bin seeding chậm hơn KD-Tree không bin ở cấu hình đó.
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
