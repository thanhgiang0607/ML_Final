# Thực nghiệm E3 — Tốc độ và bộ nhớ

## 1. Mục tiêu
Trả lời RQ3: KD-Tree và bin seeding giảm chi phí tính toán bao nhiêu, và trong điều kiện nào?

## 2. Cấu hình thực nghiệm
- Dữ liệu: `make_blobs`, centers=5, cluster_std=1.0, d=2, random_state=0.
- Bandwidth tham chiếu: h0=4.2374815, ước lượng một lần trên mẫu n=2000 điểm với quantile=0.3.
- Quét n: 500, 1000, 2000, 5000, 10000, 20000, 50000; giữ h=h0.
- Quét h: n=5000; alpha=0.25, 0.5, 1.0, 2.0; h=alpha*h0.
- Số lần lặp theo cấu hình hiện tại: 3; timeout fit: 600 giây/lượt.
- Các biến thể: Custom brute; Custom KD-Tree; Custom KD-Tree + bin seeding; sklearn MeanShift (bin seeding off/on, n_jobs=1).
- Đo thời gian bằng `time.perf_counter()`; bộ nhớ đỉnh bằng `tracemalloc` bắt đầu trước `fit(X)`. Báo cáo median và IQR.
- Môi trường: CPU Intel64 Family 6 Model 186 Stepping 3, GenuineIntel; RAM tổng (GB) 15.692; Python 3.13.6 (tags/v3.13.6:4e66535, Aug  6 2025, 14:36:00) [MSC v.1944 64 bit (AMD64)]; NumPy 2.5.4; SciPy 1.18.1; scikit-learn 1.9.1.

## 3. Kết quả tóm tắt
Bảng chi tiết nằm ở `../results/e3/e3_summary.csv`; dữ liệu từng lượt chạy ở `../results/e3/e3_runs.csv` (không commit theo quy ước nhóm).

### 3.1. Thời gian theo n
| n | Custom brute (s) | Custom KD-Tree (s) | KD-Tree + bin seeding (s) | sklearn off (s) | sklearn on (s) |
|---:|---:|---:|---:|---:|---:|
| 500 | 0.7219 | 1.569 | 0.03351 | 5.7 | 0.1786 |
| 1000 | 2.372 | 7.611 | 0.04106 | 12.41 | 0.1747 |
| 2000 | 8.943 | 40.71 | 0.09871 | 26.7 | 0.2485 |
| 5000 | 31.04 | 220.9 | 0.2414 | 95.34 | 0.2811 |
| 10000 | 147.9 | — | 0.6042 | 218.5 | 0.3688 |
| 20000 | — | — | 1.409 | 462 | 0.894 |
| 50000 | — | — | 5.596 | — | 0.8614 |

### 3.2. Thời gian theo h
| alpha | h | Custom brute (s) | Custom KD-Tree (s) | KD-Tree + bin seeding (s) | sklearn off (s) | sklearn on (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 1.0594 | 26.27 | 61.89 | 1.085 | 274.8 | 2.907 |
| 0.5 | 2.1187 | 31.35 | 186.7 | 0.7378 | 231.5 | 0.7492 |
| 1 | 4.2375 | 16.15 | 143.6 | 0.4443 | 109.5 | 0.2413 |
| 2 | 8.475 | 6.185 | 149.5 | 0.07609 | 22.78 | 0.1318 |

## 4. Trả lời các ý của RQ3
- **KD-Tree so với brute:** không ghi nhận KD-Tree nhanh hơn brute ở các kích thước n có đủ dữ liệu hợp lệ, hoặc một trong hai biến thể chưa có số liệu để kết luận.
- **Độ dốc log–log thực nghiệm** (hệ số góc của log10(thời gian) theo log10(n); chỉ tính trên các điểm hợp lệ):
  - Custom brute: 1.734 (dựa trên 5 mức n).
  - Custom KD-Tree: 2.164 (dựa trên 4 mức n).
  - Custom KD-Tree + bin seeding: 1.134 (dựa trên 7 mức n).
  - sklearn (bin seeding off): 1.214 (dựa trên 6 mức n).
  - sklearn (bin seeding on): 0.389 (dựa trên 7 mức n).
- **Bin seeding:** tại n=5000, KD-Tree có bin seeding đạt tốc độ khoảng 915× so với KD-Tree không bin seeding (tỷ số thời gian median trên số liệu hợp lệ). Số seed trung vị ở hai cấu hình lần lượt là 14 và 5000 nếu có số liệu.
- **Ảnh hưởng của h:** tham chiếu bảng thời gian theo alpha ở trên và biểu đồ `e3_time_vs_h.png`. Diễn giải KD-Tree cần đối chiếu xu hướng thời gian với `mean_neighbors_per_query`; số láng giềng tăng theo h là cơ chế dự kiến trong thiết kế nghiên cứu, còn mức tăng cụ thể phụ thuộc vào dữ liệu và kết quả đo.

## 5. Biểu đồ
- `../results/e3/e3_time_vs_n.png` — thời gian theo n, trục log–log.
- `../results/e3/e3_memory_vs_n.png` — bộ nhớ đỉnh theo n.
- `../results/e3/e3_time_vs_h.png` — thời gian theo h.

## 6. Giới hạn và lưu ý diễn giải
- `tracemalloc` ghi nhận bộ nhớ do allocator Python theo dõi; có thể không bao quát đầy đủ bộ nhớ native do NumPy, SciPy, KD-Tree hoặc thư viện sklearn cấp phát. Không diễn giải số đo này như RSS tổng của tiến trình.
- Thời gian là thời gian `fit(X)` trong môi trường đã ghi. So sánh không tự động chứng minh một thuật toán luôn nhanh hơn trên mọi máy hoặc mọi dữ liệu.
- Các lượt timeout/skip được ghi trong CSV; chúng không được dùng để tính median thời gian thành công.
- Số láng giềng trung bình và số vòng lặp được ghi khi implementation/API cho phép instrument. Nếu giá trị trống, cần kiểm tra tương thích API hoặc báo cáo là không khả dụng, không tự suy ra giá trị.

## 7. File đầu ra
- `../results/e3/e3_runs.csv` — dữ liệu từng lượt chạy.
- `../results/e3/e3_summary.csv` — median, Q1, Q3, IQR và hệ số tăng tốc so với custom brute.
- `../results/e3/env.txt` và `../results/e3/e3_metadata.json` — môi trường và cấu hình chạy.
