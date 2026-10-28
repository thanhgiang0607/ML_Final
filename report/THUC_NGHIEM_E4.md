# Thực nghiệm E4 – Độ nhạy theo bandwidth

## Phần 2: ISIC tune với đặc trưng 3D Lab

- Số ảnh: **10**.
- Đặc trưng: `features(rgb, 'lab3')`.
- Thuật toán: `MeanShiftCustom(bandwidth=h, bin_seeding=True)`.
- Bandwidth cố định: 2, 4, 6, 8, 10, 12, 16, 20.
- Bandwidth ước lượng: `estimate_bandwidth(X)` riêng cho từng ảnh.
- Số lần fit mỗi cấu hình: 1.

### Kết quả tổng hợp

| h | Số ảnh | Bandwidth TB | Dice TB | Dice std | IoU TB | Số cụm TB | Thời gian TB (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 10 | 2.0000 | 0.0533 | 0.1412 | 0.0333 | 121.10 | 1.589 |
| 4 | 10 | 4.0000 | 0.2443 | 0.2269 | 0.1577 | 14.00 | 1.058 |
| 6 | 10 | 6.0000 | 0.2490 | 0.3170 | 0.1823 | 5.50 | 0.523 |
| 8 | 10 | 8.0000 | 0.4675 | 0.4057 | 0.3893 | 2.80 | 0.309 |
| 10 | 10 | 10.0000 | 0.2891 | 0.3775 | 0.2316 | 2.30 | 0.229 |
| 12 | 10 | 12.0000 | 0.3412 | 0.4386 | 0.2994 | 2.00 | 0.150 |
| 16 | 10 | 16.0000 | 0.2668 | 0.4298 | 0.2404 | 1.50 | 0.122 |
| 20 | 10 | 20.0000 | 0.2755 | 0.4436 | 0.2547 | 1.30 | 0.093 |
| estimated | 10 | 9.4902 | 0.4607 | 0.4185 | 0.3910 | 3.30 | 0.433 |

### Kết luận

Cấu hình có Dice trung bình cao nhất là **8**, với Dice trung bình **0.4675** và độ lệch chuẩn **0.4057**.

Nếu cấu hình `estimated` đứng đầu, bandwidth cần được ước lượng riêng cho từng ảnh; không dùng bandwidth trung bình làm giá trị cố định thay thế.

Việc lựa chọn tham số chỉ sử dụng tập tune, không sử dụng tập test.
File CSV chi tiết `e4_isic_runs.csv` dùng để kiểm tra kết quả.
