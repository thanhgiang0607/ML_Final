# Thực nghiệm E1 và E2

## Mục tiêu

- E1 đối chiếu `MeanShiftCustom` với `sklearn.cluster.MeanShift` trên cùng dữ liệu và cấu hình, đánh giá số cụm, nhãn, tâm cụm và số vòng lặp.
- E2 so sánh `MeanShiftCustom`, K-Means, DBSCAN và Gaussian Mixture Model (GMM) bằng ARI và NMI với nhãn thật của dữ liệu tổng hợp.

Các phép đo sử dụng dữ liệu tổng hợp hai chiều.

## Dữ liệu

`data/make_synthetic.py` tạo 50 CSV: năm dạng dữ liệu, mỗi dạng 1.500 điểm và 10 seed từ 0 đến 9. Mỗi file có các cột `x,y,label`; hai cột tọa độ là đầu vào, nhãn được dùng để đánh giá.

| Dataset | Đặc điểm | Số cụm từ bộ sinh |
|---|---|---:|
| blobs | Cụm đẳng hướng | 4 |
| blobs_varied | Cụm có độ trải khác nhau | 3 |
| blobs_aniso | Cụm biến đổi tuyến tính, bất đẳng hướng | 3 |
| moons | Hai hình lưỡi liềm | 2 |
| circles | Hai vòng tròn đồng tâm | 2 |

## E1: đối chiếu với sklearn

Script: `experiments/e1_compare_sklearn.py`.

### Cấu hình

E1 dùng tọa độ gốc, không chuẩn hóa. Bandwidth được tính bằng `alpha * meanshift.bandwidth.estimate_bandwidth(X, quantile=0.3)`.

| Tham số | Giá trị |
|---|---|
| alpha | 0.25, 0.5, 1.0 |
| kernel | flat |
| bin_seeding | False, True |
| Backend MeanShiftCustom | brute, kdtree |
| cluster_all | True |
| tol MeanShiftCustom | 1e-3 |
| max_iter | 300 |
| Bán kính gộp mode MeanShiftCustom | h |
| sklearn n_jobs | 1 |
| Giới hạn threadpool | 1 |

`MeanShiftCustom` và sklearn sử dụng cùng X, bandwidth và chế độ seeding. Mỗi cấu hình dataset/seed/alpha/seeding có một lượt fit sklearn, dùng chung cho phép đối chiếu với brute và kdtree.

Thiết kế gồm 600 phép đối chiếu: 5 dataset × 10 seed × 3 alpha × 2 chế độ seeding × 2 backend, tương ứng 600 lượt fit `MeanShiftCustom` và 300 lượt fit sklearn. Chẩn đoán KDE chạy riêng.

### Chỉ số

| Cột kết quả | Ý nghĩa |
|---|---|
| cluster_count_equal | Số cụm của MeanShiftCustom và sklearn bằng nhau |
| ari_custom_vs_sklearn | Mức tương ứng giữa hai phân hoạch, không phụ thuộc hoán đổi số hiệu nhãn |
| labels_exact_equal | Nhãn giống hệt, gồm cả số hiệu cụm |
| center_error_max_over_h | Khoảng cách lớn nhất giữa các cặp tâm đã ghép, chuẩn hóa theo h |
| custom_n_iter, sklearn_n_iter | Số vòng lặp của MeanShiftCustom và sklearn |
| custom_truth_ari, sklearn_truth_ari | ARI của MeanShiftCustom và sklearn so với nhãn thật |

Tâm được ghép bằng `linear_sum_assignment` để tối thiểu hóa tổng khoảng cách Euclidean. Sai số ghép tâm chỉ tính trên các cặp được ghép; số tâm dư được báo cáo riêng. Cột sai số phạt khi lệch số tâm là chỉ số riêng, không thay thế sai số ghép tâm.

`comparison_mismatch` đánh dấu dòng có ít nhất một điều kiện: nhãn không giống hệt, số cụm khác, sai lệch tâm/h lớn hơn 1e-10 hoặc số vòng lặp khác. ARI bằng 1 vẫn có thể đi kèm mismatch nếu số hiệu nhãn khác.

### Kết quả

Nguồn: `results/e1/e1_runs.csv` của lần chạy đủ 10 seed.

| Chỉ số | Kết quả |
|---|---:|
| Số dòng đối chiếu | 600 |
| Cùng số cụm | 600/600 |
| Nhãn giống hệt | 600/600 |
| ARI custom-vs-sklearn nhỏ nhất | 1.0 |
| Sai lệch tâm lớn nhất/h | khoảng 1.56 × 10⁻¹⁴ |
| Dòng mismatch | 0 |

Trong 600 phép đối chiếu, `MeanShiftCustom` và sklearn cho cùng số cụm, nhãn và số vòng lặp. Sai lệch tâm lớn nhất sau chuẩn hóa theo h là 1.56 × 10⁻¹⁴. Kết quả này áp dụng cho flat kernel và các cấu hình E1 đã khảo sát.

### Chẩn đoán KDE

Chẩn đoán sử dụng Gaussian KDE đầy đủ của sklearn và quỹ đạo `MeanShiftCustom` với Gaussian kernel cắt láng giềng ở 3h. Giá trị Gaussian KDE tính trực tiếp tại các điểm quỹ đạo được đối chiếu với sklearn bằng `assert_allclose`.

Đồ thị flat update sử dụng Epanechnikov KDE. Phép đối chiếu mật độ tại các điểm quỹ đạo không kiểm tra tính tăng mật độ hoặc hội tụ của Gaussian cắt ngưỡng.

## E2: so sánh thuật toán

Script: `experiments/e2_compare_algorithms.py`.

Hai chiều dữ liệu được chuẩn hóa bằng `StandardScaler` riêng cho mỗi dataset/seed. Bốn thuật toán nhận cùng dữ liệu đã chuẩn hóa. Nhãn thật không được dùng để tối ưu ARI/NMI; K-Means và GMM được cung cấp số cụm từ bộ sinh.

### Cấu hình

| Thuật toán | Tham số |
|---|---|
| MeanShiftCustom | flat, kdtree, bin_seeding=True, cluster_all=True; h = 0.5 × sklearn estimate_bandwidth(Xs, quantile=0.3, n_samples=None) |
| KMeans | n_clusters từ bộ sinh, n_init=10, random_state=seed |
| DBSCAN | eps=0.2, min_samples=5 |
| GaussianMixture | n_components từ bộ sinh, covariance_type=full, n_init=5, random_state=seed |

E2 có 200 lượt fit: 5 dataset × 10 seed × 4 thuật toán. Kết quả được tổng hợp theo dataset/thuật toán bằng trung bình và độ lệch chuẩn qua 10 seed.

ARI và NMI được tính với toàn bộ nhãn dự đoán. Nhãn nhiễu `-1` của DBSCAN được tính như một giá trị nhãn trong hai chỉ số này, nhưng bị loại khi đếm số cụm. Tỷ lệ nhiễu được lưu riêng.

Thời gian đo chỉ bao quanh `model.fit(Xs)`, không gồm chuẩn hóa, ước lượng bandwidth hoặc bước `predict(Xs)` của GMM.

### Kết quả ARI trung bình

Nguồn: `results/e2/e2_summary.csv` của lần chạy đủ 10 seed, làm tròn bốn chữ số.

| Dataset | MeanShiftCustom | KMeans | DBSCAN | GMM |
|---|---:|---:|---:|---:|
| blobs | 0.9103 | 0.9755 | 0.8183 | 0.9774 |
| blobs_varied | 0.7237 | 0.7088 | 0.5109 | 0.8432 |
| blobs_aniso | 0.7358 | 0.6797 | 0.5374 | 0.8078 |
| moons | 0.3656 | 0.4847 | 0.9996 | 0.5046 |
| circles | 0.1542 | -0.0006 | 0.9997 | -0.0006 |

Với cấu hình này, GMM có ARI trung bình cao nhất ở ba dạng blobs; DBSCAN cao nhất ở moons và circles. Độ ổn định giữa các seed được thể hiện trong các cột độ lệch chuẩn của bảng tổng hợp.

### Giới hạn diễn giải

- K-Means/GMM có lợi thế biết trước số cụm (oracle-k).
- Cấu hình được giữ cố định, chưa có quy trình tuning riêng cho từng thuật toán. Hệ số bandwidth 0.5 và eps=0.2 chưa được xác nhận là tối ưu.
- E2 sử dụng dữ liệu chuẩn hóa và bandwidth ước lượng bằng sklearn. E1 sử dụng dữ liệu gốc và hàm `meanshift.bandwidth.estimate_bandwidth`; bandwidth của hai thí nghiệm không tương đương.
- E1 đánh giá sự tương ứng với sklearn; E2 đánh giá sự tương ứng với nhãn thật. Khớp sklearn không bảo đảm khôi phục được nhóm thật trên mọi hình dạng dữ liệu.
- Thời gian là phép đo fit trên môi trường cụ thể, chưa đủ để kết luận backend hoặc thuật toán nào luôn nhanh hơn.
- Dữ liệu tổng hợp hai chiều không đủ để suy rộng kết quả sang ảnh da hoặc dữ liệu thực tế.

## Tái lập thực nghiệm

Các lệnh chạy từ thư mục gốc `ML_Final`:

```powershell
python -m pip install -r requirements.txt
python data/make_synthetic.py
python -m pytest tests/test_vs_sklearn.py
python experiments/e1_compare_sklearn.py
python experiments/e2_compare_algorithms.py
```


## Đầu ra và kiểm chứng

| Đường dẫn | Nội dung |
|---|---|
| results/e1/e1_runs.csv | Chi tiết từng phép đối chiếu |
| results/e1/e1_summary.csv | Tổng hợp theo dataset, alpha, seeding và backend |
| results/e1/e1_mismatches.csv | Các dòng khác ít nhất một tiêu chí |
| results/e1/e1_examples.png | MeanShiftCustom/kdtree và sklearn ở seed 0, alpha=0.5, không bin seeding |
| results/e1/e1_density_steps.csv và các hình KDE | Mật độ và quỹ đạo chẩn đoán |
| results/e1/e1_metadata.json | Cấu hình, phiên bản thư viện, SHA256 đầu vào và phạm vi phép đo |
| results/e2/e2_runs.csv | Chi tiết từng lượt fit |
| results/e2/e2_summary.csv | Trung bình và độ lệch chuẩn các chỉ số |
| results/e2/e2_examples.png | Phân cụm minh họa ở seed 0 |
| results/e2/e2_score_comparison.png | ARI/NMI trung bình |
| results/e2/e2_metadata.json | Cấu hình, phiên bản thư viện và SHA256 đầu vào |

`tests/test_vs_sklearn.py` kiểm tra sự tương ứng với sklearn khi dùng flat kernel, phân hoạch trên moons, seed tường minh, hai backend, điểm nhiễu và `predict`, mode đồng hạng, giới hạn số vòng lặp và cập nhật Gaussian cắt ngưỡng.

Dữ liệu tổng hợp và các file trong `results/` được sinh lại từ script, không theo dõi trên Git. Metadata hỗ trợ đối chiếu cấu hình và đầu vào; thời gian đo và kết quả số có thể khác khi môi trường hoặc phiên bản thư viện thay đổi.
