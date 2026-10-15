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

Hai chiều dữ liệu được chuẩn hóa bằng `StandardScaler` riêng cho mỗi dataset/seed. Năm cấu hình nhận cùng dữ liệu đã chuẩn hóa.

### Cấu hình

Tham số của Mean-Shift (h chọn) và DBSCAN được chọn theo ARI cao nhất trên **seed 0**, giữ nguyên cho seed 1–9. Bảng tổng hợp chỉ tính trên **seed 1–9** (bỏ seed đã dùng để chọn tham số).

| Thuật toán | Tham số |
|---|---|
| MeanShift (h mặc định) | `MeanShiftCustom`, flat, kdtree, bin_seeding; h = h₀ = `estimate_bandwidth(Xs, quantile=0.3)` — không biết số cụm |
| MeanShift (h chọn) | như trên, h = α·h₀, α ∈ {0.25, 0.5, 0.75, 1, 1.5} chọn trên seed 0 |
| KMeans | k = số cụm đúng (oracle), n_init=10, random_state=seed |
| DBSCAN | min_samples=5, eps ∈ {0.05, 0.1, …, 0.5} chọn trên seed 0 |
| GMM | k = số cụm đúng (oracle), covariance_type=full, n_init=5, random_state=seed |

Tham số đã chọn (`results/e2/e2_params.csv`):

| Dataset | α (Mean-Shift) | eps (DBSCAN) |
|---|---:|---:|
| blobs | 0.5 | 0.1 |
| blobs_varied | 0.5 | 0.1 |
| blobs_aniso | 1.5 | 0.05 |
| moons | 1.0 | 0.2 |
| circles | 0.5 | 0.3 |

E2 có 250 lượt fit: 5 dataset × 10 seed × 5 cấu hình. ARI và NMI được tính với toàn bộ nhãn dự đoán; nhãn nhiễu `-1` của DBSCAN được tính như một nhãn, nhưng không đếm vào số cụm. Thời gian chỉ đo bước `fit`.

### Kết quả ARI trung bình (seed 1–9)

Nguồn: `results/e2/e2_summary.csv`, làm tròn ba chữ số.

| Dataset | MS (h mặc định) | MS (h chọn) | KMeans | DBSCAN | GMM |
|---|---:|---:|---:|---:|---:|
| blobs | 0.455 | 0.902 | 0.975 | 0.888 | 0.975 |
| blobs_varied | 0.697 | 0.750 | 0.734 | 0.633 | 0.864 |
| blobs_aniso | 0.657 | 0.528 | 0.704 | 0.500 | 0.842 |
| moons | 0.546 | 0.546 | 0.483 | 1.000 | 0.504 |
| circles | 0.000 | 0.154 | -0.001 | 1.000 | -0.001 |

Nhận xét:

- GMM có ARI cao nhất ở ba dạng blobs; DBSCAN cao nhất ở moons và circles (cụm không lồi).
- h mặc định chưa tốt trên blobs (0.455); giảm h còn 0.5·h₀ đưa ARI lên 0.902, gần K-Means/GMM dù Mean-Shift không biết số cụm.
- Trên blobs_aniso, α = 1.5 chọn trên seed 0 lại kém hơn h mặc định ở seed 1–9 (0.528 so với 0.657): chọn tham số trên một seed dễ bị quá khớp, và độ lệch chuẩn ARI ở bộ này lớn (≈ 0.35).
- Mean-Shift với flat kernel không tách được moons và circles, giống K-Means và GMM.

### Giới hạn diễn giải

- K-Means/GMM có lợi thế biết trước số cụm (oracle-k).
- Mean-Shift (h chọn) và DBSCAN được chọn tham số bằng nhãn thật của seed 0; kết quả seed 1–9 không dùng nhãn để chọn.
- E2 sử dụng dữ liệu chuẩn hóa; E1 sử dụng dữ liệu gốc, nên bandwidth của hai thí nghiệm không tương đương.
- E1 đánh giá sự tương ứng với sklearn; E2 đánh giá sự tương ứng với nhãn thật. Khớp sklearn không bảo đảm khôi phục được nhóm thật trên mọi hình dạng dữ liệu.
- Thời gian là phép đo fit trên môi trường cụ thể, chưa đủ để kết luận thuật toán nào luôn nhanh hơn.
- Dữ liệu tổng hợp hai chiều không đủ để suy rộng kết quả sang ảnh da hoặc dữ liệu thực tế.

## Tái lập thực nghiệm

Các lệnh chạy từ thư mục gốc `ML_Final`:

```bash
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
| results/e2/e2_params.csv | Tham số chọn trên seed 0 |
| results/e2/e2_runs.csv | Chi tiết từng lượt fit |
| results/e2/e2_summary.csv | Trung bình và độ lệch chuẩn các chỉ số |
| results/e2/e2_grid.png | Lưới 5 dataset × 5 cấu hình ở seed 0 |
| results/e2/e2_ari_bar.png | ARI trung bình ± std |
| results/e2/e2_metadata.json | Cấu hình, phiên bản thư viện và SHA256 đầu vào |

`tests/test_vs_sklearn.py` kiểm tra sự tương ứng với sklearn khi dùng flat kernel, phân hoạch trên moons, seed tường minh, hai backend, điểm nhiễu và `predict`, mode đồng hạng, giới hạn số vòng lặp và cập nhật Gaussian cắt ngưỡng.

Dữ liệu tổng hợp và các file trong `results/` được sinh lại từ script, không theo dõi trên Git. Metadata hỗ trợ đối chiếu cấu hình và đầu vào; thời gian đo và kết quả số có thể khác khi môi trường hoặc phiên bản thư viện thay đổi.
