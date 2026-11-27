# Kết quả NPR trên tập test đã làm sạch

Người cung cấp dữ liệu xác nhận đã tải một phần dataset từ nguồn do tác giả cung cấp vì giới hạn tài nguyên máy.

Đã kiểm tra hash và extrema RGB của toàn bộ 38.976 ảnh trong snapshot test. Tiêu chí loại là mọi pixel RGB bằng 0, áp dụng cho cả hai nhãn và mọi nguồn. Có 103 đường dẫn đáp ứng tiêu chí, cùng nằm trong sdv1/Fake và có cùng SHA256. Danh sách cụ thể nằm trong excluded_images.csv.

| Dataset | Real | Fake | Total | Excluded_black |
| --- | --- | --- | --- | --- |
| progan | 4000 | 4000 | 8000 | 0 |
| stylegan2 | 7988 | 7988 | 15976 | 0 |
| sdv1 | 5000 | 9897 | 14897 | 103 |

| Dataset | Real | Fake | Total | Accuracy | Precision | Recall | F1 | AP | ROC_AUC | TN | FP | FN | TP | Time | MeanTime_ms | Eligible | Complete | Metric_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| progan | 4000 | 4000 | 8000 | 0.997875 | 0.9982486865148862 | 0.9975 | 0.9978742028260598 | 0.999916305279526 | 0.9999015000000001 | 3993 | 7 | 10 | 3990 | 344.3158235998271 | 43.03947794997839 | 8000 | True | Fake=1 positive |
| stylegan2 | 7988 | 7988 | 15976 | 0.9839133700550826 | 0.9972983404091085 | 0.9704556835252879 | 0.9836939280502506 | 0.9993641769193573 | 0.9994090252805348 | 7967 | 21 | 236 | 7752 | 708.800316199995 | 44.36656961692508 | 15976 | True | Fake=1 positive |
| sdv1 | 5000 | 9897 | 14897 | 0.9756326777203463 | 0.9911395013393777 | 0.9720117207234515 | 0.9814824261592613 | 0.996000526190644 | 0.9954433969889867 | 4914 | 86 | 277 | 9620 | 710.5944841688182 | 47.700509107123466 | 14897 | True | Fake=1 positive |

sdv1 đã làm sạch có 14897 ảnh, FP=86, FN=277. Accuracy/AP/F1 trong CSV dùng thang 0–1. before_after_metrics.csv ghi cả hai phiên bản và chênh lệch điểm phần trăm.

Các logit, score, nhãn dự đoán và thời gian mỗi ảnh được lấy nguyên từ lần inference baseline đã thực chạy. Metrics và biểu đồ được tính lại trên các dòng còn lại. Lần tạo phiên bản này dùng lại dự đoán đã lưu. Time/MeanTime_ms được tổng hợp từ thời gian batch phân bổ ban đầu.

Người 2 sử dụng results/cleaned_black103/data_manifest.csv cho mọi checkpoint so sánh. Giữ image_id, nhãn, preprocessing và threshold 0.5. Dùng output riêng cho mỗi lần replay hoặc mô hình cải tiến. Để chạy inference mới toàn bộ tập đã làm sạch, dùng lệnh sau từ root dự án.

```powershell
& D:\XLA\.venv\Scripts\python.exe -X utf8 evaluate_checkpoint.py --manifest results\cleaned_black103\data_manifest.csv --checkpoint NPR.pth --output results\cleaned_replay --batch-size 16 --threads 4
```

Nguồn tải từ tác giả và việc chỉ tải một phần được ghi theo thông tin người dùng cung cấp. Link/gói archive và sự tương ứng chính xác với sdv1_new vẫn cần đối chiếu. Tiêu chí làm sạch là một thay đổi protocol phải ghi khi trình bày với giảng viên.
