Người cung cấp dữ liệu xác nhận đã tải một phần dataset từ nguồn do tác giả cung cấp vì giới hạn tài nguyên máy.

Tập test được chọn sau khi loại toàn bộ ảnh có pixel RGB đều bằng 0.

# Phân tích lỗi trên dữ liệu thực tế

| Dataset | Real | Fake | Total | Accuracy | Precision | Recall | F1 | AP | ROC_AUC | TN | FP | FN | TP | Time | MeanTime_ms | Eligible | Complete | Metric_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| progan | 4000 | 4000 | 8000 | 0.997875 | 0.9982486865148862 | 0.9975 | 0.9978742028260598 | 0.999916305279526 | 0.9999015000000001 | 3993 | 7 | 10 | 3990 | 344.3158235998271 | 43.03947794997839 | 8000 | True | Fake=1 positive |
| stylegan2 | 7988 | 7988 | 15976 | 0.9839133700550826 | 0.9972983404091085 | 0.9704556835252879 | 0.9836939280502506 | 0.9993641769193573 | 0.9994090252805348 | 7967 | 21 | 236 | 7752 | 708.800316199995 | 44.36656961692508 | 15976 | True | Fake=1 positive |
| sdv1 | 5000 | 9897 | 14897 | 0.9756326777203463 | 0.9911395013393777 | 0.9720117207234515 | 0.9814824261592613 | 0.996000526190644 | 0.9954433969889867 | 4914 | 86 | 277 | 9620 | 710.5944841688182 | 47.700509107123466 | 14897 | True | Fake=1 positive |

Trong ba nguồn local, Accuracy cao nhất: progan (99.7875%); thấp nhất: sdv1 (97.5633%). Trung bình Accuracy hai nguồn GAN (không trọng số): 99.0894%; sdv1 local: 97.5633%; chênh lệch sdv1 − GAN: -1.5262 điểm %. Đây chỉ là quan sát trên các tập local khác nội dung và tỷ lệ lớp; không chứng minh tác động nhân quả của GAN/Diffusion, đặc biệt danh tính nguồn sdv1 chưa được xác minh.

- progan: FP=7, FN=10; Real=4000, Fake=4000; Recall Fake=99.7500%.
  Trong các lỗi của nguồn này, bỏ sót Fake (FN) chiếm 58.82%; cần xem false negatives khi nghiên cứu cải tiến.
- stylegan2: FP=21, FN=236; Real=7988, Fake=7988; Recall Fake=97.0456%.
  Trong các lỗi của nguồn này, bỏ sót Fake (FN) chiếm 91.83%; cần xem false negatives khi nghiên cứu cải tiến.
- sdv1: FP=86, FN=277; Real=5000, Fake=9897; Recall Fake=97.2012%.
  Trong các lỗi của nguồn này, bỏ sót Fake (FN) chiếm 76.31%; cần xem false negatives khi nghiên cứu cải tiến.

## Theo danh mục

| dataset | category | total | errors | accuracy |
| --- | --- | --- | --- | --- |
| progan | airplane | 400 | 1 | 0.9975 |
| progan | bicycle | 400 | 5 | 0.9875 |
| progan | bird | 400 | 1 | 0.9975 |
| progan | boat | 400 | 0 | 1.0 |
| progan | bottle | 400 | 0 | 1.0 |
| progan | bus | 400 | 0 | 1.0 |
| progan | car | 400 | 0 | 1.0 |
| progan | cat | 400 | 0 | 1.0 |
| progan | chair | 400 | 0 | 1.0 |
| progan | cow | 400 | 1 | 0.9975 |
| progan | diningtable | 400 | 1 | 0.9975 |
| progan | dog | 400 | 0 | 1.0 |
| progan | horse | 400 | 0 | 1.0 |
| progan | motorbike | 400 | 2 | 0.995 |
| progan | person | 400 | 1 | 0.9975 |
| progan | pottedplant | 400 | 0 | 1.0 |
| progan | sheep | 400 | 1 | 0.9975 |
| progan | sofa | 400 | 0 | 1.0 |
| progan | train | 400 | 4 | 0.99 |
| progan | tvmonitor | 400 | 0 | 1.0 |
| sdv1 | imagenet | 14897 | 363 | 0.9756326777203463 |
| stylegan2 | car | 3994 | 21 | 0.9947421131697546 |
| stylegan2 | cat | 3994 | 1 | 0.9997496244366549 |
| stylegan2 | church | 3994 | 42 | 0.9894842263395093 |
| stylegan2 | horse | 3994 | 193 | 0.9516775162744117 |

## Quan sát đặc điểm các ảnh sai

- progan: 1/17 ảnh sai có độ lệch trung bình giữa kênh RGB < 1/255 sau resize (gần ảnh xám).
- sdv1: 0/363 ảnh sai có độ lệch trung bình giữa kênh RGB < 1/255 sau resize (gần ảnh xám).
- stylegan2: 0/257 ảnh sai có độ lệch trung bình giữa kênh RGB < 1/255 sau resize (gần ảnh xám).

Đây là thống kê trong nhóm dự đoán sai, chưa phải tỷ lệ lỗi của toàn bộ ảnh xám. Không suy ra ảnh xám gây lỗi khi chưa có nhóm đối chứng đầy đủ. Xem gallery và từng dòng CSV để kiểm tra trực tiếp.


## Gợi ý cho Người 2

Xem error_analysis.csv và gallery incorrect để kiểm tra các lỗi tự tin cao; near_threshold để xem mẫu không chắc chắn. Số lỗi theo danh mục ở trên là bằng chứng trực tiếp; không suy diễn đặc điểm thị giác gây lỗi chỉ từ tên file. Kiểm tra robustness với resize/JPEG hoặc phân tích từng domain là hướng nghiên cứu tiếp theo, chưa được đo ở baseline này. Không chọn threshold từ tập test để báo cáo cải tiến; dùng tập validation riêng và báo cáo cùng manifest test. Do tỷ lệ Fake của sdv1 đã làm sạch là 66.4362%, AP cần diễn giải cùng prevalence; không gộp các nguồn rồi tuyên bố generalization. Kiểm tra duplicate/leakage và xác minh archive trước khi kết luận tái lập paper.
