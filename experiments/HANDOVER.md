# Bàn giao thí nghiệm JPEG

Thí nghiệm đo ảnh hưởng của JPEG augmentation đến NPR trên ProGAN, StyleGAN2 và `sdv1` local. Ba phiên bản dùng trong bảng so sánh là NPR gốc, NPR fine-tune đối chứng và NPR fine-tune có JPEG augmentation.

- Dữ liệu fine-tune gồm 160 ảnh train và 40 ảnh validation cân bằng nhãn từ `car` và `chair` trong gói ProGAN validation của CNNDetection. Hai lượt fine-tune bắt đầu từ cùng checkpoint, chạy hai epoch và chọn checkpoint theo BCE thấp nhất trên validation sạch. Đối chứng chọn epoch 2, bản JPEG chọn epoch 1.
- Tập đánh giá gồm 360 ảnh, với 60 ảnh cho mỗi cặp nguồn và nhãn. Tập này lấy từ manifest test đã làm sạch bằng seed 202. Không có ảnh trùng với lượt thử 120 ảnh theo mã ảnh, SHA-256 hoặc pixel RGB đã giải mã.
- Mỗi mã ảnh được đo ở bốn điều kiện gồm ảnh gốc và JPEG `q90/q70/q50`. Cả ba phiên bản dùng evaluator của Thành viên 1 và ngưỡng `fake_score > 0.5`. Accuracy tính trên 360 ảnh. AP trung bình tính bằng trung bình AP của ba nguồn.

Các lệnh tạo lại dữ liệu và chạy đánh giá nằm trong [README thí nghiệm](README.md). Lượt đo thực tế dùng Python 3.14.7, PyTorch 2.14.0 CPU và Pillow 12.3.0.

## Kết quả trên 360 ảnh

Mỗi ô ghi Accuracy và AP trung bình theo nguồn, trên thang phần trăm. Đã đối chiếu checkpoint, mã ảnh, nhãn và tính lại các chỉ số từ 4.320 dự đoán.

| Điều kiện | NPR gốc | Đối chứng | JPEG augmentation |
|---|---:|---:|---:|
| Ảnh gốc | 98,61 / 99,86 | 97,78 / 99,86 | 97,50 / 99,87 |
| q90 | 51,94 / 61,83 | 51,39 / 63,33 | 55,28 / 63,37 |
| q70 | 51,11 / 62,24 | 50,56 / 64,02 | 53,61 / 63,37 |
| q50 | 50,56 / 65,34 | 50,56 / 66,81 | 53,33 / 66,49 |

Ở ngưỡng 0,5, bản JPEG tăng Accuracy so với đối chứng lần lượt 3,89, 3,06 và 2,78 điểm phần trăm ở `q90/q70/q50`. Số ảnh giả nhận ra tăng từ 7 lên 24 ở q90, từ 2 lên 17 ở q70 và từ 2 lên 14 ở q50, trên 180 ảnh giả mỗi điều kiện. Phần lớn ảnh nhận thêm thuộc `sdv1`. Ở q50, bản JPEG bỏ sót cả 120 ảnh giả ProGAN và StyleGAN2. AP trung bình thay đổi từ −0,65 đến +0,04 điểm phần trăm trên ba mức nén, còn Accuracy ảnh gốc giảm 0,28 điểm phần trăm so với đối chứng.

Khi báo cáo, mô tả đây là cải thiện Accuracy ở ngưỡng đang dùng trên tập con cân bằng. Kết quả theo nguồn và AP cho thấy hiệu quả còn phụ thuộc nguồn ảnh. Giữ NPR gốc làm mặc định cho demo và dùng bản JPEG để minh họa kết quả đối chiếu.

## Tệp dùng cho báo cáo và demo

| Tệp hoặc thư mục | Cách sử dụng |
|---|---|
| [Bảng kết quả CSV](results/jpeg-heldout.csv) | Dùng Accuracy, AP và các ô TN/FP/FN/TP theo từng nguồn, mô hình và điều kiện. |
| `outputs/finetune-pilot/npr_control.pth` | Checkpoint đối chứng. SHA-256 `29ce05bef689e06976fe9756beac7e40c969fe28c1a48b9d49f7910995b98cb1`. |
| `outputs/finetune-pilot/npr_jpeg.pth` | Checkpoint có JPEG augmentation. SHA-256 `9b090e811e2f5764b00c89f443325b2e4ce0f743e925e229b9fbfa75d10d4079`. |
| `outputs/finetune-pilot/config.json` và hai tệp `*_history.csv` | Dùng để đối chiếu cấu hình train và epoch được chọn. |
| `outputs/jpeg-heldout/` và `outputs/jpeg-comparison-heldout/` | Manifest, ảnh JPEG và điểm dự đoán từng ảnh phục vụ kiểm tra và minh họa. |

Khi tích hợp demo, dùng cùng hàm `load_model` của Thành viên 1 để nạp từng checkpoint. Hiển thị tên phiên bản cạnh nhãn và điểm dự đoán để người dùng biết mô hình nào đang chạy. Hai checkpoint nằm trong thư mục đầu ra cục bộ và có thể bàn giao riêng theo SHA-256 ở trên.

Hai ví dụ dưới đây nằm trong `outputs/jpeg-heldout/q50/`. Dùng cùng ảnh với cả ba checkpoint khi demo.

| Tệp | Nhãn thật | NPR gốc | Đối chứng | Bản JPEG |
|---|---|---|---|---|
| `000305.jpg` | Giả, `sdv1` | Thật | Thật | Giả, điểm 0,9367 |
| `000032.jpg` | Thật, ProGAN | Thật | Thật | Giả, điểm 0,5204 |
