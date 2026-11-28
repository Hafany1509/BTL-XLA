# Kết quả NPR + JPEG augmentation trên AutoDL

## Phương pháp

A và B đều được nhóm train NPR từ đầu với seed 100, cùng dữ liệu và lịch train.
Không dùng checkpoint tác giả để khởi tạo B. Giữ nguyên kiến trúc CNN NPR
1.437.761 tham số; chỉ bật JPEG augmentation cho B, áp dụng cả ảnh thật và giả.

| Tham số | A | B |
|---|---|---|
| Train | car, cat, chair, horse; 144.024 ảnh | Như A |
| Validation | ProGAN cùng 4 nhóm; 1.600 ảnh | Như A, không nén thêm |
| Epoch | 50 | 50 |
| Optimizer, batch | Adam, 32 | Như A |
| Learning rate | 0,0002; nhân 0,9 theo delr_freq 10 | Như A |
| Seed | 100 | 100 |
| JPEG augmentation | Tắt | Xác suất 0,5 mỗi lượt lấy ảnh |
| JPEG quality, codec | Không áp dụng | Số nguyên 50–95, chọn ngẫu nhiên OpenCV/PIL |
| Vị trí JPEG | Không áp dụng | Trước resize, ở kích thước ảnh gốc |
| Blur | Tắt | Tắt |
| Chọn checkpoint | Epoch50 cuối | Epoch50 cuối |

Tập train/val được đọc từ CNNDetection/extracted/train và val trên AutoDL.
Train resize 256 rồi random crop 224 và horizontal flip. Validation resize 256
rồi center crop 224. Loss BCEWithLogitsLoss. Chi tiết lịch LR trong code/train.py.
Mức JPEG là mức chất lượng encoder, không phải phần trăm dung lượng giữ lại.

## Đánh giá

62.003 ảnh của 8 nguồn: ProGAN 8000, StyleGAN 11982, StyleGAN2 15976,
BigGAN 4000, CycleGAN 2642, StarGAN 3998, GauGAN 10000, Deepfake 5405.
Cùng bộ ảnh và nhãn cho A/B; 4 điều kiện original/q90/q70/q50.
JPEG thử nghiệm PIL 4:2:0 ở độ phân giải ban đầu, nén trong RAM trước resize,
không ghi đè dataset. Test resize 256x256, không crop/flip, chuẩn hóa ImageNet.

Nhãn 0 = real, 1 = fake. Sigmoid(logit)> 0,5 là fake. AP tính bằng
sklearn.average_precision_score trên điểm fake, riêng từng nguồn. Mean là
trung bình cộng của 8 Acc hoặc 8 AP, không gộp tất cả ảnh để tính một AP.
Dòng Deepfake là phần riêng đã nằm trong Mean, không cộng vào Mean lần nữa.

## Trung bình 8 nguồn

| Điều kiện | A Acc / AP (%) | B Acc / AP (%) | Delta Acc / AP (điểm) |
|---|---:|---:|---:|
| Ảnh nguyên bản | 89,41 / 95,35 | 88,46 / 94,28 | -0,95 / -1,07 |
| q90 | 50,06 / 51,51 | 69,50 / 83,75 | +19,45 / +32,24 |
| q70 | 50,00 / 48,93 | 64,84 / 76,73 | +14,84 / +27,80 |
| q50 | 50,01 / 49,23 | 62,38 / 70,92 | +12,38 / +21,69 |

## Riêng Deepfake

| Điều kiện | A Acc / AP (%) | B Acc / AP (%) | Delta Acc / AP (điểm) |
|---|---:|---:|---:|
| Ảnh nguyên bản | 77,22 / 84,51 | 86,44 / 93,71 | +9,21 / +9,20 |
| q90 | 50,62 / 56,40 | 55,73 / 65,92 | +5,11 / +9,52 |
| q70 | 50,08 / 54,12 | 54,93 / 60,92 | +4,85 / +6,80 |
| q50 | 50,08 / 53,34 | 52,73 / 57,71 | +2,65 / +4,37 |

## Diễn giải cho báo cáo

JPEG augmentation giúp model B giữ được khả năng phân biệt tốt hơn A khi
đầu vào bị nén lại. Ở q90, Mean Acc tăng 19,45 điểm và Mean AP tăng 32,24 điểm.
Hiệu quả giảm dần khi JPEG mạnh hơn. Ngay cả B tại q50 vẫn chỉ đạt khoảng
62,38% Acc, nên không mô tả là đã giải quyết hoàn toàn ảnh nén.

Trên ảnh nguyên bản, B cải thiện riêng Deepfake 9,21 điểm Acc và 9,20 điểm AP,
nhưng Mean 8 nguồn giảm 0,95 điểm Acc và 1,07 điểm AP. Kết quả phù hợp với mục
tiêu độ bền JPEG trong lần thử này, chưa chứng minh tăng điểm trung bình của
Table 1 trên ảnh nguyên bản. Giả thuyết JPEG làm thay đổi dấu vết pixel có thể
giải thích đánh đổi giữa các nguồn, nhưng thí nghiệm này chưa xác định nguyên
nhân một cách riêng biệt.

Các giá trị trên là một lần train mỗi cấu hình, một seed; chưa có trung bình
nhiều seed hoặc khoảng tin cậy. Chưa kèm manifest/SHA từng ảnh để khẳng định
đã kiểm toán đầy đủ trùng lặp giữa train/validation/test. Validation ProGAN
rất cao không đảm bảo độ chính xác trên nguồn chưa thấy. Tập 8 nguồn này không
bao gồm Stable Diffusion; không suy rộng bảng này sang ảnh SD v1 của demo cũ.

## Phân biệt các checkpoint

- Model A nhóm train, dùng làm đối chứng: SHA-256 `84b1c6d44c93ecd13d374e7de681df446b1521fbc038429aedbe668fec717ec9`.
- Model B JPEG AutoDL được bàn giao: SHA-256 `d7b07f94411280632e0d2217810cc01ccf0c67650ee44ab98009f92df15f6354`.
- Model tác giả và model fine-tune pilot 160 ảnh thuộc thí nghiệm khác.
- Model CBAM/DIRE không thuộc gói kết quả này.

Trọng số A nằm ở nhánh [baseline/autodl-reproduction](https://github.com/Hafany1509/BTL-XLA/tree/baseline/autodl-reproduction/baseline/autodl_reproduction).
Không dùng mục “NPR gốc” trọng số tác giả trên web thay cho A khi giải thích
delta trong bảng này. Các CSV từng nguồn là số liệu gốc, không chỉnh tay hoặc
chọn ảnh theo dự đoán để nâng điểm. Delta tính từ số đầy đủ nên có thể lệch
0,01 so với phép trừ hai giá trị đã làm tròn trong bảng.
