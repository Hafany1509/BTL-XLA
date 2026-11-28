# Thí nghiệm và cải tiến NPR

## Bàn giao AutoDL đầy đủ

Xem [NPR + JPEG AutoDL](autodl_jpeg/README.md) để lấy checkpoint model B train từ
đầu 50 epoch, code train/test thực tế, cấu hình, log, bảng so sánh A/B trên 62.003
ảnh và hàm inference cho web. Phần dưới giữ lại thí nghiệm fine-tune pilot 160
ảnh trước đó; không trộn checkpoint, cấu hình hoặc số liệu giữa hai thí nghiệm.

Thành viên 2 phụ trách dữ liệu và huấn luyện.

1. Trong giai đoạn tái lập, chuẩn bị danh sách ảnh có nhãn, phân chia train/validation/test theo ảnh gốc và hỗ trợ kiểm tra tập test. Chuẩn bị script tạo bản JPEG ở quality 90, 70 và 50.
2. Sau khi nhóm nghiệm thu NPR gốc, fine-tune hai bản từ cùng checkpoint và cùng cấu hình. Dùng JPEG augmentation cho bản cải tiến và fine-tune thông thường cho bản đối chứng.
3. Dùng cùng công cụ đánh giá để đo NPR gốc, bản đối chứng và bản cải tiến trên ảnh gốc và từng mức nén. Bàn giao cấu hình, log, bảng Accuracy/AP và thông tin checkpoint cho frontend.

## Chuẩn bị ảnh cho thí nghiệm JPEG

Chạy trong repo nhóm bằng Python có Pillow. Ví dụ này lấy tối đa 20 ảnh cho mỗi cặp nguồn và nhãn từ bản bàn giao của Thành viên 1.

```powershell
$handover = '..\XLA\NPR-DeepfakeDetection'
python experiments/prepare_jpeg.py `
    --manifest "$handover/results/cleaned_black103/data_manifest.csv" `
    --image-root $handover `
    --output outputs/jpeg-pilot `
    --per-group 20
```

Script giữ nguyên mã ảnh, nguồn và nhãn qua bốn manifest `clean.csv`, `q90.csv`, `q70.csv` và `q50.csv`. Ba bản JPEG dùng cùng kích thước ảnh, chế độ RGB và lấy mẫu màu 4:2:0. Mỗi manifest JPEG ghi đường dẫn và SHA-256 của file mới. Kết quả trong `outputs/` được Git bỏ qua.

Khi dùng `evaluate_checkpoint.py` của Thành viên 1, đặt `--image-root` là thư mục bàn giao cho `clean.csv` và thư mục chứa các manifest JPEG cho `q90.csv`, `q70.csv` và `q50.csv`. Bỏ `--per-group` và chọn thư mục đầu ra mới khi cần tạo JPEG cho toàn bộ tập đánh giá.

## Fine-tune thử nghiệm

Tải `progan_val.zip` từ [kho CNNDetection của tác giả](https://huggingface.co/datasets/sywang/CNNDetection/tree/2dd6d95d4333188b8616f3d34970ea800ca1e339) vào `datasets/`. Script chia một phần gói validation thành 160 ảnh train và 40 ảnh validation từ `car` và `chair`. Trong gói này, 400 ảnh thật thuộc `cat` và `horse` trùng SHA-256 với bộ test bàn giao. Việc lọc theo SHA-256 phát hiện các tệp trùng byte.

Chạy các lệnh sau từ repo nhóm. Môi trường chuẩn bị dữ liệu cần Pillow. Môi trường huấn luyện cần thêm NumPy và PyTorch. Lệnh đánh giá dùng các phụ thuộc của Thành viên 1.

```powershell
$handover = '..\XLA\NPR-DeepfakeDetection'
$testManifest = "$handover/results/data_manifest.csv"
python experiments/prepare_finetune.py `
    --archive datasets/progan_val.zip `
    --test-manifest $testManifest `
    --output datasets/finetune-pilot

python -B experiments/finetune_npr.py `
    --image-root datasets/finetune-pilot `
    --train-manifest datasets/finetune-pilot/train.csv `
    --val-manifest datasets/finetune-pilot/val.csv `
    --test-manifest $testManifest `
    --output outputs/finetune-pilot

python -B experiments/evaluate_jpeg.py `
    --handover $handover `
    --conditions outputs/jpeg-pilot `
    --checkpoints outputs/finetune-pilot `
    --output outputs/jpeg-comparison-pilot
```

Hai lượt fine-tune dùng cùng checkpoint, thứ tự lấy mẫu, AdamW, learning rate `1e-5`, hai epoch và batch size 8. Bản JPEG nén ngẫu nhiên 50% lượt lấy ảnh ở một trong ba mức `q90/q70/q50`, áp dụng cho cả hai nhãn trước bước resize. Cả hai chọn checkpoint theo BCE thấp nhất trên cùng tập validation sạch. Log, cấu hình và checkpoint nằm trong thư mục fine-tune. Bảng Accuracy/AP theo mô hình, nguồn và mức nén nằm trong `comparison.csv`.

## Đánh giá trên 360 ảnh mới

Từ manifest test đã làm sạch, lấy 60 ảnh cho mỗi cặp nguồn và nhãn bằng seed 202. Loại ảnh thuộc lượt thử 120 ảnh theo cả mã ảnh và SHA-256. Dùng lại checkpoint đã chọn bằng tập validation và giữ ngưỡng dự đoán 0,5 của công cụ Thành viên 1.

```powershell
$handover = '..\XLA\NPR-DeepfakeDetection'
python -B experiments/prepare_jpeg.py `
    --manifest "$handover/results/cleaned_black103/data_manifest.csv" `
    --image-root $handover `
    --output outputs/jpeg-heldout `
    --per-group 60 --seed 202 `
    --exclude-manifest outputs/jpeg-pilot/clean.csv

python -B experiments/evaluate_jpeg.py `
    --handover $handover `
    --conditions outputs/jpeg-heldout `
    --checkpoints outputs/finetune-pilot `
    --output outputs/jpeg-comparison-heldout
```

Xem [tài liệu bàn giao](HANDOVER.md) để lấy kết quả, SHA-256 của hai checkpoint và ảnh minh họa cho demo.
