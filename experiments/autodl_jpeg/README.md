# NPR + JPEG AutoDL: bàn giao model B

> **Đã có checkpoint B đúng SHA-256 để dùng inference/web demo.** Code là bản kit đã triển khai, chưa nhận lại snapshot code đầy đủ từ AutoDL. Cấu hình ghi từ log trong phiên làm việc; log bàn giao hiện là trích đoạn hoàn thành epoch 50, chưa phải toàn bộ train.log.

Gói này phục vụ web demo và báo cáo của thí nghiệm train NPR từ đầu đủ 50 epoch,
có JPEG augmentation. Nó khác bản fine-tune pilot hai epoch ở thư mục experiments
cũ. Kiến trúc vẫn là NPR, không thêm CBAM hoặc DIRE.

## File cần lấy

| Thành phần | Đường dẫn | Công dụng |
|---|---|---|
| Model B epoch 50 | `models/npr_jpeg_50ep.pth` | Nạp trong web hoặc inference CPU |
| Inference | `inference.py` | `load_model()`, `predict()`, CLI, kiểm tra SHA-256 |
| Code kit đã triển khai | `code/` | Kiến trúc, data pipeline, train, resume, evaluator |
| Kết quả | `results/` | 8 CSV A/B và comparison.csv |
| Cấu hình | `config/experiment.json` | Cấu hình ghi nhận từ log của phiên train |
| Log | `logs/training-completion-excerpt.log` | Trích đoạn log người dùng gửi, xác nhận hoàn thành epoch 50 |
| Báo cáo | `REPORT.md` | Phương pháp, bảng kết quả, cách diễn giải và giới hạn |
| Nguồn gốc | `handover-manifest.json` | Hash checkpoint/source/artifact; trạng thái xác minh |

SHA-256 model B: `d7b07f94411280632e0d2217810cc01ccf0c67650ee44ab98009f92df15f6354`.
Hàm nạp mặc định kiểm tra đúng hash này, tránh lấy nhầm model A hoặc pilot.
Không cần checkpoint tác giả hoặc submodule upstream để dùng riêng model B.

## Lấy nhánh và chạy demo

```powershell
git clone --branch experiments/jpeg-autodl https://github.com/Hafany1509/BTL-XLA.git
cd BTL-XLA
```

Nếu đã clone repo, commit/stash phần đang sửa trước khi chuyển nhánh:

```powershell
git fetch origin
git switch --track origin/experiments/jpeg-autodl
```

Môi trường AutoDL đã chạy: Python 3.10, torch 2.1.2+cu118,
torchvision 0.16.2+cu118, NumPy 1.26.4, Pillow 10.3.0. Khi tạo môi trường CPU mới,
dùng Python 3.10 và cặp torch/torchvision tương ứng:

```powershell
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 Pillow==10.3.0
python -m pip install -r frontend/requirements.txt
python -m uvicorn frontend.app:app --port 8000
```

Mở http://localhost:8000 và chọn **NPR + JPEG AutoDL** (`jpeg_autodl`).
Chọn **giữ nguyên** để không nén thêm đầu vào; q90/q70/q50 là thí nghiệm chịu nén.
Các model pilot và trọng số tác giả hiển thị riêng nếu có đủ file.
Mục **NPR gốc** trên web là model tác giả; nó không phải model A do nhóm train
dùng để tính chênh lệch A/B trong `results/comparison.csv`.

## Dùng hàm inference cho backend khác

Chạy từ gốc repo:

```python
from experiments.autodl_jpeg.inference import load_model

detector = load_model()  # nạp một lần, tái sử dụng cho các request
result = detector.predict("anh.jpg")
print(result["label"], result["fake_score"], result["checkpoint_sha256"])
```

Hoặc CLI:

```powershell
python experiments/autodl_jpeg/inference.py anh.jpg
python experiments/autodl_jpeg/inference.py anh.jpg --jpeg-quality 90
```

Đầu vào: đường dẫn ảnh hoặc PIL.Image.Image. Đầu ra: `label` REAL/FAKE,
`fake_score`, `raw_logit`, `model_version`, `checkpoint_sha256`, thời gian CPU.
Fake khi score > 0.5, score = 0.5 được xếp REAL. Điểm chưa được hiệu chỉnh thành
độ tin cậy. Không thể suy ra nguồn sinh ảnh chỉ từ nhãn REAL/FAKE.
Không tự động xoay EXIF. RGB, resize bilinear 256x256, không crop/flip khi test,
ImageNet normalize. NPR tự tính residual bên trong mạng.
JPEG test dùng PIL 4:2:0 ở kích thước ảnh ban đầu trước resize.
Khi web đã nén trước khi gọi predict(rgb), không truyền jpeg_quality thêm lần nữa.

## Kết quả chính

Mỗi ô A/B là Acc/AP (%), Mean là trung bình không trọng số của 8 nguồn.
Delta là B trừ A theo điểm phần trăm, tính trước làm tròn.

| Điều kiện | A: NPR | B: NPR + JPEG | Delta Acc / AP |
|---|---:|---:|---:|
| Ảnh nguyên bản | 89,41 / 95,35 | 88,46 / 94,28 | -0,95 / -1,07 |
| q90 | 50,06 / 51,51 | 69,50 / 83,75 | +19,45 / +32,24 |
| q70 | 50,00 / 48,93 | 64,84 / 76,73 | +14,84 / +27,80 |
| q50 | 50,01 / 49,23 | 62,38 / 70,92 | +12,38 / +21,69 |

B cải thiện khả năng chịu nén trong lần chạy này; Mean ảnh nguyên bản giảm
khoảng 0,95 điểm Acc và 1,07 điểm AP. Không gọi đây là tăng toàn bộ Table1 gốc.
Riêng Deepfake trên ảnh nguyên bản tăng từ 77,22/84,51 lên 86,44/93,71.
Xem REPORT.md để có bảng Deepfake đầy đủ và phương pháp thí nghiệm.

## Chạy lại đánh giá trên AutoDL

Data không đưa vào Git. Giữ nguyên 8 nguồn trong bộ test đã giải nén:

```bash
cd /root/autodl-tmp/NPR-JPEG
.venv/bin/python -u code/test.py --dataroot /root/autodl-tmp/CNNDetection/extracted/test --model_path checkpoints/B-jpeg/model_epoch_last.pth --batch_size 32 --num_threads 8 --test_jpeg_quality 0 --results_csv results/B_original.csv
```

Các script env.sh/run.sh/setup.sh lấy từ kit đã triển khai; runner dùng checkpoint cuối. Không có bước chọn best trong bản bàn giao này. Đọc và sửa
đường dẫn trong env.sh khi chuyển máy. Lệnh test-A/test-B của run.sh chạy cả
original/q90/q70/q50. Checkpoint chính là epoch 50 cuối, không chọn bằng test.
Full optimizer/RNG resume state và dataset không nằm trong gói bàn giao này.

Nguồn: [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection),
[paper NPR](https://arxiv.org/abs/2312.10461).
