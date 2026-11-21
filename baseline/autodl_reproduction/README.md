# NPR: train lại và đánh giá ForenSynths trên AutoDL

Mã được điều chỉnh từ [NPR-DeepfakeDetection của tác giả](https://github.com/chuangchuangtan/NPR-DeepfakeDetection)
cho [paper NPR, arXiv:2312.10461v2](https://arxiv.org/abs/2312.10461v2).
Các module mạng và biến đổi ảnh được giữ từ bản mã nguồn đã dùng trong lần chạy.
Phần này độc lập với các thí nghiệm JPEG trong `experiments/`.

## Phạm vi và kết quả

Đã train từ đầu bằng ProGAN/LSUN, 4 category `car,cat,chair,horse`,
và đánh giá trên 8 generator của ForenSynths (Table 1).
Chưa tái lập các bộ test GAN/Diffusion ở Table 2–5.

| Nguồn trọng số | Mean Accuracy (%) | Mean AP (%) | Bằng chứng |
|---|---:|---:|---|
| Model nhóm tự train, epoch 50 | 89.4112 | 95.3477 | [CSV](results/trained-model-forensynths.csv), [log](results/trained-model-test.log) |
| Checkpoint tác giả `NPR.pth` | 92.3019 | 94.2229 | [CSV](results/author-npr-forensynths.csv) |
| Paper v2, Table 1 | 92.5 | 96.1 | Số liệu công bố, không phải kết quả nhóm tự đo |

Mean là trung bình không trọng số của 8 generator, không phải Accuracy
gộp toàn bộ ảnh. Tổng test có 62.003 ảnh. ProGAN test dùng toàn bộ category;
không giới hạn 4 category như tập train. Nhãn thật là 0, giả là 1.
Ngưỡng là **sigmoid > 0.5**; AP dùng điểm liên tục, không threshold.
Resize 256×256 bằng bilinear, không crop/flip khi test; chuẩn hóa ImageNet.

Kết quả checkpoint tác giả phải báo cáo riêng với kết quả model tự train.
Việc chạy thành công không xác nhận đã tái lập chính xác số liệu paper.
Không điều chỉnh threshold bằng tập test. Chưa có kiểm tra SHA-256 giữa các
tập ảnh để kết luận về trùng lặp dữ liệu; test đã qua kiểm tra thư mục/nhãn
và suy luận toàn bộ ảnh, không có báo cáo integrity/dedup riêng.

## Môi trường đã dùng

- Python 3.10; PyTorch 2.1.2+cu118, torchvision 0.16.2+cu118.
- NumPy 1.26.4; OpenCV headless 4.10.0.84. NumPy 2.2.6 trong môi trường ban đầu
  gây lỗi `torch.from_numpy`; không dùng cấu hình đó cho lượt chạy thành công.
- AutoDL báo GPU NVIDIA GeForce RTX 4080 SUPER.
- Seed train/test: 100; Adam, lr 0.0002, batch 32, 50 epoch;
  lr nhân 0.9 sau các epoch có chỉ số 10/20/30/40 (đếm từ 0).
- Validation: 1.600 ảnh; train loader ghi nhận 144.024 ảnh.
- Validation cuối ghi trong phiên chạy: Accuracy 99.75%, AP khoảng 99.9992%.
  Full training log không có trong các file local được bàn giao lên Git.

## Cài đặt

Chạy từ thư mục repo nhóm:

```bash
cd baseline/autodl_reproduction
# Nếu AutoDL đã có đúng torch/torchvision CUDA thì giữ nguyên.
# Nếu tạo môi trường mới, cài cặp CUDA 11.8 trước:
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
apt-get update
apt-get install -y p7zip-full unzip tmux
```

Trọng số và dataset không được đưa vào Git. Dữ liệu từ
[sywang/CNNDetection](https://huggingface.co/datasets/sywang/CNNDetection/tree/main);
xem [hướng dẫn AutoDL](README_AUTODL.md) và `prepare_autodl_data.sh` cho archive
train/validation. Script chuẩn bị giải nén cả archive, giữ bản ZIP trung gian;
cần kiểm tra dung lượng trước khi chạy. Không chạy lại nếu ảnh đã được giải nén.

## Kiểm tra và train

```bash
python train.py --dataroot /root/autodl-tmp/CNNDetection \
  --classes car,cat,chair,horse --batch_size 32 --num_threads 0 --skip_test --check_data

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python train.py \
  --dataroot /root/autodl-tmp/CNNDetection \
  --classes car,cat,chair,horse --name NPR-4class- \
  --batch_size 32 --num_threads 8 --lr 0.0002 --delr_freq 10 --niter 50 \
  --loss_freq 100 --save_latest_freq 2000 --save_epoch_freq 1 \
  --checkpoints_dir /root/autodl-tmp/checkpoints --skip_test
```

Đường dẫn có thể chỉ định trực tiếp bằng `--train_root` và `--val_root`.
`--skip_test` không bỏ validation; nó chỉ bỏ benchmark ngoài train/validation.
Các checkpoint có định dạng trọng số thuần. Không dùng `--continue_train`
để kỳ vọng tiếp tục chính xác optimizer/RNG: workflow resume gốc chưa được sửa.

## Test

Thư mục test phải trực tiếp chứa `progan,stylegan,stylegan2,biggan,cyclegan,stargan,gaugan,deepfake`.
Chạy từ `baseline/autodl_reproduction/`, sửa `MODEL` theo nơi lưu trọng số:

```bash
MODEL=/root/autodl-tmp/checkpoints/NPR-4class-2026_11_21_05_53_01/model_epoch_last.pth
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python test.py \
  --dataroot /root/autodl-tmp/CNNDetection/extracted/test \
  --model_path "$MODEL" --batch_size 32 --num_threads 8 \
  --results_csv /root/autodl-tmp/trained-model-forensynths.csv \
  2>&1 | tee /root/autodl-tmp/trained-model-test.log
```

Để đối chiếu checkpoint tác giả, dùng `--model_path` trỏ tới
`upstream/npr/NPR.pth` của repo nhóm (sau `git submodule update --init --recursive`)
và đặt tên CSV/log riêng. Runner hỗ trợ trọng số thuần, wrapper `model`/
`state_dict`, và bỏ tiền tố DataParallel `module.` trước khi nạp strict.
Kết quả CSV tác giả đã lưu được tạo bằng một bản `NPR.pth` bỏ tiền tố này;
việc bỏ tiền tố không thay đổi giá trị trọng số.

## Kiểm tra mã

```bash
python -B -m unittest discover -s tests -v
bash -n prepare_autodl_data.sh
```

Các test local kiểm tra đường dẫn, nhãn thư mục và chuẩn hóa tên checkpoint,
không thay thế chạy GPU với dataset thật. CSV/log đi kèm là kết quả AutoDL
đã chạy; thời gian test phụ thuộc máy và I/O.

## Bàn giao model

Model tự train hiện được lưu ngoài Git; SHA-256 và tên file nằm trong
`results/checkpoint-manifest.json`. Thành viên có model cần chuyển riêng
qua kho lưu trữ của nhóm và kiểm tra hash trước khi dùng. Chưa có URL tải
model tự train được công bố trong nhánh này.
