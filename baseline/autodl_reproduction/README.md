# Bàn giao NPR tự train trên AutoDL

Phần này chứa code thực tế tải từ AutoDL, model nhóm tự train, cấu hình,
log train/test và CSV kết quả. Code được điều chỉnh từ
[NPR-DeepfakeDetection](https://github.com/chuangchuangtan/NPR-DeepfakeDetection),
phương pháp của [paper NPR](https://arxiv.org/abs/2312.10461v2).
Các module NPR gốc cần để chạy model vẫn được giữ và ghi nguồn.
Checkpoint tác giả và kết quả đo trên checkpoint tác giả không thuộc bản bàn giao này.
Phần này độc lập với các thí nghiệm JPEG ở `experiments/`.

## File bàn giao

| Thành phần | Vị trí |
|---|---|
| Code đã chạy | `train.py`, `test.py`, `validate.py`, `data/`, `networks/`, `options/`, `util.py` |
| Chuẩn bị/kiểm tra data | `dataset_layout.py`, `prepare_autodl_data.sh` |
| Model nhóm tự train | [`models/model_epoch_last.pth`](models/model_epoch_last.pth) |
| Cấu hình lần train | [`config/train-options.txt`](config/train-options.txt) |
| Log đủ 50 epoch | [`results/trained-model-training.log`](results/trained-model-training.log) |
| Kết quả test | [`results/trained-model-forensynths.csv`](results/trained-model-forensynths.csv) |
| Log test | [`results/trained-model-test.log`](results/trained-model-test.log) |
| Hash model và code | `results/checkpoint-manifest.json`, `results/source-manifest.json` |
| Phiên bản cài lại môi trường | `requirements-autodl.txt` |

Code, `requirements.txt` gốc, cấu hình, log và CSV được lấy từ snapshot tải từ
instance AutoDL. Model được tải riêng từ checkpoint cuối của cùng run.
`requirements-autodl.txt` bổ sung từ các phiên bản đã ghi nhận khi cài đặt;
đây không phải bản xuất `pip freeze` đầy đủ. Dataset, cache, backup và log
chạy thử không đưa vào phần bàn giao này.

## Phạm vi và kết quả của nhóm

- Train từ đầu: ProGAN/LSUN, `car,cat,chair,horse`, 144.024 ảnh, 50 epoch.
- Validation: 1.600 ảnh; epoch 49 có Acc 99.75%, AP 99.9992203%.
- Test: ForenSynths, 8 generator, 62.003 ảnh.
- **Mean Accuracy: 89.4112%; Mean AP: 95.3477%.**
- Đã thực hiện Table 1; chưa có các bộ test của Table 2–5 trong phần này.

Mean là trung bình đều của 8 generator, không phải Accuracy gộp mọi ảnh.
Nhãn: `0_real = 0`, `1_fake = 1`. Test dùng toàn bộ category của mỗi generator;
4 category chỉ là cấu hình train/validation. Tiền xử lý test: resize bilinear
256×256, không crop, không flip, chuẩn hóa ImageNet; model ở chế độ `eval()`.
Accuracy dùng `sigmoid > 0.5`; AP tính từ điểm liên tục. Không chọn ngưỡng
trên tập test. Chưa kiểm tra trùng ảnh giữa các split bằng hash và chưa kết
luận đã khớp chính xác dữ liệu/protocol tạo bảng trong paper.

## Môi trường

Python 3.10, PyTorch 2.1.2+cu118, torchvision 0.16.2+cu118, NumPy 1.26.4,
OpenCV headless 4.10.0.84; AutoDL báo GPU NVIDIA GeForce RTX 4080 SUPER.
Seed 100, Adam, lr 0.0002, batch 32. Scheduler nhân lr với 0.9 sau các epoch
có chỉ số 10/20/30/40 (đếm từ 0). Cấu hình thực tế nằm trong `config/train-options.txt`.

Chạy từ thư mục repo nhóm:

```bash
cd baseline/autodl_reproduction
# Nếu AutoDL đã có đúng torch/torchvision CUDA thì giữ nguyên.
# Nếu tạo môi trường mới:
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-autodl.txt
```

NumPy 2.2.6 đã gây lỗi `torch.from_numpy` với môi trường này; lần chạy thành
công dùng NumPy 1.26.4. `requirements.txt` gốc không pin phiên bản, nên dùng
`requirements-autodl.txt` cho môi trường tái lập.

## Data

Data không nằm trong Git. Nguồn:
[sywang/CNNDetection](https://huggingface.co/datasets/sywang/CNNDetection/tree/main).
Đường dẫn ảnh đã dùng trên AutoDL:

```text
/root/autodl-tmp/CNNDetection/extracted/train/{car,cat,chair,horse}/{0_real,1_fake}/
/root/autodl-tmp/CNNDetection/extracted/val/{car,cat,chair,horse}/{0_real,1_fake}/
/root/autodl-tmp/CNNDetection/extracted/test/{generator}/...
```

Test có `progan,stylegan,stylegan2,biggan,cyclegan,stargan,gaugan,deepfake`.
ProGAN/StyleGAN/StyleGAN2/CycleGAN có thêm tầng category trước nhãn;
các generator còn lại chứa trực tiếp `0_real` và `1_fake`.

`prepare_autodl_data.sh` là script chuẩn bị archive từ lượt chạy: tạo ZIP
trung gian rồi giải nén toàn bộ train/validation, cần đủ dung lượng. Không
chạy lại nếu ảnh đã có. Máy cần `p7zip-full` và `unzip` để dùng script này.

## Kiểm tra và train lại

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

`--skip_test` chỉ bỏ benchmark trong lúc train; validation vẫn chạy mỗi epoch.
Có thể chỉ định ảnh bằng `--train_root` và `--val_root`. Checkpoint lưu trọng số,
không lưu đầy đủ optimizer/RNG; workflow `--continue_train` gốc chưa được sửa
để tiếp tục chính xác trạng thái huấn luyện.

## Test model bàn giao

Không cần train lại để test model kèm theo. Chạy từ `baseline/autodl_reproduction/`:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python test.py \
  --dataroot /root/autodl-tmp/CNNDetection/extracted/test \
  --model_path models/model_epoch_last.pth --batch_size 32 --num_threads 8 \
  --results_csv /root/autodl-tmp/reproduced-results.csv \
  2>&1 | tee /root/autodl-tmp/reproduced-test.log
```

Script snapshot là bản đã dùng cho các CSV/log kèm theo. Nó nạp strict trọng số
của model tự train; không cần checkpoint tác giả. CUDA GPU cần thiết vì
`validate.py` dùng `.cuda()`.

## Kiểm tra local

```bash
python -B -m unittest discover -s tests -v
bash -n prepare_autodl_data.sh
```

Unit test kiểm tra tìm đường dẫn và cấu trúc nhãn. Kết quả GPU trong bản bàn
giao là thực nghiệm AutoDL được ghi trong log; quá trình đóng gói local
không chạy lại huấn luyện hoặc suy luận GPU.
