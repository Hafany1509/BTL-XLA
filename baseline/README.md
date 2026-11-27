# Tái lập NPR gốc

Thư mục này chứa mã suy luận, đánh giá, notebook và kết quả tái lập NPR do Người 1 bàn giao ngày 12/11/2026. Mã của tác giả và checkpoint `NPR.pth` nằm trong submodule `upstream/npr/` (commit `781ced3`), nên không cần chép thêm.

## Kết quả đang dùng

Đánh giá trên 38.873 ảnh đã làm sạch (loại 103 ảnh đen toàn bộ trong sdv1/Fake), ngưỡng `fake_score > 0,5`.

| Nguồn | Real | Fake | Accuracy (%) | AP (%) | Bài báo Acc / AP |
|---|---:|---:|---:|---:|---|
| ProGAN | 4.000 | 4.000 | 99,79 | 99,99 | 99,8 / 100,0 |
| StyleGAN2 | 7.988 | 7.988 | 98,39 | 99,94 | 97,3 / 100,0 |
| sdv1 | 5.000 | 9.897 | 97,56 | 99,60 | 97,4 / 99,8 |

Checkpoint SHA-256 `3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53`.

## Chuẩn bị

1. Lấy submodule nếu chưa có:

   ```powershell
   git submodule update --init --recursive
   ```

2. Cài phụ thuộc. Bản đã kiểm tra nằm trong `requirements_person1.txt`. Toàn bộ môi trường máy Người 1 nằm trong `requirements_person1_environment.txt`.

   ```powershell
   python -m pip install -r baseline/requirements_person1.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```

3. Đặt dữ liệu tại `baseline/datasets/` với cấu trúc `progan/`, `stylegan2/`, `sdv1/`, tổng khoảng 7,7 GB. Thư mục này bị Git bỏ qua. Nếu dữ liệu đã nằm ở nơi khác, tạo liên kết thay vì sao chép:

   ```powershell
   New-Item -ItemType Junction -Path baseline\datasets -Target D:\duong\dan\datasets
   ```

## Chạy

Chạy từ thư mục `baseline/`.

Dự đoán một ảnh:

```powershell
python -X utf8 inference.py input_images/test.jpg
```

Kiểm tra nhanh 12 ảnh so với kết quả tham chiếu: làm theo bước 2 trang 3 của tài liệu bàn giao. Lệnh đánh giá không cần `--checkpoint` vì mặc định dùng `upstream/npr/NPR.pth`.

```powershell
python -X utf8 evaluate_checkpoint.py --manifest results/team_smoke/manifest.csv --output results/team_smoke/replay --batch-size 16 --threads 4
python -X utf8 compare_predictions.py --reference results/team_smoke/reference_predictions.csv --candidate results/team_smoke/replay/per_image_predictions.csv --output results/team_smoke/comparison --expect-identical
```

Kết quả đúng khi `comparison.json` có `status` là `PASS` và `paired_images` là 12. Đã kiểm tra PASS trong repo nhóm ngày 26/11/2026.

Đánh giá toàn bộ, khoảng 30 phút CPU:

```powershell
python -X utf8 evaluate_checkpoint.py --manifest results/cleaned_black103/data_manifest.csv --output results/member2_baseline --batch-size 16 --threads 4
```

Dùng trong mã Python, chẳng hạn backend demo:

```python
from inference import load_model, predict
model = load_model()            # mặc định NPR.pth của tác giả
result = predict("anh.jpg", model)  # label, fake_score, raw_logit, model_version, ...
```

## Nội dung thư mục

| Tệp | Nội dung |
|---|---|
| `inference.py` | Nạp checkpoint, tiền xử lý, `load_model()` và `predict()` |
| `evaluate_checkpoint.py`, `evaluation.py`, `compare_predictions.py` | Đánh giá theo manifest, tính chỉ số, đối chiếu hai lượt chạy |
| `prepare_clean_test.py`, `verify_clean_test.py`, `audit_image_quality.py` | Làm sạch tập test và kiểm tra |
| `reporting.py`, `inspect_provenance.py` | Vẽ hình, kiểm tra nguồn gốc checkpoint |
| `verify_*.py`, `complete_person1.py`, `finalize_person1.py`, `upgrade_handover_metadata.py`, `*_notebook.py` | Kiểm thử và tạo tài liệu bàn giao của Người 1 |
| `NPR_Reproduction_Complete.ipynb` | Notebook tái lập đầy đủ, đã có output |
| `results/cleaned_black103/` | Kết quả chính thức: manifest, dự đoán từng ảnh, chỉ số, hình |
| `results/` (tệp ở gốc) | Kết quả trước khi làm sạch và nhật ký kiểm thử |
| `HANDOVER_PERSON1.md`, `README_PERSON1.md` | Tài liệu gốc của Người 1, đường dẫn `D:\XLA\...` trong đó là máy của Người 1 |
| `smoke_npr.py` | Kiểm tra nhanh nạp checkpoint và forward |

Các đường dẫn `D:\XLA\...` trong JSON và log là thông tin lịch sử của lượt chạy gốc, không cần sửa.
